import asyncio
import json
import os
import stat
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from access import AuthorizationError, KeyRingConfigurationError, authorize, load_key_ring


class KeyRingTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.key_file = os.path.join(self.tempdir.name, "keys.json")
        with open(self.key_file, "w", encoding="utf-8") as handle:
            json.dump({"keys": [
                {"name": "reader", "key": "reader-secret-012345", "operations": ["read"], "wings": ["alpha"]},
                {"name": "writer", "key": "writer-secret-012345", "operations": ["write"], "wings": ["alpha"]},
                {"name": "operator", "key": "operator-secret-012345", "operations": ["read", "write", "admin"], "wings": ["*"]},
            ]}, handle)
        os.chmod(self.key_file, 0o600)
        self.env = {"PALACE_API_KEYS_FILE": self.key_file}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_named_independent_keys_and_scopes(self):
        self.assertEqual(authorize("reader-secret-012345", "read", "alpha", self.env), "reader")
        self.assertEqual(authorize("writer-secret-012345", "write", "alpha", self.env), "writer")
        self.assertEqual(authorize("operator-secret-012345", "admin", None, self.env), "operator")

    def test_denies_wrong_key_operation_wing_and_unscoped_read(self):
        for key, operation, wing in [
            ("unknown-secret-012345", "read", "alpha"),
            ("reader-secret-012345", "write", "alpha"),
            ("reader-secret-012345", "read", "beta"),
            ("reader-secret-012345", "read", None),
        ]:
            with self.assertRaises(AuthorizationError):
                authorize(key, operation, wing, self.env)

    def test_legacy_single_key_migrates_to_full_access_grant(self):
        env = {"PALACE_API_KEY": "legacy-secret-012345"}
        grant = load_key_ring(env)[0]
        self.assertEqual(grant.name, "legacy-palace-api-key")
        self.assertEqual(authorize("legacy-secret-012345", "admin", None, env), grant.name)

    def test_rejects_unsafe_and_ambiguous_config(self):
        os.chmod(self.key_file, 0o644)
        with self.assertRaises(KeyRingConfigurationError):
            load_key_ring(self.env)
        os.chmod(self.key_file, 0o600)
        with self.assertRaises(KeyRingConfigurationError):
            load_key_ring({"PALACE_API_KEYS_FILE": self.key_file, "PALACE_API_KEY": "legacy-secret-012345"})


class ScopedRouteAuthorizationTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.key_file = os.path.join(self.tempdir.name, "keys.json")
        with open(self.key_file, "w", encoding="utf-8") as handle:
            json.dump({"keys": [
                {"name": "reader", "key": "reader-secret-012345", "operations": ["read"], "wings": ["alpha"]},
                {"name": "writer", "key": "writer-secret-012345", "operations": ["write"], "wings": ["alpha"]},
                {"name": "operator", "key": "operator-secret-012345", "operations": ["read", "write", "admin"], "wings": ["*"]},
            ]}, handle)
        os.chmod(self.key_file, stat.S_IRUSR | stat.S_IWUSR)
        self.environ = patch.dict(
            os.environ,
            {"PALACE_API_KEYS_FILE": self.key_file, "PALACE_API_KEY": ""},
            clear=False,
        )
        self.environ.start()
        self.call = AsyncMock(return_value={"result": {"content": [{"text": "{}"}]}})
        self.call_patch = patch.object(main, "_call", self.call)
        self.call_patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        self.call_patch.stop()
        self.environ.stop()
        self.tempdir.cleanup()

    def request(self, method, path, key, **kwargs):
        return self.client.request(method, path, headers={"X-Api-Key": key}, **kwargs)

    def test_every_existing_route_has_an_explicit_policy(self):
        cases = [
            ("GET", "/health", "read"), ("GET", "/search", "read"),
            ("GET", "/context", "read"), ("GET", "/list", "read"),
            ("GET", "/stats", "read"), ("GET", "/graph", "read"),
            ("GET", "/viz", "read"), ("GET", "/repair/status", "read"),
            ("POST", "/mcp", "read"), ("POST", "/memory", "write"),
            ("POST", "/silent-save", "write"), ("POST", "/digest", "write"),
            ("POST", "/mine", "admin"), ("PATCH", "/memory/example", "write"),
            ("DELETE", "/memory/example", "admin"), ("POST", "/flush", "admin"),
            ("POST", "/reload", "admin"), ("POST", "/backup", "admin"),
            ("POST", "/repair", "admin"),
        ]
        mcp = {"params": {"name": "mempalace_search", "arguments": {"wing": "alpha"}}}
        for method, path, operation in cases:
            body = mcp if path == "/mcp" else {"wing": "alpha"}
            actual, _ = main._request_policy(method, path, {"wing": "alpha"}, body)
            self.assertEqual(actual, operation, f"{method} {path}")

    def test_read_key_is_limited_to_its_named_wing(self):
        with self.assertLogs("palace-daemon", level="INFO") as audit_logs:
            response = self.request("GET", "/search?q=x&wing=alpha", "reader-secret-012345")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any("identity=reader" in line for line in audit_logs.output))
        self.assertEqual(self.request("GET", "/search?q=x&wing=beta", "reader-secret-012345").status_code, 403)
        self.assertEqual(self.request("GET", "/search?q=x", "reader-secret-012345").status_code, 403)
        self.assertEqual(self.request("GET", "/stats", "reader-secret-012345").status_code, 403)

    def test_write_and_admin_route_denials(self):
        self.assertEqual(self.request("POST", "/memory", "writer-secret-012345", json={"wing": "alpha", "content": "ok"}).status_code, 200)
        self.assertEqual(self.request("POST", "/memory", "writer-secret-012345", json={"wing": "beta", "content": "no"}).status_code, 403)
        self.assertEqual(self.request("DELETE", "/memory/example", "writer-secret-012345").status_code, 403)
        self.assertEqual(self.request("POST", "/reload", "operator-secret-012345").status_code, 200)

    def test_mcp_unknown_target_fails_closed_for_scoped_key(self):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "mempalace_kg_stats", "arguments": {}}}
        self.assertEqual(self.request("POST", "/mcp", "reader-secret-012345", json=body).status_code, 403)
        body["params"] = {"name": "mempalace_search", "arguments": {"query": "x", "wing": "alpha"}}
        self.assertEqual(self.request("POST", "/mcp", "reader-secret-012345", json=body).status_code, 200)


if __name__ == "__main__":
    unittest.main()
