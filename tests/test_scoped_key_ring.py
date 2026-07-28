import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
import access
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

    def test_rejects_symlinked_key_ring(self):
        symlink = os.path.join(self.tempdir.name, "keys-link.json")
        os.symlink(self.key_file, symlink)
        with self.assertRaisesRegex(KeyRingConfigurationError, "must not be a symlink"):
            load_key_ring({"PALACE_API_KEYS_FILE": symlink})

    def test_key_ring_cache_reuses_unchanged_file_and_reloads_rotation(self):
        access._key_ring_cache.clear()
        with patch("access.json.loads", wraps=json.loads) as loads:
            self.assertEqual(load_key_ring(self.env)[0].name, "reader")
            self.assertEqual(load_key_ring(self.env)[0].name, "reader")
            self.assertEqual(loads.call_count, 1)

            rotated = os.path.join(self.tempdir.name, "rotated.json")
            with open(rotated, "w", encoding="utf-8") as handle:
                json.dump({"keys": [
                    {"name": "rotated", "key": "rotated-secret-012345", "operations": ["read"], "wings": ["alpha"]},
                ]}, handle)
            os.chmod(rotated, 0o600)
            os.replace(rotated, self.key_file)

            self.assertEqual(load_key_ring(self.env)[0].name, "rotated")
            self.assertEqual(loads.call_count, 2)

    def test_key_ring_read_rejects_file_replaced_after_validation(self):
        expected = access._require_safe_file(Path(self.key_file))
        replacement = os.path.join(self.tempdir.name, "replacement.json")
        with open(replacement, "w", encoding="utf-8") as handle:
            handle.write('{"keys": []}')
        os.chmod(replacement, 0o600)
        os.replace(replacement, self.key_file)

        self.assertIsNone(access._read_safe_key_ring(Path(self.key_file), expected))


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
        self.env = {"PALACE_API_KEYS_FILE": self.key_file}
        self.environ = patch.dict(
            os.environ,
            {**self.env, "PALACE_API_KEY": ""},
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

    def test_search_and_context_forward_the_authorized_wing(self):
        for path in ["/search?q=x&wing=alpha", "/context?topic=x&wing=alpha"]:
            with self.subTest(path=path):
                self.call.reset_mock()
                response = self.request("GET", path, "reader-secret-012345")
                self.assertEqual(response.status_code, 200)
                arguments = self.call.await_args.args[0]["params"]["arguments"]
                self.assertEqual(arguments["wing"], "alpha")

    def test_unknown_mcp_tool_requires_admin(self):
        body = {"params": {"name": "mempalace_future_write", "arguments": {"wing": "alpha"}}}
        self.assertEqual(main._mcp_policy(body)[0], "admin")
        self.assertEqual(
            self.request("POST", "/mcp", "writer-secret-012345", json=body).status_code,
            403,
        )

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

    def test_mcp_import_and_maintenance_tools_require_admin(self):
        for tool_name in [
            "mempalace_mine",
            "mempalace_sync",
            "mempalace_reconnect",
            "mempalace_memories_filed_away",
            "mempalace_hook_settings",
        ]:
            with self.subTest(tool_name=tool_name):
                body = {"params": {"name": tool_name, "arguments": {"wing": "alpha"}}}
                self.assertEqual(main._mcp_policy(body)[0], "admin")
                self.assertEqual(
                    self.request("POST", "/mcp", "writer-secret-012345", json=body).status_code,
                    403,
                )

    def test_checkpoint_is_scoped_write_but_other_maintenance_tools_remain_admin(self):
        checkpoint = {"params": {"name": "mempalace_checkpoint", "arguments": {
            "items": [{"wing": "alpha"}],
        }}}
        self.assertEqual(main._mcp_policy(checkpoint), ("write", ("alpha",)))
        self.assertEqual(
            self.request("POST", "/mcp", "writer-secret-012345", json=checkpoint).status_code,
            200,
        )
        maintenance = {"params": {"name": "mempalace_sync", "arguments": {}}}
        self.assertEqual(main._mcp_policy(maintenance)[0], "admin")
        self.assertEqual(
            self.request("POST", "/mcp", "writer-secret-012345", json=maintenance).status_code,
            403,
        )

    def test_silent_save_and_digest_omitted_wing_require_unrestricted_write(self):
        for path in ["/silent-save", "/digest"]:
            with self.subTest(path=path):
                operation, wing = main._request_policy("POST", path, {}, {})
                self.assertEqual(operation, "write")
                self.assertIsNone(wing)
                self.assertEqual(
                    authorize("operator-secret-012345", operation, wing, self.env), "operator"
                )
                with self.assertRaises(AuthorizationError):
                    authorize("writer-secret-012345", operation, wing, self.env)
                self.assertEqual(
                    self.request("POST", path, "writer-secret-012345", json={"entry": "ok"}).status_code,
                    403,
                )
                if path == "/silent-save":
                    with patch.object(
                        main, "_do_silent_save_write", new=AsyncMock(return_value={"success": True})
                    ):
                        response = self.request(
                            "POST", path, "operator-secret-012345", json={"entry": "ok"}
                        )
                    self.assertEqual(response.status_code, 200)
                else:
                    with (
                        patch.object(main, "ANTHROPIC_API_KEY", "test-key"),
                        patch.object(main, "_anthropic", object()),
                        patch.object(main, "_run_digest", new=AsyncMock()),
                    ):
                        response = self.request(
                            "POST", path, "operator-secret-012345", json={"messages": []}
                        )
                    self.assertEqual(response.status_code, 202)


class ProtectedWingPermissionTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.key_file = os.path.join(self.tempdir.name, "keys.json")
        with open(self.key_file, "w", encoding="utf-8") as handle:
            json.dump({"keys": [
                {
                    "name": "kit",
                    "key": "kit-secret-0123456789",
                    "permissions": {
                        "read": {"allow": ["*"]},
                        "write": {"allow": ["*"], "deny": ["wing_wren"]},
                    },
                },
                {
                    "name": "wren",
                    "key": "wren-secret-012345678",
                    "permissions": {
                        "read": {"allow": ["*"]},
                        "write": {"allow": ["*"], "deny": ["wing_kit", "carpe", "wing_mined"]},
                    },
                },
            ]}, handle)
        os.chmod(self.key_file, 0o600)
        self.env = {"PALACE_API_KEYS_FILE": self.key_file}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_per_operation_protected_wing_rules(self):
        self.assertEqual(authorize("kit-secret-0123456789", "read", "wing_wren", self.env), "kit")
        self.assertEqual(authorize("kit-secret-0123456789", "write", "general", self.env), "kit")
        self.assertEqual(authorize("wren-secret-012345678", "write", "general", self.env), "wren")
        for key, wing in [
            ("kit-secret-0123456789", "wing_wren"),
            ("wren-secret-012345678", "wing_kit"),
            ("wren-secret-012345678", "carpe"),
            ("wren-secret-012345678", "wing_mined"),
        ]:
            with self.subTest(key=key, wing=wing), self.assertRaises(AuthorizationError):
                authorize(key, "write", wing, self.env)
        with self.assertRaises(AuthorizationError):
            authorize("kit-secret-0123456789", "write", None, self.env)
        with self.assertRaises(AuthorizationError):
            authorize("kit-secret-0123456789", "admin", None, self.env)

    def test_multi_wing_operation_requires_every_target_to_be_allowed(self):
        self.assertEqual(
            authorize("kit-secret-0123456789", "write", ("general", "shared"), self.env), "kit"
        )
        with self.assertRaises(AuthorizationError):
            authorize("kit-secret-0123456789", "write", ("general", "wing_wren"), self.env)

    def test_rejects_ambiguous_permission_rules(self):
        invalid_permissions = [
            {"write": {"allow": ["*"]}, "unknown": {"allow": ["*"]}},
            {"write": {"allow": ["*"], "deny": ["*"]}},
            {"write": {"allow": ["alpha"], "deny": ["alpha"]}},
            {"write": {"allow": ["alpha", "alpha"]}},
        ]
        for permissions in invalid_permissions:
            with self.subTest(permissions=permissions):
                with open(self.key_file, "w", encoding="utf-8") as handle:
                    json.dump({"keys": [{
                        "name": "invalid", "key": "invalid-secret-012345", "permissions": permissions,
                    }]}, handle)
                os.chmod(self.key_file, 0o600)
                access._key_ring_cache.clear()
                with self.assertRaises(KeyRingConfigurationError):
                    load_key_ring(self.env)


class ProtectedWingRouteAuthorizationTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.key_file = os.path.join(self.tempdir.name, "keys.json")
        with open(self.key_file, "w", encoding="utf-8") as handle:
            json.dump({"keys": [
                {
                    "name": "kit", "key": "kit-secret-0123456789",
                    "permissions": {
                        "read": {"allow": ["*"]},
                        "write": {"allow": ["*"], "deny": ["wing_wren"]},
                    },
                },
                {
                    "name": "wren", "key": "wren-secret-012345678",
                    "permissions": {
                        "read": {"allow": ["*"]},
                        "write": {"allow": ["*"], "deny": ["wing_kit", "carpe", "wing_mined"]},
                    },
                },
            ]}, handle)
        os.chmod(self.key_file, 0o600)
        self.environ = patch.dict(os.environ, {"PALACE_API_KEYS_FILE": self.key_file, "PALACE_API_KEY": ""}, clear=False)
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

    def test_http_allows_reads_but_denies_protected_writes(self):
        self.assertEqual(self.request("GET", "/stats", "kit-secret-0123456789").status_code, 200)
        self.assertEqual(
            self.request("POST", "/memory", "kit-secret-0123456789", json={"wing": "general", "content": "ok"}).status_code,
            200,
        )
        self.assertEqual(
            self.request("POST", "/memory", "kit-secret-0123456789", json={"wing": "wing_wren", "content": "no"}).status_code,
            403,
        )
        for wing in ("wing_kit", "carpe", "wing_mined"):
            with self.subTest(wing=wing):
                self.assertEqual(
                    self.request("POST", "/silent-save", "wren-secret-012345678", json={"wing": wing, "entry": "no"}).status_code,
                    403,
                )

    def test_mcp_enforces_single_and_multi_wing_writes(self):
        add_drawer = {"params": {"name": "mempalace_add_drawer", "arguments": {"wing": "wing_wren", "room": "x", "content": "no"}}}
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=add_drawer).status_code, 403)
        add_drawer["params"]["arguments"]["wing"] = "general"
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=add_drawer).status_code, 200)

        tunnel = {"params": {"name": "mempalace_create_tunnel", "arguments": {
            "source_wing": "general", "source_room": "a", "target_wing": "wing_wren", "target_room": "b",
        }}}
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=tunnel).status_code, 403)
        tunnel["params"]["arguments"]["target_wing"] = "shared"
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=tunnel).status_code, 200)

    def test_mcp_write_without_a_determinable_wing_fails_closed(self):
        body = {"params": {"name": "mempalace_kg_add", "arguments": {"subject": "x", "predicate": "y", "object": "z"}}}
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=body).status_code, 403)

    def test_checkpoint_authorizes_every_item_and_diary_wing(self):
        body = {"params": {"name": "mempalace_checkpoint", "arguments": {
            "items": [{"wing": "general"}, {"wing": "shared"}],
            "diary": {"wing": "general"},
        }}}
        self.assertEqual(main._mcp_policy(body), ("write", ("general", "shared", "general")))
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=body).status_code, 200)

        body["params"]["arguments"]["diary"]["wing"] = "wing_wren"
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=body).status_code, 403)
        body["params"]["arguments"]["diary"]["wing"] = "general"
        body["params"]["arguments"]["items"][1]["wing"] = "wing_wren"
        self.assertEqual(self.request("POST", "/mcp", "kit-secret-0123456789", json=body).status_code, 403)

    def test_checkpoint_missing_or_malformed_wing_scope_fails_closed(self):
        cases = [
            {},
            {"items": []},
            {"items": [{}]},
            {"items": [{"wing": "general"}], "diary": {}},
            {"items": [{"wing": "general"}], "diary": []},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                body = {"params": {"name": "mempalace_checkpoint", "arguments": arguments}}
                self.assertEqual(main._mcp_policy(body), ("write", None))
                self.assertEqual(
                    self.request("POST", "/mcp", "kit-secret-0123456789", json=body).status_code,
                    403,
                )


if __name__ == "__main__":
    unittest.main()
