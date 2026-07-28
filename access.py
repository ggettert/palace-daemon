"""Small, dependency-free authorization layer for palace-daemon."""
from __future__ import annotations

import hmac
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

VALID_OPERATIONS = frozenset({"read", "write", "admin"})


class KeyRingConfigurationError(ValueError):
    """The key-ring configuration is malformed or unsafe to read."""


class AuthorizationError(PermissionError):
    """An authenticated caller is not allowed to perform a request."""


@dataclass(frozen=True)
class KeyGrant:
    """A named opaque token and its least-privilege grant."""

    name: str
    secret: str
    operations: frozenset[str]
    wings: frozenset[str]

    @property
    def unrestricted_wings(self) -> bool:
        return "*" in self.wings


@dataclass(frozen=True)
class _KeyRingFileState:
    """Metadata that invalidates a cached key ring after a safe rotation."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    uid: int


_key_ring_cache: dict[Path, tuple[_KeyRingFileState, tuple[KeyGrant, ...]]] = {}


def _require_safe_file(path: Path) -> _KeyRingFileState:
    try:
        # lstat is intentional: a key-ring path must never redirect through a
        # symlink, even when its target is an otherwise-safe regular file.
        info = path.lstat()
    except OSError as exc:
        raise KeyRingConfigurationError(f"cannot lstat PALACE_API_KEYS_FILE: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise KeyRingConfigurationError("PALACE_API_KEYS_FILE must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise KeyRingConfigurationError("PALACE_API_KEYS_FILE must be a regular file")
    if info.st_uid != os.geteuid():
        raise KeyRingConfigurationError("PALACE_API_KEYS_FILE must be owned by the daemon user")
    if info.st_mode & 0o077:
        raise KeyRingConfigurationError(
            "PALACE_API_KEYS_FILE must not be readable or writable by group/other (chmod 600)"
        )
    return _KeyRingFileState(
        device=info.st_dev,
        inode=info.st_ino,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        mode=stat.S_IMODE(info.st_mode),
        uid=info.st_uid,
    )


def _parse_key_ring(raw: object) -> tuple[KeyGrant, ...]:
    if not isinstance(raw, dict) or set(raw) != {"keys"} or not isinstance(raw["keys"], list):
        raise KeyRingConfigurationError('key-ring JSON must be exactly {"keys": [...]}')
    if not raw["keys"]:
        raise KeyRingConfigurationError("key-ring must contain at least one key")

    grants: list[KeyGrant] = []
    names: set[str] = set()
    secrets: set[str] = set()
    for entry in raw["keys"]:
        if not isinstance(entry, dict) or set(entry) != {"name", "key", "operations", "wings"}:
            raise KeyRingConfigurationError("each key must contain only name, key, operations, and wings")
        name, secret, operations, wings = (
            entry["name"], entry["key"], entry["operations"], entry["wings"]
        )
        if not isinstance(name, str) or not name or len(name) > 128:
            raise KeyRingConfigurationError("key name must be a non-empty string of at most 128 characters")
        if not isinstance(secret, str) or len(secret) < 16:
            raise KeyRingConfigurationError("each opaque key must be a string of at least 16 characters")
        if not isinstance(operations, list) or not operations or any(op not in VALID_OPERATIONS for op in operations):
            raise KeyRingConfigurationError("operations must be a non-empty list of read, write, and/or admin")
        if not isinstance(wings, list) or not wings or any(not isinstance(wing, str) or not wing for wing in wings):
            raise KeyRingConfigurationError("wings must be a non-empty list of wing names or [\"*\"]")
        if "*" in wings and len(wings) != 1:
            raise KeyRingConfigurationError('wings may contain "*" only by itself')
        if name in names or secret in secrets:
            raise KeyRingConfigurationError("key names and opaque keys must each be unique")
        names.add(name)
        secrets.add(secret)
        grants.append(KeyGrant(name, secret, frozenset(operations), frozenset(wings)))
    return tuple(grants)


def load_key_ring(env: Mapping[str, str] | None = None) -> tuple[KeyGrant, ...]:
    """Load a 0600 JSON ring, or migrate the legacy single env key in-memory.

    An empty result deliberately preserves the daemon's local-development mode
    (no authentication configured). A file and PALACE_API_KEY together are
    rejected so operators cannot accidentally grant more access than intended.
    """
    env = os.environ if env is None else env
    config_path = env.get("PALACE_API_KEYS_FILE", "")
    legacy_key = env.get("PALACE_API_KEY", "")
    if config_path and legacy_key:
        raise KeyRingConfigurationError("set either PALACE_API_KEYS_FILE or PALACE_API_KEY, not both")
    if config_path:
        path = Path(config_path).expanduser()
        # A rotation may happen between lstat and read. Retry once and only
        # cache data whose metadata is unchanged across the read.
        for _ in range(2):
            state = _require_safe_file(path)
            cached = _key_ring_cache.get(path)
            if cached is not None and cached[0] == state:
                return cached[1]
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise KeyRingConfigurationError(f"invalid PALACE_API_KEYS_FILE: {exc}") from exc
            if _require_safe_file(path) != state:
                continue
            ring = _parse_key_ring(raw)
            # Cache only after the complete validation path succeeds. The state
            # contains inode, timestamps, ownership, and mode so atomic rotations
            # and in-place edits reload safely on the next request.
            _key_ring_cache[path] = (state, ring)
            return ring
        raise KeyRingConfigurationError("PALACE_API_KEYS_FILE changed while being read")
    if legacy_key:
        return (KeyGrant("legacy-palace-api-key", legacy_key, VALID_OPERATIONS, frozenset({"*"})),)
    return ()


def authenticate(presented_key: str | None, env: Mapping[str, str] | None = None) -> KeyGrant | None:
    """Return a grant for a valid opaque key, or None when auth is disabled."""
    ring = load_key_ring(env)
    if not ring:
        return None

    matched: KeyGrant | None = None
    candidate = presented_key or ""
    # Compare every configured secret before deciding, avoiding an early-match
    # timing oracle that could reveal which key name was presented.
    for grant in ring:
        equal = hmac.compare_digest(candidate, grant.secret)
        if equal:
            matched = grant
    if matched is None:
        raise AuthorizationError("Invalid API key")
    return matched


def authorize(
    presented_key: str | None,
    operation: str,
    wing: str | None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Authorize a request and return a non-secret audit identity."""
    if operation not in VALID_OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    matched = authenticate(presented_key, env)
    if matched is None:
        return "anonymous"
    if operation not in matched.operations:
        raise AuthorizationError("API key is not permitted for this operation")
    if not matched.unrestricted_wings:
        if wing is None or wing not in matched.wings:
            raise AuthorizationError("API key is not permitted for this wing")
    return matched.name
