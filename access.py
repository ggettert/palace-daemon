"""Small, dependency-free authorization layer for palace-daemon."""
from __future__ import annotations

import hmac
import json
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

VALID_OPERATIONS = frozenset({"read", "write", "admin"})


class KeyRingConfigurationError(ValueError):
    """The key-ring configuration is malformed or unsafe to read."""


class AuthorizationError(PermissionError):
    """An authenticated caller is not allowed to perform a request."""


@dataclass(frozen=True)
class WingPermission:
    """The wing allow/deny rule for one operation."""

    allow: frozenset[str]
    deny: frozenset[str] = frozenset()

    @property
    def unrestricted(self) -> bool:
        """Whether an operation can safely run without a named wing."""
        return self.allow == frozenset({"*"}) and not self.deny

    def permits(self, wing: str) -> bool:
        return wing not in self.deny and ("*" in self.allow or wing in self.allow)


@dataclass(frozen=True)
class KeyGrant:
    """A named opaque token and its least-privilege per-operation grants."""

    name: str
    secret: str
    permissions: Mapping[str, WingPermission]

    @property
    def operations(self) -> frozenset[str]:
        """Compatibility view of operations configured for this key."""
        return frozenset(self.permissions)

    def permission_for(self, operation: str) -> WingPermission | None:
        return self.permissions.get(operation)


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


def _safe_file_state(info: os.stat_result) -> _KeyRingFileState:
    """Validate a key-ring file stat result and return cache metadata."""
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


def _require_safe_file(path: Path) -> _KeyRingFileState:
    try:
        # lstat is intentional: a key-ring path must never redirect through a
        # symlink, even when its target is an otherwise-safe regular file.
        return _safe_file_state(path.lstat())
    except OSError as exc:
        raise KeyRingConfigurationError(f"cannot lstat PALACE_API_KEYS_FILE: {exc}") from exc


def _read_safe_key_ring(path: Path, expected_state: _KeyRingFileState) -> object | None:
    """Read exactly the validated file, returning None when it was rotated.

    ``lstat`` followed by ``Path.read_text`` would leave a window in which a
    path replacement could redirect the read. Open with ``O_NOFOLLOW`` and
    compare ``fstat`` metadata to the pre-open state instead; atomic rotations
    simply cause the caller to retry.
    """
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise KeyRingConfigurationError(f"cannot open PALACE_API_KEYS_FILE: {exc}") from exc
    try:
        state = _safe_file_state(os.fstat(fd))
        if state != expected_state:
            return None
        with os.fdopen(fd, encoding="utf-8") as handle:
            fd = -1
            return json.loads(handle.read())
    except (OSError, json.JSONDecodeError) as exc:
        raise KeyRingConfigurationError(f"invalid PALACE_API_KEYS_FILE: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _parse_wings(
    value: object, field: str, *, allow_wildcard: bool, reject_duplicates: bool = True
) -> frozenset[str]:
    if not isinstance(value, list) or not value or any(not isinstance(wing, str) or not wing for wing in value):
        raise KeyRingConfigurationError(f"{field} must be a non-empty list of wing names")
    if reject_duplicates and len(set(value)) != len(value):
        raise KeyRingConfigurationError(f"{field} must not contain duplicate wing names")
    if "*" in value and (not allow_wildcard or len(value) != 1):
        raise KeyRingConfigurationError(f'{field} may contain "*" only by itself')
    return frozenset(value)


def _parse_permissions(value: object) -> dict[str, WingPermission]:
    if not isinstance(value, dict) or not value:
        raise KeyRingConfigurationError("permissions must be a non-empty object keyed by operation")
    permissions: dict[str, WingPermission] = {}
    for operation, scope in value.items():
        if operation not in VALID_OPERATIONS:
            raise KeyRingConfigurationError("permissions may contain only read, write, and/or admin")
        if not isinstance(scope, dict) or not {"allow"} <= set(scope) or set(scope) - {"allow", "deny"}:
            raise KeyRingConfigurationError("each permission must contain allow and optional deny only")
        allow = _parse_wings(scope["allow"], f"permissions.{operation}.allow", allow_wildcard=True)
        deny_value = scope.get("deny", [])
        if not isinstance(deny_value, list) or any(not isinstance(wing, str) or not wing or wing == "*" for wing in deny_value):
            raise KeyRingConfigurationError(f"permissions.{operation}.deny must be a list of named wings")
        if len(set(deny_value)) != len(deny_value):
            raise KeyRingConfigurationError(f"permissions.{operation}.deny must not contain duplicate wing names")
        deny = frozenset(deny_value)
        if "*" not in allow and allow & deny:
            raise KeyRingConfigurationError(f"permissions.{operation}.allow and deny must not overlap")
        permissions[operation] = WingPermission(allow, deny)
    return permissions


def _parse_key_ring(raw: object) -> tuple[KeyGrant, ...]:
    if not isinstance(raw, dict) or set(raw) != {"keys"} or not isinstance(raw["keys"], list):
        raise KeyRingConfigurationError('key-ring JSON must be exactly {"keys": [...]}')
    if not raw["keys"]:
        raise KeyRingConfigurationError("key-ring must contain at least one key")

    grants: list[KeyGrant] = []
    names: set[str] = set()
    secrets: set[str] = set()
    legacy_fields = {"name", "key", "operations", "wings"}
    scoped_fields = {"name", "key", "permissions"}
    for entry in raw["keys"]:
        if not isinstance(entry, dict) or (set(entry) != legacy_fields and set(entry) != scoped_fields):
            raise KeyRingConfigurationError(
                "each key must use legacy name, key, operations, wings or name, key, permissions"
            )
        name, secret = entry["name"], entry["key"]
        if not isinstance(name, str) or not name or len(name) > 128:
            raise KeyRingConfigurationError("key name must be a non-empty string of at most 128 characters")
        if not isinstance(secret, str) or len(secret) < 16:
            raise KeyRingConfigurationError("each opaque key must be a string of at least 16 characters")
        if set(entry) == legacy_fields:
            operations = entry["operations"]
            if not isinstance(operations, list) or not operations or any(op not in VALID_OPERATIONS for op in operations):
                raise KeyRingConfigurationError("operations must be a non-empty list of read, write, and/or admin")
            # Legacy rings historically accepted duplicate names; retain that
            # behavior while normalizing them into the in-memory set.
            wings = _parse_wings(
                entry["wings"], "wings", allow_wildcard=True, reject_duplicates=False
            )
            permissions = {operation: WingPermission(wings) for operation in operations}
        else:
            permissions = _parse_permissions(entry["permissions"])
        if name in names or secret in secrets:
            raise KeyRingConfigurationError("key names and opaque keys must each be unique")
        names.add(name)
        secrets.add(secret)
        grants.append(KeyGrant(name, secret, permissions))
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
            raw = _read_safe_key_ring(path, state)
            if raw is None or _require_safe_file(path) != state:
                continue
            ring = _parse_key_ring(raw)
            # Cache only after the complete validation path succeeds. The state
            # contains inode, timestamps, ownership, and mode so atomic rotations
            # and in-place edits reload safely on the next request.
            _key_ring_cache[path] = (state, ring)
            return ring
        raise KeyRingConfigurationError("PALACE_API_KEYS_FILE changed while being read")
    if legacy_key:
        all_wings = WingPermission(frozenset({"*"}))
        return (KeyGrant("legacy-palace-api-key", legacy_key, {operation: all_wings for operation in VALID_OPERATIONS}),)
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


def _requested_wings(wing: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if wing is None:
        return None
    if isinstance(wing, str):
        return (wing,) if wing else ()
    wings = tuple(wing)
    if not wings or any(not isinstance(item, str) or not item for item in wings):
        return ()
    return wings


def authorize(
    presented_key: str | None,
    operation: str,
    wing: str | Iterable[str] | None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Authorize an operation against every target wing and return audit identity.

    A request with no determinable wing is allowed only by an unrestricted rule.
    This prevents a rule with protected-wing denies from bypassing those denies
    through cross-wing endpoints or MCP tools with opaque identifiers.
    """
    if operation not in VALID_OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    matched = authenticate(presented_key, env)
    if matched is None:
        return "anonymous"
    permission = matched.permission_for(operation)
    if permission is None:
        raise AuthorizationError("API key is not permitted for this operation")
    requested_wings = _requested_wings(wing)
    if requested_wings is None:
        if not permission.unrestricted:
            raise AuthorizationError("API key is not permitted for this wing")
    elif not requested_wings or any(not permission.permits(target) for target in requested_wings):
        raise AuthorizationError("API key is not permitted for this wing")
    return matched.name
