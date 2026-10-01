"""
Task 6.4 — Password hashing.

Argon2id is the OWASP-recommended default for new applications
(https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html).

argon2-cffi ships parameters that are sane for interactive login latency.
Never store plaintext passwords. Never log a password. Never return one.
"""

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_hasher = PasswordHasher()


def hash_password(plaintext: str) -> str:
    if not plaintext:
        raise ValueError("password must not be empty")
    return _hasher.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    try:
        _hasher.verify(hashed, plaintext)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False
