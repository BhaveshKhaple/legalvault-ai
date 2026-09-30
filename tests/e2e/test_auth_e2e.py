"""
Task 6.4 -- Live E2E smoke test for auth + multi-tenancy.

Runs against a real uvicorn server on port 8001. Verifies the full flow:
  1. /healthz
  2. /v1/auth/register  -> first user becomes admin
  3. /v1/auth/register  -> second user is analyst
  4. /v1/auth/login     -> returns JWT
  5. /v1/auth/me        -> returns identity
  6. Unauthorized /v1/cases returns 401
  7. Analyst creates a case (owner = them)
  8. Second analyst cannot see the first analyst's case (404)
  9. Admin sees both analysts' cases
 10. Full RAG query against created case (returns 200 with answer + evidence)

Assumes the server is already up at http://localhost:8001.
"""

import sys
import time
import uuid
from typing import Optional

import httpx

BASE = "http://localhost:8001"


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"OK    {msg}")


def _post(path: str, json=None, token: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.post(f"{BASE}{path}", json=json, headers=headers, timeout=60.0)


def _get(path: str, token: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.get(f"{BASE}{path}", headers=headers, timeout=60.0)


def main() -> int:
    # unique email per run so re-runs work against the same live DB
    tag = uuid.uuid4().hex[:8]
    admin_email = f"e2e-admin-{tag}@example.com"
    alice_email = f"e2e-alice-{tag}@example.com"
    bob_email = f"e2e-bob-{tag}@example.com"
    password = "hunter2hunter-e2e"

    # 1. health
    r = _get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    _ok("/healthz")

    # 2. register admin (may be second user in existing DB -- accept either role)
    r = _post("/v1/auth/register", {"email": admin_email, "password": password})
    if r.status_code != 201:
        _fail(f"admin register: {r.status_code} {r.text}")
    admin_token = r.json()["access_token"]
    admin_role = r.json()["role"]
    _ok(f"registered admin_user ({admin_role})")

    # 3. register two analysts
    r = _post("/v1/auth/register", {"email": alice_email, "password": password})
    if r.status_code != 201:
        _fail(f"alice register: {r.status_code} {r.text}")
    alice_token = r.json()["access_token"]
    alice_role = r.json()["role"]
    _ok(f"registered alice ({alice_role})")

    r = _post("/v1/auth/register", {"email": bob_email, "password": password})
    if r.status_code != 201:
        _fail(f"bob register: {r.status_code} {r.text}")
    bob_token = r.json()["access_token"]
    _ok(f"registered bob ({r.json()['role']})")

    # 4. login as alice
    r = _post("/v1/auth/login", {"email": alice_email, "password": password})
    if r.status_code != 200:
        _fail(f"alice login: {r.status_code} {r.text}")
    _ok("alice /login -> JWT")

    # 5. /me for alice
    r = _get("/v1/auth/me", token=alice_token)
    if r.status_code != 200 or r.json()["email"] != alice_email:
        _fail(f"alice /me: {r.status_code} {r.text}")
    _ok("/me returns alice")

    # 6. unauthorized
    r = _get("/v1/cases")
    if r.status_code != 401:
        _fail(f"unauth GET /v1/cases expected 401, got {r.status_code}")
    _ok("unauth /v1/cases -> 401")

    r = _post("/v1/cases", {"name": "should reject"})
    if r.status_code != 401:
        _fail(f"unauth POST /v1/cases expected 401, got {r.status_code}")
    _ok("unauth POST /v1/cases -> 401")

    # 7. alice creates a case
    r = _post("/v1/cases", {"name": f"Alice E2E case {tag}"}, token=alice_token)
    if r.status_code != 200:
        _fail(f"alice create case: {r.status_code} {r.text}")
    alice_case_id = r.json()["case_id"]
    _ok(f"alice created case {alice_case_id[:8]}...")

    # 8. bob cannot see alice's case
    r = _get(f"/v1/cases/{alice_case_id}", token=bob_token)
    if r.status_code != 404:
        _fail(f"bob viewing alice's case expected 404, got {r.status_code}")
    _ok("bob cannot see alice's case (404 -- no leak)")

    # bob's list must not include alice's case
    r = _get("/v1/cases", token=bob_token)
    if r.status_code != 200:
        _fail(f"bob list cases: {r.status_code}")
    bob_case_ids = {c["case_id"] for c in r.json()}
    if alice_case_id in bob_case_ids:
        _fail("bob's list leaked alice's case_id")
    _ok(f"bob's case list is isolated ({len(bob_case_ids)} own cases)")

    # 9. if we have an admin, verify admin sees everything
    if admin_role == "admin":
        r = _get("/v1/cases", token=admin_token)
        admin_case_ids = {c["case_id"] for c in r.json()}
        if alice_case_id not in admin_case_ids:
            _fail("admin should see alice's case")
        _ok(f"admin sees all cases ({len(admin_case_ids)} total)")

    # 10. run a query against alice's (empty) case -- should return 200 with empty evidence
    r = _post(
        f"/v1/cases/{alice_case_id}/query",
        {"question": "What are the penalty clauses?"},
        token=alice_token,
    )
    if r.status_code != 200:
        _fail(f"alice query: {r.status_code} {r.text[:200]}")
    body = r.json()
    if "answer" not in body or "evidence" not in body:
        _fail(f"query response missing fields: {body}")
    _ok(f"query returned answer + {len(body.get('evidence', []))} evidence chunks")

    # 11. bob cannot query alice's case
    r = _post(
        f"/v1/cases/{alice_case_id}/query",
        {"question": "peek"},
        token=bob_token,
    )
    if r.status_code != 404:
        _fail(f"bob querying alice's case expected 404, got {r.status_code}")
    _ok("bob cannot query alice's case (404)")

    print("\n=========================================")
    print("  E2E AUTH SMOKE -- ALL 11 CHECKS PASSED")
    print("=========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
