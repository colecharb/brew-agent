"""This code must never write to the database.

It reads production tables as a real user. A stray `.insert(` or `.rpc(` is a
one-character mistake away from mutating somebody's brew history, so the ban is
enforced by the test suite rather than by a comment nobody re-reads.
"""

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "brew_agent"

# PostgREST/supabase-py write verbs. `.rpc(` is included because several of this
# schema's RPCs are SECURITY DEFINER and mutate (log_brew, moderate_content).
WRITE_CALLS = re.compile(r"\.(insert|upsert|update|delete|rpc)\s*\(")

# Supabase mutations always go through a table handle; these are the only
# same-named methods a reader legitimately calls.
ALLOWED = re.compile(r"\b(dict|set|os\.environ|counter|stats|by_user|result)\.update\s*\(")


def _source_files():
    return sorted(PACKAGE.rglob("*.py"))


def test_package_has_sources():
    assert _source_files(), "expected python sources under brew_agent/"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_write_calls(path):
    offenders = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]
        if WRITE_CALLS.search(code) and not ALLOWED.search(code):
            offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "database write call in read-only package:\n" + "\n".join(
        offenders
    )


def test_the_guard_actually_catches_a_write(tmp_path):
    """A guard that never fires is worse than none — prove it fires."""
    assert WRITE_CALLS.search('client.table("brew").insert({"id": 1})')
    assert WRITE_CALLS.search('client.table("brew").delete().eq("id", x)')
    assert WRITE_CALLS.search('client.rpc("log_brew", params)')
    assert not WRITE_CALLS.search('client.table("brew").select(SELECT)')
