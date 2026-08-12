"""Parse `supabase/seed.sql` into `Brew` objects for offline tests.

The seed dump is a real pg_dump of the dev database, so the pair extraction can
be exercised against genuine data shapes — non-numeric grind settings, unrated
brews, empty notes, mixed grind scales — without a database or any network
access.

The dump is not part of this package. It lives in the `dial` app repo, which is
where this package sits as a submodule, so the default path finds it there. Set
`BREW_AGENT_SEED_SQL` to point somewhere else; without it, the tests that need
real data skip.

Test-only. The agent never reads this file; it goes through `db.py`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from brew_agent.models import Brew

DEFAULT_SEED_SQL = Path(__file__).resolve().parents[3] / "supabase" / "seed.sql"


def seed_sql_path() -> Path:
    override = os.environ.get("BREW_AGENT_SEED_SQL")
    return Path(override) if override else DEFAULT_SEED_SQL


def _parse_table(sql: str, table: str) -> list[dict[str, Any]]:
    """Read one `INSERT INTO "schema"."table" (...) VALUES (...), ...;` block.

    Hand-rolled because the values contain newlines, commas, and doubled
    apostrophes inside quoted strings, which a naive split would shred.
    """
    header = re.search(
        r'INSERT INTO "\w+"\."%s" \(([^)]*)\) VALUES\n' % re.escape(table), sql
    )
    if not header:
        return []
    columns = [c.strip().strip('"') for c in header.group(1).split(",")]

    rows: list[dict[str, Any]] = []
    i = header.end()
    while True:
        while i < len(sql) and sql[i] in "\t\n ":
            i += 1
        if i >= len(sql) or sql[i] != "(":
            break
        i += 1
        values: list[str | None] = []
        current = ""
        quoted = False
        while i < len(sql):
            char = sql[i]
            if quoted:
                if char == "'":
                    if i + 1 < len(sql) and sql[i + 1] == "'":
                        current += "'"
                        i += 2
                        continue
                    quoted = False
                    i += 1
                    continue
                current += char
                i += 1
                continue
            if char == "'":
                quoted = True
                i += 1
                continue
            if char == ",":
                values.append(current.strip())
                current = ""
                i += 1
                continue
            if char == ")":
                values.append(current.strip())
                i += 1
                break
            current += char
            i += 1
        rows.append(
            {
                column: (None if value == "NULL" else value)
                for column, value in zip(columns, values)
            }
        )
        while i < len(sql) and sql[i] in ",;":
            if sql[i] == ";":
                return rows
            i += 1
    return rows


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def load_seed_brews() -> list[Brew]:
    """Every non-hidden brew in the seed dump, with `coffee_id` resolved.

    Raises `FileNotFoundError` when the dump is not on disk, which the
    `seed_brews` fixture turns into a skip.
    """
    path = seed_sql_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    sql = path.read_text()
    bags = {row["id"]: row for row in _parse_table(sql, "profile_coffee")}
    names = {
        table: {row["id"]: row["name"] for row in _parse_table(sql, table)}
        for table in ("coffee", "grinder", "brewer", "burr", "filter")
    }

    brews = []
    for row in _parse_table(sql, "brew"):
        if row.get("hidden_at"):
            continue
        bag = bags.get(row["profile_coffee_id"]) or {}
        coffee_id = bag.get("coffee_id")
        time_value = _num(row.get("time"))
        days = _num(row.get("days_off_roast"))
        brews.append(
            Brew(
                id=row["id"],
                created_by=row.get("created_by"),
                brew_timestamp=row["brew_timestamp"],
                profile_coffee_id=row.get("profile_coffee_id"),
                coffee_id=coffee_id,
                coffee_name=names["coffee"].get(coffee_id),
                grinder_id=row.get("grinder_id"),
                brewer_id=row.get("brewer_id"),
                burr_id=row.get("burr_id"),
                filter_id=row.get("filter_id"),
                grinder_name=names["grinder"].get(row.get("grinder_id")),
                brewer_name=names["brewer"].get(row.get("brewer_id")),
                burr_name=names["burr"].get(row.get("burr_id")),
                filter_name=names["filter"].get(row.get("filter_id")),
                grind_setting=row.get("grind_setting"),
                coffee_weight=_num(row.get("coffee_weight")),
                target_weight=_num(row.get("target_weight")),
                brew_weight=_num(row.get("brew_weight")),
                water_temp=_num(row.get("water_temp")),
                time=None if time_value is None else int(time_value),
                days_off_roast=None if days is None else int(days),
                notes=row.get("notes") or "",
                recipe=row.get("recipe") or "",
                rating=None if row.get("rating") is None else int(row["rating"]),
                rebrew_of=row.get("rebrew_of"),
            )
        )
    return brews
