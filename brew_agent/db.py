"""Read-only data access, authenticated as a real user.

## How this connects, and what that does and doesn't buy

The client uses the app's anon key and signs in with `sign_in_with_password`,
so every request carries a user JWT and reads at exactly the app's own
privilege level. No service-role key is used here, consistent with
`analytics/.env.example` and `docs/admin-operations.md`.

Be precise about what RLS enforces, because it is not what you might assume:

- `brew` SELECT is **world-readable** by policy (`schema.sql:3573`) — a brew is
  visible unless `hidden_at` is set or a `user_block` row stands between the
  two users. There is no owner clause. Scoping brews to a user is therefore
  application-level filtering on `created_by` no matter which key connects.
- `profile_coffee` is owner-only (`schema.sql:3585`). That policy is what keeps
  `cost`, `notes`, `amount_grams`, and `archived_at` private, and it is the
  real reason to authenticate as a user rather than escalate. Cross-user bag
  lookups go through `profile_coffee_public` (`schema.sql:2797`), a
  postgres-owned view exposing only the FK and the freshness dates.
- Catalogue rows are read through the `*_resolved` views. Reading the base
  tables surfaces merged duplicates with stale names.

## Read-only

Only SELECT helpers live in this module. `tests/test_read_only.py` greps the
whole package for write verbs and fails the suite if any appear.
"""

from __future__ import annotations

import logging
from typing import Any

from supabase import Client, create_client

from .config import SupabaseConfig
from .models import Brew

log = logging.getLogger(__name__)

# Mirrors apiHooks/brewList.ts:9-66 — the embed shape the app already proves
# works against these policies — plus days_off_roast, which the feed doesn't
# render but a diagnosis needs.
SELECT = """
    id,
    created_by,
    brewTimestamp:brew_timestamp,
    profileCoffee:profile_coffee_public!profile_coffee_id (
      id,
      coffeeId:coffee_id,
      coffee:coffee_resolved!coffee_id (
        name,
        roaster:roaster_resolved!roaster_id ( name ),
        origin,
        process,
        variety
      ),
      roastDate:roast_date
    ),
    grinder_id,
    brewer_id,
    burr_id,
    filter_id,
    grinder:grinder_resolved!grinder_id ( name ),
    brewer:brewer_resolved!brewer_id ( name ),
    burr:burr_resolved!burr_id ( name ),
    filter:filter_resolved!filter_id ( name ),
    grindSetting:grind_setting,
    coffeeWeight:coffee_weight,
    targetWeight:target_weight,
    brewWeight:brew_weight,
    waterTemp:water_temp,
    daysOffRoast:days_off_roast,
    time,
    recipe,
    notes,
    rating,
    rebrewOf:rebrew_of
"""

PAGE_SIZE = 500
DEFAULT_LIMIT = 20


class BrewDatabase:
    """Authenticated read-only access to brews, beans, and gear."""

    def __init__(self, client: Client, user_id: str) -> None:
        self._client = client
        self.user_id = user_id

    @classmethod
    def connect(cls, config: SupabaseConfig | None = None) -> "BrewDatabase":
        config = config or SupabaseConfig.from_env()
        client = create_client(config.url, config.anon_key)
        session = client.auth.sign_in_with_password(
            {"email": config.email, "password": config.password}
        )
        if not session or not session.user:
            raise RuntimeError(
                "Supabase sign-in returned no session. Check BREW_AGENT_EMAIL / "
                "BREW_AGENT_PASSWORD."
            )
        log.info("signed in as %s", session.user.id)
        return cls(client, session.user.id)

    # --- the three tools exposed to the model ------------------------------
    #
    # Every one takes `as_of`: the `brew_timestamp` of the brew being diagnosed.
    # History is everything *strictly before* it. Nothing later can come back.
    #
    # This is not an eval-only nicety. Diagnosing a brew means reasoning from
    # what was known at the time, and in production nothing later exists anyway —
    # the brew being diagnosed is the newest one. But in the eval the later brews
    # are sitting right there in the same table, and without the cutoff
    # `get_user_brews_with_bean` cheerfully returns the held-out answer. It did,
    # on the first real run: the agent read the next brew, saw it went badly, and
    # extrapolated past it. Scored "correct" and meant nothing.

    def get_brew(self, brew_id: str, as_of: str | None = None) -> Brew | None:
        """One brew: structured parameters plus the free-text notes and recipe.

        `<=` rather than `<` here, since the brew being diagnosed sits exactly
        at the cutoff and looking it up is the whole point.
        """
        rows = self._select().eq("id", brew_id).limit(1).execute().data
        if not rows:
            return None
        brew = Brew.from_api_row(rows[0])
        if as_of and brew.brew_timestamp > as_of:
            return None
        return brew

    def get_user_brews_with_bean(
        self,
        coffee_id: str,
        user_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
        as_of: str | None = None,
    ) -> list[Brew]:
        """History on the same coffee, most recent first.

        `coffee_id` is a `coffee.id` — the shared catalogue product — so this
        spans every bag of that coffee, not just the one bag. Brews reach it
        through `profile_coffee`, hence the bag lookup first.
        """
        bag_ids = self._bag_ids_for_coffee(coffee_id, user_id)
        if not bag_ids:
            return []
        query = self._select().in_("profile_coffee_id", bag_ids)
        if user_id:
            query = query.eq("created_by", user_id)
        return self._ordered(query, limit, as_of)

    def get_user_brews_with_gear(
        self,
        grinder_id: str,
        brewer_id: str,
        min_rating: int,
        user_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
        as_of: str | None = None,
    ) -> list[Brew]:
        """Well-rated brews on the same grinder-and-brewer setup.

        Grinder and brewer together are what make a grind number mean anything.
        The grinder sets the units; the brewer sets the regime. The Z1 in this
        dataset reads in microns for everything, but its espresso brews sit at
        5-250 and its filter brews at 475-600 — a baseline mixing the two is
        worse than no baseline.

        `.gte` on a nullable smallint drops unrated brews, which is what we want
        here — `min_rating` presupposes a rating.
        """
        query = (
            self._select()
            .eq("grinder_id", grinder_id)
            .eq("brewer_id", brewer_id)
            .gte("rating", min_rating)
        )
        if user_id:
            query = query.eq("created_by", user_id)
        return self._ordered(query, limit, as_of)

    # --- harness-only, never exposed to the model --------------------------

    def fetch_all_brews(self, max_rows: int = 5000) -> list[Brew]:
        """Every visible brew, for building the eval's holdout pairs.

        Not a tool. The eval needs to scan for consecutive same-coffee brews,
        which none of the three tools can express.
        """
        brews: list[Brew] = []
        offset = 0
        while offset < max_rows:
            page = (
                self._select()
                .order("brew_timestamp", desc=False)
                .order("id", desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
                .execute()
                .data
            )
            if not page:
                break
            brews.extend(Brew.from_api_row(row) for row in page)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        return brews

    # --- internals ---------------------------------------------------------

    def _select(self):
        # `hidden_at` is enforced by RLS for other users' brews, but say it
        # explicitly so moderated rows never reach the model or the eval.
        return self._client.table("brew").select(SELECT).is_("hidden_at", "null")

    def _bag_ids_for_coffee(self, coffee_id: str, user_id: str | None) -> list[str]:
        query = self._client.table("profile_coffee_public").select("id").eq(
            "coffee_id", coffee_id
        )
        if user_id:
            query = query.eq("profile_id", user_id)
        return [row["id"] for row in query.execute().data]

    @staticmethod
    def _ordered(query: Any, limit: int, as_of: str | None = None) -> list[Brew]:
        # The cutoff goes in the query, not in a post-filter, so `limit` counts
        # rows the caller may actually see. Post-filtering would silently return
        # fewer than asked for whenever recent brews were trimmed.
        if as_of:
            query = query.lt("brew_timestamp", as_of)
        rows = (
            query.order("brew_timestamp", desc=True)
            .order("id", desc=True)
            .limit(limit)
            .execute()
            .data
        )
        return [Brew.from_api_row(row) for row in rows]
