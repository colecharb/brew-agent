# brew-agent

Internal experiment. Does an LLM with three read-only tools recover the brewing
adjustments Dial's users actually made, better than a static rule table does?

Not part of the app. No UI, no app integration, never writes to the database.
`internal/` is excluded from Metro, TypeScript, and Prettier; the venv, `.env`,
traces, and eval output are gitignored.

## Setup

```bash
cd internal/brew-agent
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env      # then fill it in
```

`.env` needs the app's Supabase URL and anon key, the email and password of a
real account, and an Anthropic API key.

## The one command

```bash
.venv/bin/python -m brew_agent.eval.run --n 24
```

It prints one row per arm:

```
arm           n |   ok  wrong  quiet    dir |     magnitude |   when improved |  held  err
------------------------------------------------------------------------------------------
rules        60 |    9      4     26   23% |     5/9   56% |      4/15   27% |    42    0
```

- **ok / wrong / quiet** — over the pairs where the user moved the grind: moved
  it the same way, moved it the opposite way, or recommended no grind change.
- **dir** — `ok / (ok + wrong + quiet)`. Staying quiet counts as a miss, so a
  conservative arm cannot hide. The row above reads: the rule table is right
  about two thirds of the time *when it speaks*, but it has nothing to say on
  two thirds of the pairs.
- **when improved** — the same rate, restricted to pairs where the user's own
  change raised the rating. Those are the pairs where the change demonstrably
  worked, so this is the headline.
- **magnitude** — of the `ok` pairs, how often the size was within 0.5x–2x of
  what the user actually did.

Useful flags: `--arms rules` (no API key needed), `--include-leaky` (see below).

## How it's evaluated

The ground truth is already in the data. For any brew followed by another brew
of the same coffee by the same user, the later brew is what that user decided to
change. Hold it out, hand the agent the earlier brew and its tasting notes, and
compare.

**372 pairs** survive the filter chain over the seed dump; **332** after
excluding leakage. Each filter is counted and printed, so nothing is dropped
silently.

Three decisions that the data forced:

**Pairs are keyed on grinder *and* brewer.** A grind number only means something
within one setup. The grinder sets the units and the brewer sets the regime — the
Z1 in this dataset reads in microns throughout, but its espresso brews sit at
5–250 and its filter brews at 475–600. A delta across the two is noise.

**Direction is scored as the sign of a delta, never as "finer" or "coarser".**
34 grinders disagree about what a bigger number means, no table says which is
which, and a wrong entry in such a table would silently invert the metric for
every pair using that grinder rather than failing. So the agent is told the
current setting and must answer with a number on the same dial; matching signs
is a hit whichever way that particular grinder counts.

**10.8% of otherwise-usable pairs state the answer in the input.** Users write
things like *"I'll dial this down to 485 microns next brew"* in the notes that
become the complaint. Those are excluded by default and counted. `--include-leaky`
scores them anyway, which is a harness self-test rather than a measurement: any
arm that reads the notes should approach 100% there, and one that doesn't has a
bug.

Sampling round-robins across users, because one logger owns roughly three
quarters of the eligible pairs.

## The arms

| arm | model | history |
|---|---|---|
| `rules` | none | none |
| `no_tools` | one call | none |
| `agent` | tool loop | three tools |

`rules` is keyword matching with one assumption baked in — that a higher grind
number means coarser. That holds for microns and most dials, but it is exactly
the guess the agent is meant to avoid by reading the user's own history. Where
it is wrong, this arm is confidently backwards, which is the point of having it.

Between them the two baselines separate two questions: if `agent` doesn't beat
`no_tools`, the tools aren't earning their cost; if `no_tools` doesn't beat
`rules`, the model isn't either.

## Tools

Three data tools plus `submit_recommendation`, which is how the agent finishes —
making the answer a tool call means a malformed one is retried by the model
against the schema rather than by a regex, and the loop has an unambiguous
termination signal.

1. `get_brew(brew_id)` — parameters, notes, recipe, resolved coffee and gear.
2. `get_user_brews_with_bean(coffee_id, user_id?)` — history with the same
   coffee. `coffee_id` is the catalogue product, so it spans every bag.
3. `get_user_brews_with_gear(grinder_id, brewer_id, min_rating, user_id?)` —
   well-rated brews on the same setup, for a personal baseline.

`user_id` is optional. It is a relevance control, not a security boundary — see
below. Every returned row carries `created_by` and the gear names so the model
can see when it is looking at somebody else's dial, and traces record whether
each call was scoped or global.

The loop has a hard cap (`BREW_AGENT_MAX_ITERATIONS`, default 6). On hitting it,
one more call is made with `tool_choice` pinned to `submit_recommendation`, so a
run that explores forever still produces a scoreable answer — flagged `hit_cap`
in the trace.

## On RLS, precisely

The client uses the anon key and signs in as a real user, so it reads at exactly
the app's own privilege level. No service-role key is used anywhere here.

What that does *not* buy is per-user scoping of brews. `brew` SELECT is
world-readable by policy (`schema.sql:3573`) — a brew is visible unless
`hidden_at` is set or a `user_block` stands between the two users. Dial's brews
are a public feed by design. So filtering on `created_by` is application-level
regardless of which key connects, and `user_id` on the tools is about relevance:
grind numbers don't transfer across people and gear, so a baseline built from
someone else's brews would be actively wrong.

What the user session *does* buy is that `profile_coffee` stays owner-only
(`schema.sql:3585`), keeping `cost`, `notes`, `amount_grams`, and `archived_at`
private. Cross-user bag lookups go through the `profile_coffee_public` view,
which exposes only the foreign key and the freshness dates.

## Traces

One JSON file per (arm, pair) under `traces/<run_id>/`: the complaint and the
brew as given, the held-out next brew, every tool call in order with its
arguments and the actual rows returned, per-call latency, the assistant text at
each step, token usage, and the final recommendation with its score.

They are readable end to end without cross-referencing anything. They also
contain other users' tasting notes, since brews are public — `traces/` and
`evals/output/` are gitignored, so keep them local.

## Tests

```bash
.venv/bin/python -m pytest
```

No network, no API key, no database. Pair extraction and scoring run against the
committed `supabase/seed.sql`, with the funnel counts pinned so a drift in the
filter chain fails loudly instead of quietly measuring a different population.
The agent loop is exercised with a scripted client, and one test greps the whole
package for database write verbs.
