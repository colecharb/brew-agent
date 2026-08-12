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
All sampled pairs
arm           n |   ok  wrong  quiet  elsew    dir |     magnitude |   when improved |  held  err
-------------------------------------------------------------------------------------------------
rules        60 |   11      4     24      0   28% |    7/11   64% |      5/16   31% |    41    0

Pairs whose note describes a taste problem
arm           n |   ok  wrong  quiet  elsew    dir |     magnitude |   when improved |  held  err
-------------------------------------------------------------------------------------------------
rules        15 |    6      3      3      0   50% |     4/6   67% |       3/6   50% |     2    0

What each arm bet on
  rules      grind_setting 19, none 41
```

- **ok / wrong / quiet** — over the pairs where the user moved the grind: moved
  it the same way, moved it the opposite way, or recommended no grind change.
- **elsew** — of those quiet pairs, how many proposed a *different* lever
  instead. Still a miss on grind, but an opinion about temperature is not the
  same thing as having none, and only some arms can tell them apart: `rules` and
  `classify` reach for the grind or for nothing, so this column is always 0 for
  them. Without the split, the `classify` → `no_tools` rung reads as a
  capability gap when part of it is the arms with lever choice being penalised
  for using it.
- **dir** — `ok / (ok + wrong + quiet)`. Staying quiet counts as a miss, so a
  conservative arm cannot hide. The first row reads: the rule table is right
  about two thirds of the time *when it speaks*, but it has nothing to say on
  two thirds of the pairs.
- **when improved** — the same rate, restricted to pairs where the user's own
  change raised the rating. Those are the pairs where the change demonstrably
  worked, so this is the headline.
- **magnitude** — of the `ok` pairs, how often the size was within 0.5x–2x of
  what the user actually did.

The second table appears once the labelling pass has run (`--label`). It matters
more than it looks: most notes are not complaints at all — *"Yes."*, *"For
Clemi's latte"*, *"Fantastic. So open and sweet."* The user moved the grind for
reasons never written down, and no arm can be right about those. Scoring them
drags every arm toward the same middle, which is why the same rule table reads
26% across all pairs and 50% on the ones where a right answer exists.

Pairs are evaluated six at a time (`--concurrency`,
`BREW_AGENT_EVAL_CONCURRENCY`), which takes a four-arm hundred-pair run from
over an hour to about ten minutes. Completion order is nondeterministic but the
artefacts are not: scores are collected in sample order once the pool drains, so
a parallel run and a serial one produce byte-identical results. `--concurrency 1`
forces strict order.

Useful flags: `--arms rules` (no API key needed), `--label`, `--exclude-leaky`,
`--include-leaky` (all below).

## How it's evaluated

The ground truth is already in the data. For any brew followed by another brew
of the same coffee by the same user, the later brew is what that user decided to
change. Hold it out, hand the agent the earlier brew and its tasting notes, and
compare.

**372 pairs** survive the filter chain over the seed dump; **367** after
redaction. Each filter is counted and printed, so nothing is dropped silently.

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

**28% of otherwise-usable pairs state the answer in the input**, in two
grammars. The plan: *"I'll dial this down to 485 microns next brew"*, *"clearly
needs to be a little finer ground"*. And the verdict: *"Sour, drying. Too
coarse"* — past tense, but the answer just as plainly. A keyword table can read
neither; a model reads both perfectly. Left in, they would flatter exactly the
arms under test.

Sampling round-robins across users, because one logger owns roughly three
quarters of the eligible pairs.

### Why redaction, and not telling the model to ignore it

The offending sentence is cut before any arm sees the note. The obvious
alternative — instructing the model to disregard a stated adjustment — was
rejected for three reasons:

- **It can't be verified.** The phrase is in context either way. "Disregard" is
  a change in influence, not an observable action, so no artifact in the trace
  shows whether it worked.
- **It can't be detected after the fact.** Leaky notes usually also contain real
  taste evidence pointing the same way, so a contaminated answer and a reasoned
  one look identical.
- **It's asymmetric.** `rules` cannot read *"wasn't grinding coarse enough"*
  under any instruction. Only the model arms would be on the honour system, and
  they are the ones being measured.

Redaction has none of that: mechanical rather than a request, identical for
every arm, auditable in the trace, and it **keeps the pair** — 367 rather than
the 267 exclusion leaves. Cuts are by whole sentence, since a fragment is both
unreadable and still suggestive. Leak and evidence almost always sit in separate
sentences:

> *"Body and zing both! My guess is that maybe 5–10 um coarser could open this
> brew up"* → *"Body and zing both!"*

Three modes, all runnable:

| mode | flag | what it is for |
|---|---|---|
| redact | default | the measurement |
| exclude | `--exclude-leaky` | conservative cross-check — a big gap means redaction is leaving hints |
| raw | `--include-leaky` | harness self-test; any note-reading arm should near-ace it, and the gap against the default measures the contamination directly |

Detection is a regex plus an optional model labelling pass (`--label`), cached
to `labels/` and gitignored. Either detector is enough to cut a sentence, and a
label calling a note clean never un-redacts a regex match. A failed labelling
call fails closed. The funnel prints how many leaks only the labeller found — near
zero means the regex is doing the job.

The labelling pass is one call per note, run 8 at a time
(`BREW_AGENT_LABEL_CONCURRENCY`; lower it if your rate limits complain). A
first pass over ~700 notes takes a few minutes; after that it is cached by brew
id, so later runs only pay for notes you have added since. Progress is written
every 25 notes, so interrupting it loses at most 25 and resumes where it left
off.

## The arms — a ladder, one rung at a time

Each arm adds exactly one capability to the one below it, so the gap between any
two rungs prices that one thing.

| arm | reads the note | picks the change | reads history | reads everyone's |
|---|---|---|---|---|
| `rules` | keyword table | fixed ±5% | — | — |
| `classify` | **model** | fixed ±5% | — | — |
| `no_tools` | model | **model** | — | — |
| `agent` | model | model | **three tools** | — |
| `agent_community` | model | model | three tools | **a fourth** |

- `rules` → `classify` — the value of language understanding. Same 5% step, same
  ±2°C, same "higher is coarser" assumption; only the reader changes. A test
  asserts both arms emit byte-identical numbers for the same verdict, because
  the moment that diverges the delta stops measuring one variable.
- `classify` → `no_tools` — the value of letting the model choose the size and
  the lever, not just the direction.
- `no_tools` → `agent` — the value of retrieval.
- `agent` → `agent_community` — the value of everyone else's history. Same loop,
  same three tools, one more; `agent` is untouched so both arms stay comparable
  with the runs already recorded.

### The community rung

A grind number is comparable across people exactly when the grinder and brewer
match — which is why holdout pairs are keyed on that pair. But the tight query
(this coffee, this setup, somebody else) matches **14 of 100** sampled pairs and
usually returns a single brew. Widening is the normal path, not a fallback.

So `get_community_brews` returns up to three labelled groups rather than one
relaxed list, because they answer different questions:

| group | what transfers |
|---|---|
| same coffee, same setup | everything, grind number included |
| same setup, other coffees | the grind *range* this equipment works in |
| same coffee, other setups | ratio, temperature, days off roast — **not** grind |

That third row is the whole reason for the labels. Transferability is a property
of the *parameter*, not the setup: 92°C is 92°C on anyone's kettle, while 500 on
a Z1 means nothing on an EK43. A relaxed result handed over unlabelled is how a
stranger's grind number gets read off the wrong dial — and `agent`'s current
wrong-direction rate is 2 in 63, which is what that would spend.

Coverage says which groups will actually carry the run: same coffee on any gear
reaches **72 of 100** pairs (median 6 rows), same setup on any coffee **21**. So
the measurable question is whether same-coffee-different-gear data helps, and
the number to watch is **`wrong`, not `ok`** — community data that helps raises
correct calls, community data that pollutes raises wrong ones.

The viewer is supplied by the harness like the time cutoff, never by the model,
so "everyone else" can never quietly grow to include the user.

`classify` is deliberately blinkered: its prompt carries the note and nothing
else — no grind setting, no dose, no gear — and it is offered only
`classify_taste`, never `submit_recommendation`. That is what stops it quietly
becoming `no_tools`.

Its two abstaining verdicts, `both` and `neither`, recommend nothing and so are
between them the only way this arm loses without ever being wrong. The first
version of its brief said *"judge only what the note says about flavour"* and
twice more encouraged it to say nothing when unsure — and the arm duly returned
`neither` on a note opening *"Shot pulled way too fast"*, an unambiguous
under-extraction call containing no flavour word at all. It scored below the
keyword table it exists to beat, which was a result about the brief and not
about reading. The brief now admits evidence about how the brew ran, warns that
real notes rarely use the textbook words, and scopes both ways out narrowly.

What it does not do is mention the scoring. A model told that silence is
penalised will guess to protect the number rather than read the note, and the
rung stops measuring anything. The prompt describes the task; the metric stays
outside it.

`rules` stays rather than being replaced by `classify`, because it is the only
rung with no model in it at all. Without it, a gap between keyword matching and
a full recommender could not be attributed to either cause. It carries one
assumption worth naming — that a higher grind number means coarser. That holds
for microns and most dials, but it is exactly the guess the agent is meant to
avoid by reading the user's own history. Where it is wrong, this arm is
confidently backwards, which is the point of having it.

The fixed step has a floor, for a reason worth knowing about if you change it. A
percentage is not expressible on a coarse dial: 5% of 7 is 0.35, and a
whole-number grinder rounds that straight back to 7. Both fixed-step arms then
returned a confident recommendation of the setting already in use, which scores
as an abstention and is indistinguishable from having had no opinion — a silent
loss on 15 of the 367 pairs, all of them dials between 4 and 10. Where a
percentage rounds away the step is now the smallest increment the dial can
express; where 5% is expressible it is still exactly 5%.

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

A fourth, `get_community_brews`, is offered only to `agent_community` — see the
ladder above.

`user_id` is optional. It is a relevance control, not a security boundary — see
below. Every returned row carries `created_by` and the gear names so the model
can see when it is looking at somebody else's dial, and traces record whether a
`user_id` was given.

### The time cutoff

Every tool is bounded to brews made **strictly before** the one being diagnosed.
The cutoff is `brew.brew_timestamp`, taken from the brew itself so it cannot
drift out of sync with the question, and it is not a tool parameter — the model
cannot widen it.

This is not a detail. On the first live run, without it,
`get_user_brews_with_bean` returned the held-out next brew, and the agent said
so in its own reasoning:

> *"The next attempt at 4.1 with hotter water actually got worse"*

It read what the user did next and extrapolated one step further. Scored
"correct", meant nothing. In production the cutoff is invisible — the brew being
diagnosed is always the newest — but in the eval the answer is sitting in the
same table two rows down.

Guarded at three levels: `Toolbox.dispatch` takes `as_of` as a **required**
argument with no default, so omitting it raises rather than leaks;
`tests/test_db.py` asserts the queries carry `.lt("brew_timestamp", …)` into
PostgREST; and `tests/test_agent.py` runs the agent against a database
containing a known future brew and asserts no tool ever returns it.

An empty history result therefore means "no earlier brews", not "lookup failed",
and the tool descriptions say so.

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

Where an arm quotes the note back — `classify`'s `evidence` field — the quote is
checked against the note it read and the trace records `evidence_verbatim`. One
live call returned a malformed token fragment there instead of a quote. It cost
no accuracy, because only the verdict feeds the arithmetic, which is exactly how
it went unnoticed for a whole run; what it cost was the trace's standing as an
account of the answer. The per-arm count of misquotes is printed after the
tables, so a recurrence is visible without grepping.

It was not a one-off: 53 of 100 pairs on Sonnet 5, and 67 of the same 100 on
Haiku 4.5. Since the flag says *whether* and never *why* — and a check stricter
than the field deserves looks exactly like a model that invents quotes — the
quotes can be bucketed against the notes they came from:

```bash
.venv/bin/python -m brew_agent.eval.audit_evidence traces/<run_id> [traces/<other>]
```

Dropped punctuation means the check is too strict; spans stitched from separate
sentences mean the field should ask for one contiguous quote; a paraphrase means
the model is summarising and the expectation is wrong. Each wants a different
fix, so the buckets are the point. Offline, over traces already on disk, and it
takes several runs at once to compare models over the same pairs.

## Tests

```bash
.venv/bin/python -m pytest
```

No network, no API key, no database. Pair extraction and scoring run against
`supabase/seed.sql`, with the funnel counts pinned so a drift in the filter
chain fails loudly instead of quietly measuring a different population. The
agent loop is exercised with a scripted client, and one test greps the whole
package for database write verbs.

The seed dump is not part of this repository — it is a pg_dump of the dev
database and lives in the `dial` app repo, which is where this package sits as a
submodule. The default path finds it there. Elsewhere, point at your own dump:

```bash
BREW_AGENT_SEED_SQL=/path/to/seed.sql .venv/bin/python -m pytest
```

Without it the thirty-five tests that need real data skip, and the rest run.
