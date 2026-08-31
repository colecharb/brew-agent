# brew-agent

Does an LLM with three read-only tools recover the brewing
adjustments Dial's users actually made, better than a static rule table does?

Yes, and the tools are most of the reason: on the pairs where the user's own
change demonstrably worked, a keyword table gets the direction right 26% of the
time, the same model without tools 48%, and with history to read 74%. The
[full results](#results) and the ladder of arms that isolates each cause are
below; the short version is that **retrieval buys more than reading does.**

A research harness. No UI, no app integration, and it never writes to the
database — one test greps the whole package for write verbs to keep it that way.
It reads [Dial](https://github.com/colecharb/dial)'s Supabase instance at exactly
the app's own privilege level and vendors into that repo as a submodule at
`internal/brew-agent`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env      # then fill it in
```

`.env` needs the app's Supabase URL and anon key, the email and password of a
real account, and an API key for whichever provider serves the model — see
[Which model](#which-model). The default is Cohere's `command-r7b-12-2024`, so
`COHERE_API_KEY`.

## The one command

```bash
.venv/bin/python -m brew_agent.eval.run --n 24
```

It prints one row per arm — shape shown here, real numbers under
[Results](#results):

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

## Which model

One model per run, by construction: every arm reads the same `ModelConfig`. The
ladder prices capability — retrieval against no retrieval — and two rungs on
different models would price the swap instead. `BREW_AGENT_MODEL` sets it and
the provider follows from the name (`claude-*` Anthropic, `command-*` and
`north-*` Cohere), with `BREW_AGENT_PROVIDER` for anything the prefixes don't
know. A name it cannot place fails at startup rather than authenticating
against the wrong API, which reads like a bad key rather than a bad model name.

The labelling pass is the one exception, via `BREW_AGENT_LABEL_MODEL`. It is
not a rung: it gates every arm identically, so a better labeller changes the
population rather than the comparison. It is also the component whose errors
are *not* symmetric — a leak it misses only helps the arms that can read prose —
which makes it the last thing to economise on and the first thing to raise.

`brew_agent/providers.py` is the whole of the vendor difference. Three things
diverge, and each is a place where a naive port is wrong rather than merely
ugly:

- **Tool arguments.** Cohere returns them as a JSON *string*. Parsed on the way
  in, and an unparseable one raises rather than degrading to `{}` — five nulls
  score as an abstention and are indistinguishable from a model with no
  opinion.
- **The transcript.** Anthropic wants its own assistant blocks handed back
  verbatim, because thinking blocks carry signatures that must survive the
  round trip; Cohere wants the pre-tool reasoning in `tool_plan` and each tool
  result as its own `role: "tool"` message. The loop keeps a neutral transcript
  and each provider renders it.
- **The schema.** Cohere's strict mode has no `anyOf`, which is exactly what
  `tools.nullable()` emits. Those fields are unwrapped and dropped from
  `required`, so "leave this alone" is spelled by absence rather than by null.
  Dropping them from `required` is the load-bearing half: under `strict_tools`
  a required property must be present, so leaving them in would make every
  answer set a dose, a yield, a temperature and a time — turning "change one
  thing" into "change everything", which no scoring column could tell from a
  recommendation that meant it.

Two smaller asymmetries are handled and worth knowing about. Anthropic can pin
a *named* tool; Cohere's `tool_choice` only says `REQUIRED`, meaning "some
tool" — equivalent here because every forced call site offers exactly one, and
a warning fires the moment that stops being true. And `effort` is Anthropic's
parameter: on Cohere it is not sent, and a run that set it is not comparable to
one that did not.

Cost, per 100 pairs, measured against the committed seed dump rather than
guessed — mean brew row 247 tokens, a typical `agent_community` trajectory
15k peak context and ~40k billed input:

| model | `agent_community` | four-arm run | labelling ~700 notes |
|---|---|---|---|
| `command-r7b-12-2024` | $0.16 | $0.26 | $0.02 |
| `claude-haiku-4-5` | $4.33 | $7.18 | $0.54 |
| `command-a-03-2025` | $10.65 | $17.54 | $1.25 |

A full pass over all 439 eligible pairs on R7B, labelling included, is about
$1.20. The ceiling — every agent pair burning all six iterations with every
tool returning its full 20 rows — is $3.24.

R7B's ceiling is 4000 output tokens and 128k context, against a 28k worst case
here, so context is not what limits it. Whether a 7B model holds a six-hop tool
loop is the open question, and the honest reason to run it: if it does, "with
history to read" beating "without" stops being a claim about frontier models.

## How it's evaluated

The ground truth is already in the data. For any brew followed by another brew
of the same coffee by the same user, the later brew is what that user decided to
change. Hold it out, hand the agent the earlier brew and its tasting notes, and
compare.

**372 pairs** survive the filter chain over the committed seed dump; **367**
after redaction. The recorded runs go against the live database, which has grown
since — the funnel under [Results](#results) is the same chain over 876 brews and
ends at 439. Either way each filter is counted and printed, so nothing is dropped
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
wrong-direction rate is 7 in 63, which is what that would spend.

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

## Results

Run `20260811T030004Z`. 100 pairs, sampled round-robin across 11 users from the
439 eligible after redaction (876 brews → 715 consecutive → 590 with notes → 564
same grinder → 470 same brewer → 459 both rated and numerically comparable → 448
something changed → 439 surviving redaction). `claude-haiku-4-5` on every model
arm, no effort parameter, labelling pass on. Every recorded run predates the
provider shim, so they are all Anthropic; the default is now
`command-r7b-12-2024`, and nothing below has been re-run on it.

```
All sampled pairs
arm                 n |   ok  wrong  quiet  elsew    dir |     magnitude |   when improved |  held  err
-------------------------------------------------------------------------------------------------------
rules             100 |   15      8     40      0   24% |    9/15   60% |      7/26   27% |    70    0
no_tools          100 |   21     19     23      2   33% |   12/21   57% |     11/26   42% |    39    0
agent             100 |   31      7     25     11   49% |   23/31   74% |     18/26   69% |    27    0
agent_community   100 |   32      3     28     12   51% |   27/32   84% |     21/26   81% |    26    0

Pairs whose note describes a taste problem
arm                 n |   ok  wrong  quiet  elsew    dir |     magnitude |   when improved |  held  err
-------------------------------------------------------------------------------------------------------
rules              52 |   12      7     17      0   33% |    6/12   50% |      6/23   26% |    27    0
no_tools           52 |   17     13      6      1   47% |    8/17   47% |     11/23   48% |     7    0
agent              52 |   25      5      6      3   69% |   17/25   68% |     17/23   74% |     5    0
agent_community    52 |   26      2      8      6   72% |   21/26   81% |     20/23   87% |     3    0

What each arm bet on
  rules            grind_setting 30, none 70
  no_tools         grind_setting 55, none 42, water_temp 3
  agent            coffee_weight 3, grind_setting 48, none 37, time 10, water_temp 2
  agent_community  coffee_weight 2, grind_setting 43, none 32, time 20, water_temp 3
```

**Retrieval is the rung that pays.** `no_tools` → `agent` is +16 points of
direction (33% → 49%) and +27 on the headline (42% → 69%), the largest gap on the
ladder by a wide margin. Reading the note better is not what closes it: the model
is identical across those two arms and only the history is added.

**The gain is not that the model speaks more often — it is that it stops being
wrong.** `no_tools` is the *least* quiet arm on diagnosable pairs (6 abstentions
against `rules`' 17) and buys that confidence badly: 13 wrong directions out of
36, close to a coin flip on the ones it commits to. `agent` speaks on almost the
same number of pairs and halves the errors to 5. A grind dial has no shared
orientation across 34 grinders, so an arm without history is guessing which way
the number runs; an arm with it can look.

**Community data is a precision play, not a recall one.** `agent_community` adds
one correct call over `agent` — well inside noise — but cuts wrong directions
from 7 to 3 and lifts magnitude accuracy from 74% to 84%. This is what the rung
was designed to test and the number to watch was `wrong`, not `ok`: strangers'
brews narrow *how far to move* rather than revealing new problems. It also
carries a real cost, visible in the levers row — it bets on `time` 20 times
against `agent`'s 10, and time is scored at 17%.

**Two things are still not working.** Ratio and time are near zero for every arm
(`agent_community` manages 2/30 and 15/88), and temperature is indistinguishable
from noise at n=14. The honest reading is that this harness measures grind and
gestures at everything else; the non-grind levers need either more pairs or a
scoring model that does not treat one held-out brew as ground truth for
parameters users change casually.

**Stability.** The preceding run (`20260810T202320Z`, same config, same arms)
put the ladder at 24 / 33 / 41 / 46% direction against this run's 24 / 33 / 49 /
51%. `rules` is deterministic and reproduced exactly; the ordering of the four
arms held; the absolute gaps moved by up to 8 points. Read the ladder, not the
decimals — at n=100 with no confidence intervals, only the ordering and the
large gaps are load-bearing, and the +1 correct call from the community rung is
not.

One caveat worth stating plainly: `agent_community` called `get_community_brews`
on only 40 of the 100 pairs, and the tight group (same coffee, same setup, someone
else) returned rows on none of them — the retrieved evidence was 21 pairs of
same-coffee-other-setup and a single same-setup-other-coffee. So the community
result above is what a mostly-unused fourth tool bought. Whether forcing the call
helps or just adds the transferability trap the labels exist to prevent is the
next thing to measure, not something these numbers answer.

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

Both wires are covered the same way. `tests/test_providers.py` asserts on the
native payload that went out and the normalised object that came back, never on
the neutral form in between — that being the part that cannot be wrong. A
mistranslation does not raise, it answers; the arms cannot tell what produced a
`ModelResponse`, which is the point of the shim and the reason it needs pinning
rather than trusting.

The seed dump is not part of this repository — it is a pg_dump of the dev
database and lives in the `dial` app repo, which is where this package sits as a
submodule. The default path finds it there. Elsewhere, point at your own dump:

```bash
BREW_AGENT_SEED_SQL=/path/to/seed.sql .venv/bin/python -m pytest
```

Without it the thirty-five tests that need real data skip, and the rest run.
