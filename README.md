# ARC Director

Hierarchical RL over a DSL register machine for ARC-AGI-2.

A **manager** proposes goals every K steps in a learned discrete latent space; a
**worker** writes one typed DSL statement per step to reach them. Both read a
shared transformer encoder over the task's demonstration pairs, and each has its
own LSTM — the transformer→LSTM shape from the A2C-EPN Alchemy work, with the
episodic-memory block transplanted onto the program being written.

Training starts on **generated tasks whose solutions are provably inside the
action space**, walks a curriculum up to the full operator set, and only then
meets real ARC.

Full rationale in [`DESIGN.md`](DESIGN.md).

---

## Why this shape

The predecessor (`C:\arc-2-solution`) had an LLM emit whole programs as text and
optimised with token-level PPO. Its own coverage audit found DSL v0 could
provably express **1.8%** of ARC-AGI-2 training tasks, so a flat learning curve
could not be diagnosed: either the agent could not learn, or the answer was not
in the language.

Three changes follow from that:

| | Before | Now |
|---|---|---|
| Policy output | a whole program, as text | one statement per step, factored discrete action |
| Invalid programs | penalised after the fact | probability zero, by masking |
| Credit assignment | one scalar per program | per statement, plus goal-reaching reward |
| Expressivity | 40 first-order ops (v0) or 178 nested ops (v1, text-only) | 97 flat ops, first-order, usable as an action space |
| First contact with ARC | the whole pool at once | the subset a bounded search **proved** is reachable |
| Exploration | entropy bonus | manager goals + goal-space novelty |

---

## Install and check

```bash
cd C:\arc-director-solution
pip install -e .[dev]
pytest -q
```

On Windows, the repository also includes an idempotent setup script:

```powershell
.\scripts\setup.ps1
```

It creates `.venv`, installs CUDA PyTorch and the project, verifies the GPU,
and runs the correctness suite. Use `-CudaWheel cpu` for a CPU-only smoke-test
environment.

The current suite is 105 tests over the machine, the operators, the environment,
the networks, the trainer and the evaluation protocol. It runs on CPU in about
ten seconds.

Two of those tests are worth knowing about, because they cover the failure modes
that are otherwise invisible:

- `test_the_update_replays_exactly_what_was_acted_on` — a recurrent on-policy
  trainer computes gradients in a *second* forward pass. If it disagrees with
  the one that acted, the gradient belongs to a policy that never ran and the
  loss curve looks fine. The rollout stores its sampled log-probabilities and
  the update asserts the drift is below 1e-4.
- `test_evaluation_never_looks_at_the_test_output` — poisoning the test output
  must not change which candidate gets selected.

---

## Run

### Launch button (recommended)

Open **Run and Debug** in VS Code and choose one of the included buttons:

- **ARC Director - START (generated programs -> ARC-1 + ARC-2)** starts a new
  run and refuses to overwrite an existing checkpoint.
- **ARC Director - RESUME** continues whichever checkpoint is furthest along.
- **ARC Director - SMOKE TEST + DASHBOARD** runs a short CPU-friendly pipeline
  check.

The start button deliberately runs two compatible phases. It first learns from
self-generated DSL programs on small grids, then transfers that checkpoint to
the full curriculum over ARC-1 and ARC-2. Before the ARC phase it reuses local
dataset copies when available (including `C:\arc-2-solution\data`), otherwise
it downloads the official datasets, and mines the certified reachable subset.
The one-time reachability search is CPU-heavy.

Each training phase opens the live dashboard at
`http://127.0.0.1:8321/`. The dashboard reads the durable
`train_log.jsonl`, so the charts and the raw record cannot disagree. Watch
held-out generalization for promotion, ARC dev exactness/pass@N for real-task
progress, and top-operator share for policy collapse.

The same pipeline is available without VS Code:

```bash
python scripts/run_launch.py --fresh
python scripts/run_launch.py --resume
```

```bash
# Warm-up only: generated tasks, small grids, CPU-runnable.
python scripts/train.py --config configs/warmup_small.yaml

# Look at what the curriculum is actually generating.
python scripts/show_warmup.py --list
python scripts/show_warmup.py --stage w3_objects --n 3

# Find the ARC tasks the action space provably covers (writes data/reachable.json).
python scripts/mine_reachable.py --out data/reachable.json --budget 6

# Start a clean strict-Director pipeline. Old hybrid checkpoints remain under
# runs/warmup_small and runs/full for diagnosis and are never overwritten.
python scripts/run_launch.py --fresh

# Evaluate under the ARC protocol (2 attempts, demonstrations-only selection).
python scripts/evaluate.py --config configs/full.yaml \
    --checkpoint runs/director_proper/checkpoint.pt --split evaluation --attempts 16
```

Training prints one line per `log_every` updates and appends the full record to
`<run_dir>/train_log.jsonl`:

```
[   120] steps=  61,440 stage=w0_geometry_1 solve=0.585 gen=0.585 ret=+3.72 goal_r=-0.080 H_w=2.35 top=FLIP_V:0.22 sps=150
```

`solve` is exactness on the scored demonstrations; `gen` is exactness on the
**held-out** demonstration as well. `gen` is the number to watch — it cannot be
reached by fitting what you were scored on, and it is what gates promotion.
`top=` is the most-chosen operator and its share: if that climbs past ~0.4 the
policy is collapsing onto one action, which is the failure mode to watch for.

Both are measured at training temperature with an entropy bonus in force, so
they read well below what the policy does greedily — see the temperature table
below. `scripts/inspect_run.py <run_dir>` prints the whole curve.

The shipped warm-up and ARC configs also enable crossed epiplexity duels. One
fork pairs a low-temperature Director with a high-temperature worker and the
other reverses the pairing. Each critic measures its value-loss improvement on
the same transitions and fixed targets before and after every A2C update; those
deltas are summed over four updates. Director and worker winners are selected
independently and recombined, with the worker owning the shared visual trunk.
The dashboard reports both AUC comparisons and cumulative exploration win rates.

The configs enforce the Director-proper contract: one latent per DSL action,
worker reward equal to latent progress only, and task reward visible only to
the Director. Clean runs write to `runs/director_proper_warmup` and
`runs/director_proper`; earlier hybrid checkpoints are preserved.

Run the ablation alongside anything you train:

```bash
python scripts/train.py --config configs/ablation_flat.yaml   # same setup, hierarchy off
```

---

## What is where

```
src/arc_director/
  dsl/
    operators.py    40 v0 operators, carried over unchanged
    ops_v2.py       57 more: the mechanism classes the coverage audit named
    machine.py      the register machine: masked factored action space   <- the core
    search.py       bounded beam search, for certifying ARC coverage
    synthetic.py    random grid generation (carried over)
  arc/              grids, tasks, scoring, augmentation (carried over)
  env/
    task_env.py     one episode = one program-synthesis attempt
    vec.py          synchronous batching with autoreset
  curriculum/
    generator.py    warm-up tasks by random rollout of the machine itself
    stages.py       operator groups and the default ladder
    sources.py      where a task comes from; when to promote
  models/
    encoders.py     grid encoder, demonstration-pair encoder
    epn.py          set attention + the A2C-EPN episodic memory block
    goal.py         the discrete goal autoencoder (Director's abstract actions)
    agent.py        trunk, manager, worker, factored actor
  train/
    director.py     A2C over two timescales
    evaluate.py     the ARC protocol: demos-only selection, augmentation voting
  config.py         YAML -> objects, with cross-validation
```

---

## The action space in one example

```
g0 = INPUT
s0 = COMPONENTS(g=g0)
o0 = LARGEST(s=s0)
g1 = CROP(x=o0)
g2 = ROTATE90(g=g1)
RETURN g2
```

Five environment steps. Each is `(op, arg_0..arg_4)` chosen from 98 operators
and a 36-entry argument vocabulary, masked to what the machine will accept.
Registers are recency rings — `g[0]` is always "the newest grid", which is the
working canvas, the thing the goal reward is measured against, and the answer
when the episode ends.

Every statement executes on **every demonstration at once**, plus the test
input. A statement that fails on any of them is rejected wholesale and leaves no
trace. That is both the right semantics for program synthesis and what keeps a
single action mask valid for the whole machine.

---

## Handling ARC's shape

ARC-AGI-2 training tasks carry 2–10 demonstrations (97% have ≤ 5) and 1–4 test
inputs; 65% of pairs preserve grid size.

- Demonstrations are **contexts**, not a concatenated prompt. Adding one adds a
  context; the action space does not change.
- The encoder pools over a **masked set**, so two demonstrations and six are the
  same computation. (Permutation invariance and padding isolation are both
  tested.)
- Tasks with more demonstrations than `max_demos` resample a subset each
  episode, so nothing is permanently discarded.
- One demonstration is **held out** from the reward and used as the metric that
  gates promotion.
- Augmentation (colour permutation × D4 × reordering, ~2.9M variants) is sampled
  per episode, and the evaluator inverts it so several attempts can vote.

---

## What has actually been run

Everything below is CPU, on the warm-up ladder. No claim is made about ARC
accuracy — an ARC run needs a GPU and has not happened.

**DSL coverage.** A bounded beam search (depth ≤ 3, 4 s/task) certifies exact
programs for **6.5% of ARC-1 training** (26/400), **4.25% of ARC-2 training**
(17/400), and 3.2% of the merged de-duplicated pool (32/1009) — against the
predecessor's 4.25% / 1.8%. The new operators are why: 18 of the 32 merged
certificates use `MIRROR_CONCAT`, which the old DSL did not have. Every program
that fitted all demonstrations was also exact on the test pair — 32 for 32 —
which is direct support for the premise the evaluation protocol rests on.

**Warm-up rung 0 is solved.** The task: infer which of seven dihedral operators
a task applies, from three demonstrations, and commit to it. Random baseline is
0.125. The training solve rate reaches 0.60 by 60k environment steps and
plateaus — but that plateau is the entropy bonus, not the policy. Evaluated
properly on 60 freshly generated tasks under the full ARC protocol
(demonstrations-only selection, 8 attempts):

| sampling temperature | exact |
|---|---:|
| 1.0 (the training distribution) | 0.833 |
| 0.5 | 0.933 |
| 0.25 | **1.000** |

The held-out demonstration is solved at the same rate as the scored ones
throughout, so it is inferring the rule rather than fitting what it is graded
on. Two consequences: the set encoder demonstrably reads demonstrations and
conditions on them, and the curriculum's promotion thresholds — first written as
if the training rate were an accuracy — had to be recalibrated, or no rung would
ever have been cleared.

**The hierarchy is not yet doing anything measurable.** An ablation with the
goal reward switched off (`configs/ablation_flat.yaml` — same network, same
curriculum, same seed, worker ignores the manager) matches the hierarchical run
rung for rung: 0.615 vs 0.595 at 66k, 0.620 vs 0.575 at 128k. On a rung whose
answer is one operator, there is no temporal structure for a manager to abstract
over, so this is expected rather than damning — but it does mean nothing
measured so far is evidence for Director over flat A2C, and the ablation should
be run alongside anything that follows.

**The ARC stages run end to end** (30×30 grids, reachable-task pool, periodic
dev-split evaluation) at ~70 steps/s on CPU. That is a pipeline check, not a
result.

Getting there took three fixes that are written up in `DESIGN.md` §7, because
each one failed in a way that looks like "the model just isn't learning":

1. Director's absolute goal reward pays the worker to stand still and penalises
   it for stopping — the halt rate went to zero and the solve rate sat at the
   random baseline for 300k steps.
2. The earlier hybrid leaked task reward into the worker and crossed Director
   goal boundaries. That experiment is retained as history; the active strict
   Director gives the worker no task reward at all.
3. Squared value loss on returns that jump an order of magnitude the moment the
   agent starts succeeding sends that gradient through the shared trunk.

Crossed Director/worker epiplexity dueling is now implemented at that split.
Still deliberately deferred is a general `FOREACH` block, which is the largest
remaining expressivity gap.
