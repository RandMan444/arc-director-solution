# Design

Why each piece is the way it is, what was carried over from `C:\arc-2-solution`,
what was deliberately changed, and what is deferred.

---

## 1. What changed from the previous attempt, and why

The predecessor had a 0.6B language model emit whole DSL programs as text, with
PPO over tokens and epiplexity dueling deciding which exploration branch
survived. Two things in its own audit explain why it stalled:

| Finding (from `DSL_COVERAGE_AUDIT.md`) | Consequence |
|---|---|
| DSL v0 certified exact programs for **18 / 1000** ARC-2 training tasks (1.8%) | A flat learning curve had two explanations — the agent cannot learn, or the answer is not in the language — and no way to tell them apart |
| v1 fixed expressivity with 178 *functional* primitives (nesting, lambdas) | Only expressible as generated text, which forced the LLM-token-PPO setup and its enormous credit-assignment horizon |
| Reward was terminal on a whole program | One scalar for a 200-token emission |

This project keeps the DSL idea and throws out the token policy. Three changes:

1. **The DSL becomes a machine, not a language.** One statement per environment
   step, from a masked discrete action space. Credit assignment is per
   statement, and a syntactically invalid program is not merely penalised — it
   has probability zero.
2. **Expressivity is restored first-order.** The mechanism classes the audit
   named are added as flat, vectorised operators (`ops_v2.py`), so they fit an
   action space without lambdas.
3. **Exploration is structured, not entropic.** A Director manager proposes
   goals in a learned latent space every K steps; the worker is rewarded for
   reaching them. That replaces "raise the entropy bonus and hope".

---

## 2. The action space

### One statement per step

```
action = (op, arg_0 … arg_4)
```

- `op`: one of 97 operators, or `HALT`.
- `arg_i`: one entry of a shared 36-entry vocabulary — 11 register references,
  22 integer literals, 2 booleans, and `DEFAULT`.

The joint space is ~1.5 billion combinations; the factored form is six
softmaxes. Arguments are sampled autoregressively, each conditioned on the
operator and the arguments already chosen.

### Masking is the load-bearing part

Every head is masked against what the machine will actually accept: an operator
appears only when each of its parameters has at least one legal argument, and an
argument appears only when its type is assignable and, for a register, that
register holds a value. A sampled statement is therefore always type-correct.
What remains are genuine runtime failures (`LARGEST` of an empty set), which are
informative and reported rather than fatal.

Type compatibility is static, so it is precomputed once per `MachineSpec`; the
per-step mask is two numpy operations (~17 µs for the full 98×5×36 tensor).

### Recency-indexed registers

There is no destination-register head. A result is pushed onto a ring for its
type, so index 0 always means "the value this type most recently took". This
removes an action head, keeps the reference space at 11 entries, and makes
`Grid[0]` a natural canvas: it is the working grid, the thing the goal reward is
measured against, and the answer when the episode ends.

Depths are 4 grids / 2 objects / 2 object-sets / 2 integers / 1 bool. `Color`
shares the integer bank because the DSL treats the two as assignable.

### Operator set

97 operators: the 40 from v0, plus 57 in `ops_v2.py` covering the audit's gaps —
palette statistics, arithmetic and branching, scaling/splitting/cell-wise
combination, gravity/rays/symmetry, cell-set algebra, and vectorised per-object
maps (`MAP_RECOLOR`, `MAP_MOVE`, `MAP_BOX`, `PAINT_SET`, …) that stand in for
`apply(f, objects)` without adding lambdas to the action space.

**Still missing: a general `FOREACH`.** The audit found per-object iteration in
66% of known ARC-1 solutions. The vectorised maps cover "do the same thing to
every object"; they do not cover "do something *different* to each object based
on its properties". A `MAP_BEGIN(set)` / `MAP_END` block that binds each object
in turn to a special register and replays the body is the natural extension, and
it is the single largest expressivity item left. It is deferred on purpose: it
adds control flow to the action space, and the point of this iteration is to
find out whether the fundamentals learn anything at all.

---

## 3. The ARC strategy

### 1–10 demonstrations, 1–4 test inputs

Measured over ARC-AGI-2 training: 2–10 demonstrations, 97% have ≤ 5; 93% of
tasks have exactly one test input; 65% of pairs preserve grid size.

Three mechanisms, no special cases:

1. **Contexts, not concatenation.** The machine holds one value bank per
   demonstration plus one for the test input, and every statement executes on
   all of them at once. A statement that raises on *any* context is rejected
   wholesale and leaves no trace — which is what keeps register types identical
   across contexts, so one mask serves the whole machine. This is also the right
   semantics: a program that works on three demonstrations and crashes on the
   fourth is not a solution.
2. **A masked set encoder.** Contexts are a padded stack plus a mask, encoded
   per-slot and pooled permutation-invariantly. Two demonstrations and six are
   the same computation with a different mask. (Tested: `test_agent.py`.)
3. **Resampling.** Tasks with more demonstrations than `max_demos` draw a fresh
   subset each episode, so nothing is permanently discarded.

The test input is a context so its statements execute — but its output is never
in the observation, and never in the reward.

### The held-out demonstration

One demonstration is withheld from the reward and used only as a metric. It is
the in-training answer to "did it find the rule or overfit the demonstrations it
was scored on", and it moves long before test accuracy does. **Curriculum
promotion is gated on it**, so a rung cannot be cleared by demonstration-fitting.

### Augmentation

Carried over unchanged: colour permutation × D4 × demonstration reordering,
sampled once per task per episode (`8 · 9! ≈ 2.9M` distinct transforms with the
background pinned). If the rule is `f`, the augmented task's rule is `g∘f∘g⁻¹`
— same structure, same difficulty. This is what makes a 1000-task pool
effectively unbounded and is the main defence against memorisation.
`test_env.py` checks the property directly: an augmented rotation task is still
solved by exactly one single dihedral operator.

### Evaluation

ARC's actual rule: demonstrations are all you may look at when choosing an
answer, and you get two attempts.

1. Run N augmented attempts.
2. Keep only those exact on **every** demonstration — a filter that uses no test
   information.
3. Invert each survivor's augmentation and vote. Agreement across augmentations
   is evidence of a rule rather than a coincidence.
4. Score the top two distinct answers as attempt 1 and attempt 2.

`pass@n` is also reported. It peeks, so it is not an ARC score — it is an honest
upper bound on what better candidate selection could buy.

### The reachable subset

`scripts/mine_reachable.py` runs a bounded beam search over the action space on
every ARC task and records exact solutions. Each node is expanded by trying
*every* legal argument combination for operators that have few, and sampling for
operators that have many — so a one-step answer like "rotate" or "gravity down"
can never be missed by bad luck. Hits are proof of coverage; misses are not
proof of anything. The certified tasks become the first ARC rung, so a failure
there is unambiguously the agent's. This directly fixes the ambiguity that made
the previous project's ARC numbers unreadable.

Measured at depth ≤ 3, beam 12, 4 s/task:

| Pool | Certified (demo-exact **and** test-exact) | v0 baseline (previous audit) |
|---|---:|---:|
| ARC-1 training (400) | **26 (6.5%)** | 17 (4.25%) |
| ARC-2 training (first 400) | **17 (4.25%)** | ~1.8% over 1000 |
| ARC-1 + ARC-2 training, de-duplicated (1009) | **32 (3.2%)** | — |

Three things stand out.

Coverage roughly doubled against the predecessor's typed enumeration despite a
*shorter* time budget, and the credit goes to the new operators: 18 of the 32
merged certificates use `MIRROR_CONCAT`, which v0 did not have. `SYMMETRIZE`,
`GRAVITY`, `UPSCALE`, `DOWNSCALE` and `FRAME` account for several more.

Every program that fit all demonstrations was also exact on the test pair —
26/26, 17/17, 32/32 — which is direct evidence for the premise the whole
evaluation protocol rests on: a short DSL program that fits every demonstration
is very likely the rule, not a coincidence.

And 3.2% is still a small number. The reachable rung is a diagnostic, not a
destination; the route to a larger one runs through §2's `FOREACH` gap.

---

## 4. The warm-up curriculum

Generated by **random rollout of the machine itself**: sample grids, take N
random legal actions, keep the result as a task whose inputs are the grids and
whose outputs are the resulting canvases.

That construction guarantees the task is reachable in the worker's own action
space at exactly the advertised length — so warm-up failures are learning
failures, never expressivity failures. It also yields a ground-truth program for
free, which makes the curriculum inspectable (`scripts/show_warmup.py`) and
would give behaviour cloning something to imitate if it is ever wanted.

Filtering is where the quality is. Random rollouts are mostly degenerate, so a
candidate is rejected unless the output differs from the input, is not blank or
single-coloured, varies across demonstrations (a constant output ignores the
input), and depends on **every** statement — checked by backward liveness from
the returned canvas, so a 4-action rollout whose answer needs only one action is
not labelled as a 4-operator task.

The ladder unlocks operator *groups*, never a different action space: heads
cannot change width mid-run, so a stage is a mask over the full operator list.

| Rung | Operators | Program length | Grids | Promote at |
|---|---|---|---|---|
| `w0_geometry_1` | geometry | 1 | 4–6 | 0.55 |
| `w1_geometry_2` | + colour | 1–2 | 4–7 | 0.45 |
| `w2_shape` | + shape | 1–2 | 4–8 | 0.35 |
| `w3_objects` | + objects, paint | 2–3 | 5–10 | 0.30 |
| `w4_full` | everything | 2–4 | 5–12 | 0.25 |
| `a0_arc_reachable` | everything | — | ARC | 0.20 |
| `a1_arc_small` | everything | — | ARC | 0.15 |
| `a2_arc_all` | everything | — | ARC | terminal |

Promotion needs a rolling generalisation rate above the threshold **and** a
minimum episode count, so a lucky streak cannot skip a rung.

The thresholds look low because they are measured on the training distribution
— full sampling temperature, with an entropy bonus actively pushing the policy
off its argmax. A rung-0 checkpoint whose training rate had plateaued at 0.60
scored **0.93 at temperature 0.5 and 1.00 at temperature 0.25** on 60 freshly
generated tasks, under the full ARC protocol (demonstrations-only selection).
The training rate is a *signal that the rung is learned*, not an accuracy; the
first pass at this ladder set the thresholds as if it were an accuracy, and the
curriculum would never have advanced.

---

## 5. The architecture

```
obs ─┬─ GridEncoder ── PairEncoder ── SetAttention ──┐
     └─ register / scalar features ─────────────────-┴── state s_t   (shared trunk)
                                                        │
                  ┌─────────────────────────────────────┴──────────────┐
           Manager LSTM                                        Worker LSTM
      (once per K steps)                             (every step; also reads the
                  │                                   EPN episodic memory)
          8×8 goal code                                          │
                  │                                    op head → arg heads
        GoalAutoencoder.decode                            (masked, autoregressive)
                  │
              goal g ────────────────────────────────────────────► worker input
```

### Answering "transformer→LSTM for both, or only the manager?"

**Shared transformer trunk, separate LSTMs.** Director's manager and worker both
read the same world-model state and differ only in their heads; the analogue
here is that both read the same demonstration-set encoding. There is no reason
to learn "what does this task look like" twice, and a shared trunk is what makes
the manager's goals live in the same space the worker's reward is measured in.

What is *not* shared is recurrence: the manager runs at the abstract timescale
and advances its LSTM once per goal, the worker advances once per statement.

**The worker also gets the episodic memory; the manager does not.** The worker
is the one that must avoid rewriting the statement it just wrote. The manager
sees a coarser picture by construction. (`use_memory` is a config flag if you
want to ablate it.)

**Manager gradients do not reach the trunk** by default
(`manager_grads_to_trunk: false`). Two objectives pulling on one representation
is the classic way to get a hierarchy that trains neither half well; the
worker's is the denser signal, so it gets the representation.

### The EPN piece

`EpisodicMemory` is the block from the A2C-EPN Alchemy notebook, transplanted:
each past statement is a memory slot, the current state is concatenated onto
every slot before projection, slots are attended with a fill mask, and the
result is max-pooled into the LSTM input. Attention is causal by construction —
only written slots are visible — so no triangular mask is needed. Positional
encoding *is* used here, because for a program being written one statement at a
time the order is the point.

`SetAttention` over the demonstration slots is the same block with the
positional encoding removed, because demonstrations are a set.

### Two deliberate deviations from Director

**No world model, no imagination.** Director learns an RSSM and trains both
policies inside it, because pixel environments are expensive to step. This
environment runs at >1000 steps/second and is exactly simulable — the DSL
interpreter *is* the model. Learning a latent surrogate for something we can run
exactly would buy nothing and cost a large model. So the hierarchy is trained
model-free on real rollouts.

The cost of that choice: Director's goal space sits on top of a world model
trained by reconstruction, which makes it stationary. Here the goal
autoencoder sits on trunk features that the policy loss is also moving. It is
trained on detached features, so it chases the representation rather than
fighting it, but the drift is real and is the first thing to suspect if goals
stop meaning anything. The fix, if needed, is an auxiliary reconstruction loss
on the trunk (predict the target grid from the pair token), which would pin the
representation the way Director's world model does.

**A2C, not Dreamer's actor-critic in imagination.** The reference architecture
this borrows from is A2C, and recurrent PPO needs sequence-aware minibatching
that would double the trainer. Nothing here is sample-limited. Swapping in PPO
means replacing the policy-loss term and keeping everything else — the rollout
already stores behaviour log-probabilities for the ratio.

### Rewards

| Consumer | Reward |
|---|---|
| Worker | `1.0 · Δmax_cosine(goal, state)` — and nothing else |
| Manager (extrinsic) | environment/task reward |
| Manager (exploration) | goal-autoencoder reconstruction error (novelty), normalised |
| Environment | `2.0 · Δ(shaping accuracy) − 0.02 per step − 0.10 per error`, `+5.0` on solving every visible demonstration |

`max_cosine(g, s) = g·s / max(‖g‖,‖s‖)²` — cosine scaled by the ratio of the
smaller norm to the larger, so a worker cannot score by pointing in the right
direction from anywhere.

The environment's shaping is a difference of potentials, so the total shaping
over an episode is `final − initial` and returning the input unchanged earns
nothing.

The shipped configs now enforce strict Director ownership. The Director emits a
new latent before every DSL action. The worker never receives environment
reward in either its observation or its return; it is optimized only for making
the one-step transition requested by that latent. Task quality is therefore
entirely the Director's responsibility. `director_proper: true` validates these
invariants at startup rather than allowing the old hybrid silently.

### Credit assignment across the two timescales

Every goal lasts exactly one environment transition. A new goal terminates the
worker's credit horizon, so its return is the immediate latent-following reward.
The Director acts at that same frequency and receives ordinary task-return GAE;
it alone is responsible for composing those one-step instructions into a
successful program.

The manager's transitions are `(state at goal k, code, Σ rewards over the
window, state at goal k+1)`. Windows are ragged — an episode can end mid-window
— so the target is one backward pass that accumulates discounted reward until it
reaches the next goal boundary and bootstraps there. Both manager critics use
the same recursion on their own reward channel. Hand-checked in
`test_training.py`.

### The invariant that matters most

A recurrent on-policy trainer collects with one forward pass and computes
gradients with another. If those two disagree — a state reset in the wrong
place, a stale feedback vector, a mismatched manager code — the gradient is
computed for a policy that never ran, and **nothing in the loss curve would say
so**. So the rollout stores the log-probabilities it actually sampled under,
the update compares them against the replayed ones, and a test asserts the drift
is below 1e-4.

---

## 6. Crossed, per-role epiplexity dueling

This is now implemented in `DirectorTrainer.duel_update`. Every duel episode
forks identical agent parameters, Adam moments, recurrent state, live
environments, curriculum state, and random-number generators. It then runs two
crossed treatments:

- low-temperature Director + high-temperature worker;
- high-temperature Director + low-temperature worker.

Temperature is the treatment rather than entropy-loss weight because it changes
the first rollout from an identical fork. Each A2C chunk measures critic loss
before and after its optimizer step on the same stored transitions and the same
fixed return targets. The notebook semantics are preserved literally: append
`loss_before - loss_after` to a list for each chunk, then sum that list to get
the duel episode's epiplexity/AUC.

The two roles duel independently. The Director winner supplies
`manager_lstm`, goal head, both manager critics, and their Adam state. The worker
winner supplies the observation trunk, worker recurrence, memory, actor, worker
critic, goal autoencoder, and their optimizer state. This ownership follows the
existing gradient boundary: with `manager_grads_to_trunk: false`, only the
worker trains the shared trunk. The configuration validator refuses the duel if
that boundary is removed.

After recombination all vector environments begin fresh episodes. That avoids
carrying a recurrent state or partially executed DSL program produced by a
different pairing into the hybrid winner. Both forks' environment interactions
still count toward the step budget and throughput; the worker winner owns the
continuing curriculum history because it owns the executed statement policy.

The manager's exploration reward (goal-space novelty) and the worker's action
temperature still operate in genuinely different spaces — *which goals to try*
versus *which statements to try*. The dashboard exposes both AUC comparisons,
the current winner for each role, and cumulative exploratory win rates.

---

## 7. What the first runs showed

Two failures found by running it, both fixed, both worth keeping in mind because
they will recur in any variant of this design.

### Director's absolute goal reward stops the agent from ever halting

Director rewards the worker with `cos(goal, state)` every step. Here that
produced a policy with **halt rate 0.00** and a solve rate pinned at the random
baseline (0.10-0.20 on a rung where random is 0.125), for 300k steps.

The mechanism is specific to this environment and to any environment where the
agent chooses when to stop:

* The goal decoder is trained on states the agent actually visits, so a manager
  can name a goal very close to where the worker already is, and the worker
  collects a large reward for changing nothing. Observed goal reward ~0.88 per
  step while the task reward went nowhere.
* Because the reward is per step, **ending the episode costs the worker the
  rest of its goal reward**. HALT is precisely the action that means "the answer
  is what I have now", and the worker learned never to use it.

The delta form — `cos(g, s_next) - cos(g, s)` — telescopes over the goal window
to `final - initial` similarity. Standing still pays nothing and stopping costs
nothing. With it, the halt rate went to 0.96, mean program length dropped from
"always spend every step" to ~1-2 statements, and the solve rate reached 0.63
(5x random) in **41k** steps rather than being flat at 300k.

### Historical hybrid failure: task reward leaked into the worker

The old hybrid run above then *collapsed*: 0.63 → 0.10 within 10k steps, worker entropy
2.25 → 0.78, and the policy locked onto a single operator (a 0.145 solve rate is
exactly 1/7 — the odds of one fixed choice among seven geometry operators). It
reproduced exactly, at the same point, with Huber value losses and a higher
entropy floor. Neither was the cause.

The cause was Director's worker horizon. Director ends the worker's return at
each goal change, because its worker is paid only for reaching the current goal.
This worker is *also* paid task reward — and the task reward that matters, the
solve bonus, lands on the **last step of the episode**. With `manager_every: 2`,
truncating at goal boundaries meant no statement written more than two steps
before the end was ever credited for the episode ending well. The visible
symptom was the halt rate decaying (0.61 → 0.54 → 0.43 → 0.03) as the agent
drifted from "apply the right operator, then commit" toward flailing until the
horizon, taking the solve rate down with it.

Setting `worker_bootstrap_across_goals: true` — the old worker's horizon is the
episode — removed the collapse. The same configuration that peaked at 0.63 and
fell to 0.10 by 56k steps now sits at 0.59–0.62 through 72k with entropy stable
at 2.2 and no sign of the drift. A `length_bonus` was added at the same time,
paying for the slack left on the clock when a program solves, which gives HALT
a reason to exist beyond a 0.02 step cost.

That result explains the historical hybrid but is no longer the active
configuration. The strict one-step Director removes worker task reward rather
than extending worker credit across Director decisions.

The general lesson, and the one to carry into any Director variant: **the
worker's reward and the worker's horizon are one decision, not two**. Mixing
task reward into a goal-conditioned worker while keeping the goal-length horizon
is incoherent, and it fails in a way that reads as "the policy plateaued and
then destabilised" rather than as a credit-assignment bug.

Huber value losses and the higher entropy floor were kept. They did not cause
the collapse, but a squared error on returns that jump an order of magnitude the
moment the agent starts succeeding is a gradient the *shared trunk* does not
need.

### The hierarchy is not yet earning its keep

`configs/ablation_flat.yaml` is `configs/warmup_small.yaml` with
`worker_goal_weight: 0.0` and `manager_exploration_weight: 0.0` — same network,
same curriculum, same seed, but the worker ignores the manager entirely and is a
flat recurrent A2C agent. On warm-up rung 0 the two runs are indistinguishable:

| steps | Director | flat ablation |
|---:|---:|---:|
| 66k | 0.595 | 0.615 |
| 87k | — | 0.570 |
| 128k | 0.575 | 0.620 |
| 148k | — | 0.605 |
| 240k | 0.660 | — |

The mean goal reward is also slightly *negative* throughout (≈ −0.09), i.e. the
worker drifts away from the manager's goals rather than toward them. On this
rung that is unsurprising and not damning: the task is one operator inside a
six-step episode, so there is no temporal structure for a manager to abstract
over, and a goal set every two steps has nothing useful to say. But it does mean
**nothing measured so far is evidence for Director over flat A2C**, and the
honest reading of the warm-up result is "a recurrent set-encoder agent learns
rung 0 to 0.6", with the hierarchy along for the ride.

The rungs where the hierarchy should start to pay are `w3_objects` and
`w4_full` (three- and four-statement programs, object mechanics) and ARC itself,
where a program has genuine sub-structure — find the objects, then select, then
transform, then paint. Running the ablation alongside every future run is
cheap and is the only way to keep that claim honest. It is also, not
incidentally, the natural harness for the per-role epiplexity duel in §6: the
ablation pair is already the "manager off / manager on" axis.

---

## 8. Known gaps and the order to attack them

1. **`FOREACH` / per-object bodies** — the largest expressivity item (§2).
2. **Goal-space drift** — no auxiliary loss pins the trunk (§5).
3. **Encoder cost at 30×30** — the grid encoder dominates. A per-episode cache
   of the input/target embeddings (they never change within an episode) is
   worth ~3×; the current code re-encodes them each step for simplicity, and
   already batches the whole rollout through the trunk in one pass.
4. **No PPO** — A2C is fine at this scale but is sample-hungrier.
5. **One-step Director horizon.** This is deliberately fixed at one action per
   latent. A learned longer abstraction can be tested later, but may not leak
   task reward or credit directly into the worker.
6. **Single test input.** Tasks with 2–4 test inputs currently put one in the
   machine per episode; the evaluator re-runs the selected program on each, but
   they are not jointly optimised.
