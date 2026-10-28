# The plan for what remains

A working document, 2026-10-28, written at the end of the first library wave. It says what is done, what remains in
each of four tracks (library, render, inventory, methods), in what order, how each wave is launched and merged, what
it costs, and where to stop. It is meant to be run: each wave below has its scope, its agent split and its exit
check, and section 6 is the short procedure for starting one.

Companion documents: `README.md` (the design), `LIB.md` (the library audit), `RENDER.md` (the render audit), and each
chapter's `README.md` (its inventory table).

## 1. Where things stand

| track | state |
| --- | --- |
| Design | settled: slots and modes, the layout by what a page does, the README table format, build order. Open decisions are listed in section 7 |
| Inventory | 52 chapter READMEs, about 1,280 method rows, about 450 library TODOs. Statuses are stale: nothing has been ticked since the library work began |
| Library | merged: option tables (modes, solvers, schedules, losses and penalties), keep and fittedAt, dataset and environment registries, distances, the five bases (trainer, agent, gradient mode, sweep and race, stream), and the domains timeseries, svm, gp, trees, ensembles, plus growth of cluster, decomposition, manifold, learn and linear. Not run: the full gate |
| Render | merged: the session layer (producer and player, eager, seeks, reset, held changes), comparison of runs under one cursor, the sweep panel, the dataset and environment selectors, union controls. Parked: two worktrees (`agent-r3nd1a-panels` with 32 uncommitted files for the new record and panel kinds, `agent-r3nd1b-overlays` with 19 for the rows overlays and curve families) |
| Methods | 34 rows built across 12 early areas under the old chapter folders; one slot-built method (`linear/linearClassifier`) with a test and no page; none moved to the new layout |

Everything on main since the registries merge has been checked only by each agent's own tests, tsc, layering and the
examples runner. The full-gate hold remains on through waves L1–L3 and lifts in wave L4.

## 2. Standing rules for every wave

- **One writer per file.** Agents work on labelled worktrees (`agent-<id>-<word>` under `.claude/worktrees/`); the main
  checkout is the merge point and nothing else. Worktrees and branches are removed as soon as their branch is merged.
- **Tests.** While the hold is on, agents run only their own test files, engine and render tsc, `make barrels`,
  `make layers`, biome on their files, the examples runner on their files, and scoped fixture generation and check for
  their own oracle modules. After the hold is lifted, one `make verify` on main per wave, and none by agents.
- **Merge hotspots**, resolved by hand and always the same way. `docs/structure.md`: the domain count line, the
  domain table rows, and new section numbers (keep both sides, renumber the later one); `index.ts` summaries of shared
  folders (keep both sentences); `fixtures/manifest.json` (`make manifest`); `packages/meta/src/modules/index.ts`
  (`make barrels`). After every merge: `make barrels`, engine tsc, `make layers`, remove the worktree.
- **Agents do not edit** chapter READMEs, `methods/README.md`, `LIB.md`, `RENDER.md` or `PLAN.md`; the inventory pass
  (track C) updates statuses once per wave.
- **Cost.** An agent on a domain of this size uses 550k to 700k tokens and an hour; a wave of seven is about 4M tokens
  and an hour or so of wall time if the machine is free. Plan on that, not on less. Stop a wave early by messaging each
  agent to finish its current piece and report (it works: nothing is lost).
- **Stop rule.** Every wave ends with a merge and a report; nothing is left half-merged. The user decides after each
  wave whether the next is launched.

## 3. Track A: the library

The library is about a quarter done by weight. Four more waves finish what `LIB.md` lists; the order is by what they
unblock. Each wave is seven agents on disjoint domains.

### Wave L1: the supervised and statistical core

| agent | scope |
| --- | --- |
| covariance-anomaly | `lib/covariance` (shrinkage, fast MCD, graphical lasso); `lib/anomaly` (isolation forest and extended, LOF and COF, score combination, contamination threshold) |
| kernel-gam | `lib/kernel-methods` (kernel ridge, online kernel RLS); `lib/gam` (smooth terms, penalised design, backfitting, penalised IRLS, REML and UBRE, EDF, GAMLSS) |
| linear-rest | group lasso and multi-task lasso, regression diagnostics, ordinal regression, mixed models (REML), Theil-Sen, zero-inflated and hurdle GLMs, the multinomial family, in `lib/linear` |
| label-learning | `lib/learn/semi-supervised.ts` (self-, co-, tri-training, label propagation and spreading, Laplacian regularisation, S3VM), `weak-supervision.ts` (majority vote, Dawid-Skene with one-coin and GLAD, label model, noise transition, confident learning, PU risks, proportions, multiple-instance SVM), `noisy-labels.ts` (small-loss selection, co-teaching) |
| active-uncertainty | `lib/learn`: the active strategies still missing (expected model change, BALD, density-weighted, core sets, BADGE), uncertainty (deep ensembles, MC dropout), coresets, drift detectors (DDM, EDDM, ADWIN, KSWIN, PSI), shift estimation (black-box shift, EM priors), density ratios (classifier, KLIEP, uLSIF), calibration additions, conformal under shift |
| stats-survival | `lib/survival` (Kaplan-Meier, Nelson-Aalen, log-rank, Cox with ties, AFT, concordance); `lib/stats` additions (experiments: CUPED, post-stratification, delta method, cluster and switchback designs; design: randomisation, blocking, factorial, ANOVA; shrinkage; Little's MCAR; concentration bounds) |
| sketches-sampling | `lib/sketches` (Bloom, counting Bloom, cuckoo, count-min, HyperLogLog, MinHash and LSH, reservoir, quantile sketches, top-k, skip list); `lib/sampling` variance reduction and gradient estimators |

Exit check: merged, `make layers` clean, and (after the hold) one gate on main.

### Wave L2: systems, graphs, text, structured domains

| agent | scope |
| --- | --- |
| automata-fractals | `lib/dynamics` automata (elementary, Life-like, Langton, sandpile, forest fire), agent-based (Schelling, Nagel-Schreckenberg, Wa-Tor, Sugarscape), lattice gases and Boltzmann, fractals (escape-time sets, IFS with box counting, L-systems and a turtle, named maps and attractors) |
| celestial-control | `dynamics`: celestial mechanics (Kepler's equation, n-body Hamiltonian, restricted three-body, null geodesics), named systems (Kuramoto, Rössler, Thomas, the double pendulum), control gaps (step-response measures, Padé, lead-lag, Pontryagin, RLS, H-infinity, adaptive) |
| graph-spreading | `lib/graph`: spreading (Gillespie epidemics, cascades, influence), percolation, stepped growth, effective resistance, neighbour sampling; `lib/information` (bottleneck, MDL, stepped coders) |
| causal | `lib/causal` (estimation, quasi-experimental designs, heterogeneous effects, sensitivity), `metrics/uplift.ts`, the causal generators in `datasets` |
| rules-patterns-logic | `lib/rules` (CN2, PRISM, RIPPER, rule lists, rule sets), `lib/patterns` (Apriori, FP-growth, Eclat, association rules, PrefixSpan, GSP, subgroup discovery), `lib/logic` extras (production systems, uncertain reasoning, argumentation, description logic, Markov logic, ProbLog, differentiable logic) |
| text-embedding | `lib/embedding` (word2vec, GloVe, knowledge-graph scorers), `lib/text` (n-gram models with smoothers, RAKE and TextRank, inverted index and probabilistic retrieval, record linkage, spelling channel), `lib/learn` click models, `lib/metrics` sequence-labelling and ranking additions |
| numerics-planted | `lib/numerics` (matrix completion, stationary iterations, two-grid, subspace iteration, Jacobi SVD steps, Lanczos stepped, float16 and bfloat16); `lib/datasets` and `data/modifiers`: image modifiers (crops, flips, mixup, cutmix) and the roughly thirty planted problems and generators with truth listed in `LIB.md` section 1 (leaks, duplicates, shifts, panels, discontinuity designs, transactions, tagged corpora, captioned scenes, in-context tasks, coupled series, spoken digits), registered with tags |

### Wave L3: neural, reinforcement, search, the long tail

| agent | scope |
| --- | --- |
| nn-blocks | `lib/nn`: energy blocks (Hopfield, RBM), continuous blocks (CNF, Hamiltonian), operators (DeepONet, FNO), forecasting (N-BEATS), recommender blocks, graph layers (relational, equivariant, graph transformer), masked convolutions, receptive fields, SWAG |
| nn-training | `lib/nn` parametrisations (LoRA, adapters, prefix tuning), adversarial attacks and certification, `lib/learn` privacy (per-example clipping) and privacy attacks, a multi-module trainer for GANs, `lib/optim/landscape.ts` (landscape slices with filter normalisation) |
| losses-generative | `lib/losses` (CTC, preference objectives, invariance penalties, robust classification losses, focal and smoothing as table entries, VQ, noise-contrastive, sliced score matching, Barlow and VICReg, evidential); `stochastic/diffusion-models.ts` (schedules, VP and VE SDEs, DDPM and DDIM samplers as algorithms, guidance); `dists/bijectors` additions |
| explain-vision | `lib/explain` internals (logit and tuned lens, activation patching, feature visualisation), `lib/vision` stepped forms (Canny stages, Hough peaks, watershed, mean shift, structure from motion, calibration iterations) |
| rl-search | `lib/agents` and `lib/environments` additions (tile coding and semi-gradient learners, goal-conditioned and HER, best-arm bandits, sliding-window UCB, off-policy MC, random walk and Baird, changing mazes), `lib/search` and `lib/problems` (particle swarm, NEAT, puzzles, tours, SAT instances) |
| timeseries-rest | the rest of `lib/timeseries` (cointegration with the Johansen and MacKinnon tables, shapelets, the feature set, EGARCH, Box-Cox MSTL, multiplicative ETS intervals), `lib/pgm` additions (rating systems, IRT, HMM variants, online CRF trainers, Swendsen-Wang), `lib/manifold` PaCMAP and TriMap |
| core-gaps | `lib/learn` additions not in L1 (`theory.ts`: PAC, VC, Rademacher, margin, stability, PAC-Bayes, regret bounds, bias-variance, degrees of freedom; `feature-selection.ts` knockoffs, selective inference, feature synthesis, mRMR, Relief; `model-selection/` thresholdout, the ladder, corrected resampled t-test, 5×2 CV), `lib/optim` additions (`online.ts` expert learners: weighted majority, FTL, FTPL, fixed share; dual coordinate descent for linear SVM), `lib/decomposition` and `lib/manifold` tail items, plus the TODOs the waves above surface and `LIB.md` section 2.4 |

### Wave L4: the gate, the fixes, the move

One agent (or me) after the hold is lifted: run `make verify` on main, triage failures (they will be few, cross-domain,
and mostly import or barrel collisions), fix. Then the structural moves still owed from the design: the built areas
into the new layout (section 5, M0), and the stale `_parked` references. This wave is small.

Library total: three waves of seven plus a short fix wave, about 12M tokens and 3 to 4 working sessions.

## 4. Track B: render

Render is a smaller track with a clear order (`RENDER.md` section 4). Two agents per wave, disjoint files.

### Wave R1: panels and records (two agents)

- **panels:** the new record kinds in `lib/exposition` (`vectorField`, `strip`, `table`, `structureSteps`,
  `pointsSteps`, `complexPlane`, `simplex`, `reachability`, a scene per step) with their headless and React views, the
  matching `Panel` kinds (`field`, `strip`, `table`, `images`, `graph`, matrix marks), one gallery recipe each. The
  parked worktree `agent-r3nd1a-panels` holds a start (32 files, uncommitted): its first act is to read it, keep what
  type-checks, finish and commit.
- **overlays-controls:** the rows overlays as one mechanism, the curve families as options of `curves` and `sweep`,
  validity greying and the model selector, the comparison over a slot, and the first exemplar page (the linear
  classifier, six solvers, one player). The parked worktree `agent-r3nd1b-overlays` holds a start (19 files).

### Wave R2: specialised panels and infrastructure (two agents)

- **specialised:** zoomable image, accumulating 3-D point cloud, complex plane, simplex, loss-landscape slice, stability
  region, equation against truth, audio in sync, transcript, samples over training, small multiples as a layout
  feature, the inner session.
- **infrastructure:** the worker producer (sessions computed off the main thread, the tape streamed), page export (PNG,
  GIF, tape JSON), narratives (one tape, several pages).

Render total: two waves of two, about 2.5M tokens. The gallery stays the test bed: each piece is shown on a toy method
before any real method uses it.

## 5. Track C: the inventory, and Track D: the methods

### C. Keeping the inventory true (a small standing job)

The chapter tables are the to-do list, so they must tick. One agent per wave runs the inventory pass: parse every
chapter README, check each `needs from lib` citation against the library's exports (the checks the table agents ran
by script), turn `TODO` into a plain citation when the piece has landed, set `status` when a method exists, and
rewrite the counts at the top of `PLAN.md` section 1 and `LIB.md` section 2. Make it a script
(`scripts/inventory.ts`, `make inventory`) so it runs without an agent afterwards. Output: a report of what changed
and a table of what is now buildable (every `needs` cell resolved), which is the input to track D.

### D. The methods

A method is a few dozen lines over the library (design section 1). The work is mechanical once its pieces exist, so it
is done in bulk, chapter by chapter, from the tables.

**M0: the move and the exemplars (one agent, small).**
- Move the twelve built areas into the new layout by `git mv` and an import path (the map is in design section 3).
- Delete the chapter-folder placeholders as their rows move; the READMEs move with them to
  `docs/methods/chapters/` or stay as the area's README, as the user decides (section 7).
- Three exemplar methods written by hand and kept as the templates: a task method (the linear classifier with six
  solvers: done), a problem method (a small network on a planted problem over the trainer base), and a study (a
  regularisation sweep over the sweep base). Each with its smoke test and a gallery page (needs R1).

**M1 onward: bulk waves.** Seven agents per wave, each given one or two chapters whose tables are fully buildable,
the three templates, and the rules: one file per method named in its row, the contract is the row's `tracks` column,
the smoke test runs a session and a probe at an early step, no fixtures, the row's status updated by the inventory
pass. The order follows the library: supervised and unsupervised basics (every estimator in the new domains becomes a
method in a few lines: about 150 methods), optimisation and linear algebra and probability and statistics studies
(about 120), sequence models and Bayesian methods (about 80), time series (about 50), the neural trainer pages (about
100), agents and search (about 80), data, signals, vision (about 150), applications (about 160). Roughly eight waves;
each is about 3M tokens because methods are small. Stop after wave three if the aim is a good first release: that
covers the core third of the inventory.

## 6. How to start a wave (the short procedure)

Say "go" with the wave name, and the following happens with no further questions.

1. **State check** (two minutes): `git status` clean, `git worktree list` has no leftovers, `make barrels` and
   `make layers` clean. If the hold is lifted, one `make verify` on main first; fix what it shows before launching.
2. **Worktrees:** one per agent, `git worktree add .claude/worktrees/agent-<id>-<word> -b agent-<id>-<word> main`.
3. **Briefs:** each agent gets the standard brief (below) plus its scope row from this document.
4. **Merge as they report:** `git merge --no-ff`, resolve the hotspots of section 2, `make barrels`, engine tsc,
   `make layers`, remove the worktree and branch, one line to the user per merge.
5. **Close:** the inventory pass, then a short report (what merged, what was left out, the next wave's cost).

**The standard brief** (copy, fill the angle brackets):

> You are building <library code | render code | methods> for mdml in /Users/nialltwomey/workspace/whitebox-js,
> working ONLY in the worktree .claude/worktrees/<name> on branch <name> (from main). Read first, binding: CLAUDE.md;
> docs/structure.md for the domains you touch; packages/engine/src/methods/README.md, LIB.md (or RENDER.md) and the
> chapter README tables for your scope, which name the exact functions, modes and tracks wanted. Old parked code for
> completeness: `git show 933f0dd8:_parked/...` (read, never restore wholesale). Conventions: strict TypeScript with
> isolatedDeclarations, TSDoc with a checked @example on every core export, a summary paragraph opening every file and
> index.ts, tests beside code (unit, property, golden against the reference library in the `oracle-core` uv group),
> `make barrels` after adding files, docs/structure.md updated for every new file with a new subsection per new domain,
> every iterative routine an Algorithm whose state serialises, estimators on the iterative base, no console, no
> Math.random, no clocks, no local copies of helpers. Test rules: no `make verify`, no repo-wide vitest; your own
> test files, tsc, barrels, layers, biome, the examples runner, scoped fixture generation and check. Never add a uv
> group. Commit on your branch when clean, trailers `Co-Authored-By: Claude <model> <noreply@anthropic.com>` and
> `Claude-Session: <the session url>`. Do not merge, push or rebase; do not edit READMEs or the plan documents.
> Priority order inside your scope: finish a piece fully before starting the next; if told to stop, finish the piece in
> hand, commit what type-checks and passes, and report. Report: commits, public API excerpts, paths logged, goldens
> added, what you left out and why, that the full gate was not run.

## 7. Decisions still open (none blocks starting wave L1)

1. Whether the chapter READMEs stay in their folders as each area's README, or move to `docs/methods/chapters/` once
   the area's methods exist (recommended: stay, and become the area's README).
2. Whether the two parked render worktrees are finished by the R1 agents (recommended) or discarded.
3. Where the planned domains land when an agent finds a better home than `LIB.md` names (recommended: the agent's
   choice, recorded in its report and in structure.md).
4. Whether to push the 142 unpushed commits before the next wave (CI would run once on code changes; the user's rule is
   no push that triggers extra actions for docs-only changes, and these are mostly code).
5. How far to go: the core third (library waves L1 and L2, render R1, methods M0 to M3, about 20M tokens) or the whole
   inventory (everything above, about 45M tokens).

## 8. Sizes at a glance

| stage | agents | tokens (est.) | result |
| --- | --- | --- | --- |
| gate on main and fixes | 1 | 0.6M | the first verified state since the registries |
| L1, L2, L3 | 7 each | 4M each | the library of `LIB.md`, minus a tail |
| L4 | 1 | 0.6M | gate, fixes, the move |
| R1, R2 | 2 each | 2.5M in all | the panels of `RENDER.md` |
| C (each wave) | 1 | 0.3M | the tables tick, the buildable list |
| M0 | 1 | 0.6M | the layout, three templates |
| M1 to M8 | 7 each | 3M each | the methods, in bulk |

Total: about 45M tokens for everything; about 20M for the core third.
