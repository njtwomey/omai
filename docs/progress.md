# Progress log

Status at the pause on 2026-10-24 (evening). `docs/structure.md` is the plan; this file records what is built, what
each workstream left, and the queue for resuming. Everything below is on `main`; every merge passed `make verify`.

## Where things are

| layer                                    | state                                                                                                                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| core (`packages/engine/src/lib/`)        | all 39 domains of the plan, including `exposition` (typed records) and `datasets` (moved from methods, with themed collections)                |
| methods (`packages/engine/src/methods/`) | phases 1 to 4 complete; phase 5 partly (classical AI, evolutionary, patterns, complex systems, neural, generative flows and VAEs)              |
| views (`packages/render`)                | figure model, charts, layouts, structures, maths, animation and export, controls, interaction, and a view for every exposition record kind     |
| hub (`packages/hub`)                     | over 150 entries in 14 collections, run in a worker, state in the URL, training played live                                                    |
| site (`apps/site`)                       | one app, `make dev` at http://localhost:5300: gallery (about 90 recipes in 16 sections), hub, docs (generated from TSDoc, executable examples) |
| tooling                                  | `make verify` (scoped gate, at most two test runs at once, runs changed files' examples), `make examples`, site tests in Vitest                |

## Done, by workstream

- **Render core**: theme and palettes, the figure model (grids, aspects, limit policies, linking), chart layers (lines,
  points, bars with aligned error bars, distributions, smooth heatmaps, fields, functions, evaluation, multivariate,
  tuning studies), maths (KaTeX, macros, live equations, derivations), trace figures, animation timelines and
  playback, export (SVG, compressed PNG, GIF, WebM, animated SVG), `FigureFrame` (one expand and one export per figure).
- **Gallery and shell**: sidebar mirroring folders (collapsed except the current path, filter, folder links),
  previous and next with arrow keys, dark by default with a top-right toggle; controls (number field with 11-step
  dropdown, cycling selects, control panels, controls from a space, dataset picker, method controls), interaction
  (drag, constrained drag, add points, probe, brush, linked figures), the player (fps menu, scrubber), layouts
  (layered, trees, force, …), structure diagrams (graphical models, architectures, U-Nets, Bayes nets with tables,
  plates, factor graphs, self-loops), scenes, worlds (grid worlds, mazes, ray casting with WASD/arrows and a ray count).
- **Exposition records and views**: 29 record kinds (paths, trees, ensembles, margins, surfaces with training steps,
  clustering/mixture/embedding steps, components, posteriors, tables, pipelines, searches, episodes, boards,
  populations, images, curves, diagrams, value grids, confusion matrices, state sequences, …), each with a view and a
  gallery recipe; records carry their input rows, names and directions so views draw from the record alone; every
  iterative trainer with a decision surface records it through training and the views play it.
- **Methods**: supervised (linear and GLM family with paths and CV, trees, ensembles, SVMs, neighbours, naive Bayes,
  discriminants, GPs, kernel ridge, multiclass), unsupervised (clustering, mixtures, density, decomposition, manifolds,
  covariance, anomaly), paradigms (semi-supervised, online, active learning), probabilistic models (HMMs, LDA,
  conjugate models, ratings, state space), the RL playground (12 pairings that learn in seconds), tuning (grid,
  random, TPE, GP searches with pruning), classical AI (search, adversarial, CSPs, planning), evolutionary, patterns,
  complex systems, neural (MLPs, perceptron, backprop by hand, optimiser and training studies, LeNet, RNNs),
  generative (RealNVP, VAEs). Goldens against scikit-learn, scipy, statsmodels and hmmlearn where cheap.
- **Docs**: `mdml-meta` (catalogue extraction, example runner, coverage), docs section mirroring the source tree,
  highlighted, executable examples with `// =>` outcomes, search.
- **Core fixes found on the way**: LP verdicts cross-checked, PELT exact with a minimum segment, symbolic product
  parsing, NUFFT accuracy, homography with four points, B-spline validation, BVH tie-breaking, `eigh` deflation,
  Kalman EM's M-step with fixed matrices, flaky property tests made sound.

## Left, by workstream (from the agents' final reports)

- **Generative**: GANs (vanilla, WGAN-GP, mode-collapse toggle, coverage diagnostics), DDPM diffusion, then
  energy-based or score models and an autoregressive model; flows beyond RealNVP; conditional and denoising
  autoencoders; sharper digit samples (KL warm-up, longer budget).
- **Classical AI**: the 8-puzzle and other puzzles as path search, local search, beam search; expectimax and MCTS;
  GraphPlan and partial-order planning; TSP GA, neuroevolution, quality diversity; Eclat, sequences, subgroup
  discovery; more complex systems (boids, epidemics, sandpiles); the remaining gofai files; goldens (mlxtend, AIMA).
- **Neural**: an L-BFGS MLP solver, character-level language models, the other vision and sequence models of the plan;
  `curves` records for the regularisation and learning-rate studies; gallery recipes for the curves, diagram,
  value-grid and confusion-matrix views; a screenshot pass over the switched neural pages.
- **Records and views**: training steps for multinomial/one-vs-rest logistic regression and multiclass SVC; confusion
  matrices outside neural classifiers; pipeline card text overflow, very wide trees, narrow colour-bar labels; one
  dendrogram layout instead of two; a per-call knob for recording cost.
- **Methods quality**: probit on separable data runs to its iteration limit (slow test); the casino HMM decodes 48% at
  seed 0; kernel k-means cannot separate concentric rings (documented limit).
- **Tensor performance** (not started beyond a benchmark, kept in the session scratchpad): fast paths for same-shape,
  scalar, row and column broadcasting in `lib/tensor/_cpu` (`map2`, `binary`, `compare`, `where`, `reduce`,
  `broadcastTo`), with a property test of bit-identity; then re-profile matmul and convolution. A trial gave about
  10% on flows.
- **Render leftovers**: MP4 export, GIF size, legend icons for scatter series, empty space around some 3-D scenes.
- **Docs**: examples in a worker (stop infinite loops), figures and traces drawn in example outputs, linked type names,
  the concept taxonomy and how-tos.

## Queue for resuming

1. Tensor-core performance pass (above).
2. Generative: GANs and diffusion.
3. Phase 5 groups not started, one per agent, each with hub entries and the process shown over time: causal inference
   and experimentation; scientific ML; retrieval and recommendation; information, algorithms and foundations studies;
   optimisation and Monte Carlo studies; learning theory; the domain areas (time series, text, vision, audio, signal).
4. The "left" items above, by priority: gallery first, then hub, then docs.

## How the work ran (for resuming)

- Agents work in labelled worktrees (`git worktree add .claude/worktrees/<label> -b wt/<label> main`), each site-facing
  one served live with `vite --port 530x` from its worktree so the user can watch; main is served at :5300.
- The lead merges each branch with `--no-ff --no-commit`, regenerates the fixture manifest or hub catalogue on
  conflict, gates with `make verify`, and commits only on success.
- Standing preferences: one site; dark by default; navigation mirrors directories (never tiers); one frame per
  figure; every recipe has a live control; training shown over time; tidy and direct.
