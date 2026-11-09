# Blind spots of the survey: which omai domains had a reference lens, which had none, and what no name comparison can see

Part of the [completeness survey](README.md). The survey measures omai against named libraries, so it can only report on a
domain through a library that was enumerated for it. This file lists every folder of `packages/engine/src/lib` and the areas
of `omai-render` against the libraries of the first and second pass, says which domains had no lens at all, names the closest
libraries or textbooks to measure them against, and gives a short spot-checked list of what a learner would expect there and
omai lacks. The second pass was triggered by the owner's question: have we covered everything, or is there nothing we have not
at least partly covered?

The honest answer is neither. A domain with a lens has a number. A domain without one has a hole in the picture, and this file
is the map of the holes. Even with a lens, the survey is never complete (the last section says why).

## Domain map

"Introspected" means the names came from the installed package; "memory" means a list written from knowledge of the library.
"none" means no reference list was made.

| omai domain                                                | lens (first pass, second pass)                                                                                         | how                               | verdict of the lens                                                                                                 |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| agents                                                     | Stable-Baselines3, CleanRL, PettingZoo; d3rlpy, imitation, learn2learn                                                 | memory                            | thin: all from memory                                                                                               |
| anomaly                                                    | PyOD; river, deepchecks, Darts anomaly                                                                                 | memory                            | thin: all from memory                                                                                               |
| autodiff                                                   | torch.autograd, torch.func; optax second order                                                                         | introspected                      | good                                                                                                                |
| base                                                       | none                                                                                                                   |                                   | infrastructure (errors, JSON, numerics helpers): no lens needed                                                     |
| causal                                                     | DoWhy, EconML; causal-learn, gCastle                                                                                   | memory; introspected              | good for discovery, thin for estimation                                                                             |
| cluster                                                    | scikit-learn; hdbscan, scikit-learn-extra, river                                                                       | introspected; memory              | good                                                                                                                |
| context                                                    | none                                                                                                                   |                                   | infrastructure (keys, defaults)                                                                                     |
| covariance                                                 | scikit-learn.covariance; the third pass (`covariance.md`): pyRiemann, skfolio, skggm, robpy, geomstats, the R packages | introspected; source read; memory | good for the sklearn estimators; the wider area was thin until the third pass: see covariance.md                    |
| data                                                       | torch.utils.data; skorch, torchtext, albumentations                                                                    | introspected; memory              | partial: tf.data, Hugging Face datasets and Dask not enumerated                                                     |
| datasets                                                   | scikit-learn.datasets, torchvision.datasets, statsmodels data (in first pass)                                          | introspected                      | partial: OpenML, UCI, seaborn and TFDS catalogues not enumerated                                                    |
| decomposition                                              | scikit-learn                                                                                                           | introspected                      | good                                                                                                                |
| distances                                                  | scipy.spatial.distance, sklearn.metrics.pairwise                                                                       | introspected                      | good                                                                                                                |
| dists                                                      | scipy.stats, torch.distributions, NumPyro; TFP, PyMC, Pyro                                                             | introspected; memory              | good                                                                                                                |
| dynamics (ODE, control, filtering)                         | scipy.integrate, python-control, filterpy, pysindy, ruptures                                                           | introspected                      | good                                                                                                                |
| dynamics (PDE, automata, celestial, fractals, lattice gas) | none for most (scipy.integrate has ODEs only)                                                                          |                                   | **no lens**; see below                                                                                              |
| embedding                                                  | gensim; sentence-transformers, PyG knowledge-graph models                                                              | introspected; memory              | good for word vectors                                                                                               |
| ensembles                                                  | scikit-learn; XGBoost, LightGBM, CatBoost features                                                                     | introspected; memory              | good                                                                                                                |
| environments                                               | gymnasium, PettingZoo; OpenSpiel game names                                                                            | introspected                      | good                                                                                                                |
| explain                                                    | SHAP, Captum; LIME, interpret                                                                                          | introspected; memory              | good                                                                                                                |
| exposition                                                 | none                                                                                                                   |                                   | omai's own record types: no lens exists                                                                             |
| fairness                                                   | fairlearn                                                                                                              | introspected                      | thin: AIF360 is installed and was not enumerated                                                                    |
| fields                                                     | none                                                                                                                   |                                   | **no lens**; see below                                                                                              |
| foundations                                                | none                                                                                                                   |                                   | **no lens** (textbook mathematics); see below                                                                       |
| gam                                                        | statsmodels GLMGam only                                                                                                | introspected                      | thin: pyGAM and R's mgcv not enumerated                                                                             |
| games                                                      | nashpy, OpenSpiel                                                                                                      | introspected                      | good                                                                                                                |
| gp                                                         | GPyTorch, GPflow, BoTorch                                                                                              | introspected; memory              | good                                                                                                                |
| graph                                                      | NetworkX; PyTorch Geometric                                                                                            | introspected; memory              | good                                                                                                                |
| graphics                                                   | none (shapely and trimesh only for geometry)                                                                           |                                   | **no lens**; see below                                                                                              |
| information                                                | pyitlib, NPEET                                                                                                         | introspected                      | good for measures; coding theory has no library lens                                                                |
| kernel-methods                                             | scikit-learn kernel_ridge                                                                                              | introspected                      | thin: KRLS, NORMA, online kernels have no library                                                                   |
| kernels                                                    | scikit-learn kernels, GPyTorch, GPflow                                                                                 | introspected                      | good                                                                                                                |
| learn                                                      | scikit-learn non-model modules; modAL, small-text, Snorkel, cleanlab, Opacus, River, Avalanche                         | introspected; memory              | good                                                                                                                |
| linear                                                     | scikit-learn, statsmodels                                                                                              | introspected                      | good                                                                                                                |
| logic                                                      | ProbLog (found covered)                                                                                                | memory                            | **thin**: clingo, python-sat, pyperplan, python-constraint are installed and not enumerated; see below              |
| losses                                                     | torch.nn losses, optax losses, pytorch-metric-learning                                                                 | introspected; memory              | good                                                                                                                |
| manifold                                                   | scikit-learn, UMAP, TriMap, PaCMAP                                                                                     | introspected; memory              | good                                                                                                                |
| metrics                                                    | torchmetrics, scikit-learn.metrics; seqeval, MAPIE metrics                                                             | introspected; memory              | good                                                                                                                |
| mixture                                                    | scikit-learn                                                                                                           | introspected                      | good                                                                                                                |
| nn                                                         | torch.nn; timm, torchvision.models, torchaudio, kornia                                                                 | introspected; memory              | good for layers, missing named architectures                                                                        |
| numerics                                                   | scipy (linalg, sparse, special, interpolate, integrate, fft, ...)                                                      | introspected                      | good                                                                                                                |
| optim                                                      | scipy.optimize, torch.optim; optax, cvxpy, pymoo, cma, nevergrad                                                       | introspected                      | good                                                                                                                |
| patterns                                                   | mlxtend, pysubgroup                                                                                                    | memory; introspected              | good                                                                                                                |
| pgm                                                        | pgmpy, hmmlearn; pystruct, CRFsuite                                                                                    | introspected; memory              | good                                                                                                                |
| problems                                                   | none                                                                                                                   |                                   | **no lens** (AIMA problem instances); see search                                                                    |
| probprog                                                   | NumPyro, PyMC, TFP; Pyro, BlackJAX                                                                                     | introspected; memory              | good                                                                                                                |
| random                                                     | numpy.random.Generator (methods, by group)                                                                             | introspected                      | good for distributions; key splitting has no counterpart to compare                                                 |
| rl                                                         | gymnasium, Stable-Baselines3, CleanRL; d3rlpy                                                                          | introspected; memory              | good                                                                                                                |
| rules                                                      | imodels, interpret                                                                                                     | memory                            | thin: from memory                                                                                                   |
| sampling                                                   | NumPyro, ArviZ; BlackJAX                                                                                               | introspected                      | good                                                                                                                |
| scene                                                      | none                                                                                                                   |                                   | omai's structure layer: no lens exists                                                                              |
| search                                                     | none                                                                                                                   |                                   | **no lens** (AIMA code, simpleai not installed); see below                                                          |
| signal                                                     | scipy.signal; librosa, torchaudio                                                                                      | introspected; memory              | good                                                                                                                |
| sketches                                                   | datasketch                                                                                                             | introspected                      | good                                                                                                                |
| space                                                      | gymnasium.spaces, Optuna distributions                                                                                 | introspected                      | good                                                                                                                |
| stats                                                      | scipy.stats, statsmodels; pingouin                                                                                     | introspected                      | good                                                                                                                |
| stochastic                                                 | sdeint (SDE solvers only)                                                                                              | introspected                      | **thin**; see below                                                                                                 |
| survival                                                   | lifelines, scikit-survival                                                                                             | memory                            | thin: from memory                                                                                                   |
| svm                                                        | scikit-learn.svm                                                                                                       | introspected                      | good                                                                                                                |
| symbolic                                                   | SymPy                                                                                                                  | introspected, by area             | area level only: 847 names judged in groups                                                                         |
| systems                                                    | python-control                                                                                                         | introspected                      | good                                                                                                                |
| tensor                                                     | numpy namespace (by group); torch tensor functions only through torch.nn, optim, linalg, fft, special                  | introspected                      | partial: `torch.*` tensor functions not enumerated                                                                  |
| text                                                       | NLTK, gensim; torchtext, seqeval                                                                                       | introspected; memory              | partial: spaCy and Hugging Face tokenizers skipped on purpose                                                       |
| timeseries                                                 | statsmodels, sktime, tslearn, prophet; stumpy, ruptures, Darts, GluonTS, neuralforecast                                | introspected; memory              | good                                                                                                                |
| trace                                                      | none                                                                                                                   |                                   | omai's own tape: no lens exists                                                                                     |
| trees                                                      | scikit-learn tree; imodels, river                                                                                      | introspected; memory              | good                                                                                                                |
| tune                                                       | Optuna; scikit-optimize, BoTorch                                                                                       | introspected; memory              | good                                                                                                                |
| vision                                                     | scikit-image, torchvision; kornia, albumentations                                                                      | introspected; memory              | good (OpenCV itself not enumerated)                                                                                 |
| worlds                                                     | none                                                                                                                   |                                   | **no lens** (grid-world and maze utilities)                                                                         |
| omai-render (charts, views)                                | ArviZ plots (first pass), yellowbrick, scikit-plot, mlxtend plotting                                                   | memory                            | partial: seaborn, Plotly, Altair, matplotlib galleries not enumerated; the chart list in `repertoire.md` is by hand |

Domains with no lens at all: base, context, exposition, scene, trace (omai's own infrastructure, nothing to compare), and
fields, foundations, graphics, problems, search, worlds, the non-ODE part of dynamics. Domains with a lens that is
too thin to call a measurement: logic, stochastic, rules, survival, gam, kernel-methods, fairness, anomaly, agents.

## Quick passes where there was no lens

For each domain below the closest libraries or textbooks are named, and a few things a learner would expect were looked up in
the export list of omai-engine (`has` by name, then a search of the source). Same honesty rules as the rest of the survey: a
name found is not a behaviour verified, and a name not found was searched for under its aliases.

### search and planning (`search`, `problems`, `worlds`)

Reference: Russell and Norvig (AIMA) and its code, `simpleai`, `python-constraint`, `pyperplan` (the last two are in the oracle
environment but were not enumerated). All of the uninformed and informed search family are present (breadth-first, depth-first,
depth-limited, iterative deepening, uniform cost, bidirectional, greedy best-first, A*, weighted A*, IDA*, RBFS, beam), local
search (hill climbing with restarts, simulated annealing, tabu, late acceptance, local beam, genetic algorithms, evolution
strategies, genetic programming, MAP-Elites, novelty search, ant colony), adversarial search (minimax, alpha-beta, negamax,
expectimax, MCTS), constraint satisfaction (AC-3, backtracking, min-conflicts), planning (STRIPS and PDDL parsing, forward
and regression planning, GraphPlan, relaxed-plan heuristics, HTN, partial-order planning) and problem instances (8-puzzle,
N-queens, Sudoku, map colouring, TSP with Held-Karp, SAT generators, Hanoi, blocks world).

Not found: SMA* (memory-bounded A*), AND-OR search for nondeterministic problems, online search (LRTA*, online DFS), local
search in continuous space beyond what `optim` has, any-angle path planning (Theta*, Jump Point Search) and sampling-based motion
planning (RRT, PRM), D* Lite. Size S each except motion planning (M). Priority low to medium; RRT and D* Lite are the two a
robotics or AI course reaches for after A*.

### logic and knowledge (`logic`)

Reference: AIMA chapters 7 to 10 and 12, Clingo/ASP, SAT solvers (python-sat), Prolog systems, ProbLog, Description Logic
reasoners. Present: propositional logic with CNF and Tseitin, DPLL, CDCL, WalkSAT, MaxSAT, model counting, Quine-McCluskey,
first-order unification and resolution, Prolog-style SLD resolution with a standard library, Datalog, answer-set programming
(stable models), argumentation frameworks, description logic (ALC tableau), fuzzy logic and possibility theory, Dempster-Shafer
evidence, production systems with Rete, ILP (FOIL, Progol, Metagol, version spaces, differentiable ILP), Markov logic networks
with learning, neural theorem provers, and ProbLog-style probabilistic logic programs with EM learning (an early guess in this pass
said ProbLog was absent; reading the source showed it is present).

Not found (searched by name and in the source): binary decision diagrams (BDD, OBDD), SMT theories (linear arithmetic with a simplex core,
congruence closure), modal and temporal logic model checking (CTL, LTL on Kripke structures) and planning as SAT. BDDs and a CTL model
checker are the two a logic-for-computer-science course teaches; each is S to M. Not checked: equality reasoning (paramodulation,
superposition) and constraint logic programming over finite domains.

### stochastic processes (`stochastic`)

Reference: Ross, Karlin and Taylor, Oksendal; `sdeint` (SDE schemes, introspected), `statsmodels` for some processes. Present:
Poisson, renewal, compound Poisson, Hawkes, general point processes, Galton-Watson branching, CTMCs, Brownian motion and bridges,
fractional Brownian motion, Ornstein-Uhlenbeck, geometric Brownian motion and Black-Scholes, jump diffusions, Ito and Stratonovich
integrals, Ito's lemma, SDE solvers, Gaussian-process sampling, percolation.

Also present, found by searching the source after a first guess that they were missing: queueing theory (M/M/1, M/M/c, M/G/1, Erlang
B and C, a traced G/G/c simulation with Little's law), Gillespie's algorithm for birth-death and continuous-time chains, the Polya
urn and random measures (Dirichlet and beta processes, Indian buffet), mixing times and the spectral gap of Markov chains.

Not found: Jackson networks and other queueing networks, coupling from the past (perfect sampling), martingale convergence and
optional-stopping demonstrations as named functions, and the higher-order stochastic Runge-Kutta schemes of `sdeint` (strong order
1.5). The domain is deep; the lens was thin, and a textbook pass found little missing.

### information theory and coding (`information`)

Present in depth: entropy and divergences, mutual information with nearest-neighbour estimators, transfer entropy, channel
capacity, rate-distortion, source coding (Huffman, arithmetic, range, Lempel-Ziv, Burrows-Wheeler), error-correcting codes (Hamming,
Reed-Solomon, LDPC, convolutional with Viterbi), the information bottleneck, MDL and NML. Missing against `pyitlib` and `dit`
(the latter not installed): multi-information (total correlation), dual total correlation, the variation of information, lautum
information, CorEx, and partial information decomposition (redundancy, synergy). S each; priority medium for total correlation and
the variation of information.

### neural fields and graphics (`fields`, `graphics`)

Reference: nerfstudio, tiny-cuda-nn, gsplat, Instant-NGP (fields); Shirley's "Ray Tracing in One Weekend", PBRT, Foley and van
Dam (graphics). No library was enumerated for either. Present in `fields`: implicit neural representations (SIREN, Fourier
features, hash-grid encodings), NeRF with volume rendering and hierarchical sampling, 3-D Gaussian splatting with adaptive density
control, PINN losses with loss balancing (`losses/physics.ts`). Present in `graphics`: BRDFs (Lambert, Phong, Blinn-Phong,
Cook-Torrance with GGX, Fresnel), path tracing, BVH, mipmapped textures, polygon clipping, depth buffer, cameras, cosine-hemisphere
sampling, Bezier curves (`foundations`).

Not found: marching cubes and surface extraction from a field (marching squares exists as `contourLines`), radiosity, ambient
occlusion, normal and bump mapping, tone mapping, subdivision surfaces (Catmull-Clark, Loop), anti-aliasing by supersampling or MSAA
as a named step, neural SDF training with the eikonal loss (the SDF test shapes exist), Poisson surface reconstruction. Priority
low; marching cubes is the one that pairs with the SDF datasets.

### dynamics beyond ODEs (`dynamics`)

Reference: scipy.integrate for ODEs (covered), textbooks (Strogatz, Hairer and Wanner, Press et al., Bate-Mueller-White),
py-pde, FEniCS and Golly for the rest (none installed or enumerated). Present: ODE solvers (RK, DOP853, Radau, BDF, LSODA, symplectic,
dense output, events, adjoint), finite-difference, method of lines, Crank-Nicolson, Chebyshev collocation, FEM, multigrid, chaos
(Lorenz, Rossler, bifurcation diagrams, Lyapunov spectra, Poincare sections), cellular automata (elementary, Life, Langton's ant,
sandpile, forest fire, Schelling, Sugarscape, traffic, Wa-Tor), lattice gas and lattice Boltzmann, escape-time fractals, IFS,
L-systems, Newton fractals, Kepler, N-body, epidemics, flocking, Kuramoto.

Not found: delay differential equations, Lambert's problem (orbit determination from two positions), incompressible
Navier-Stokes solvers (projection method) beyond lattice Boltzmann, molecular dynamics with Lennard-Jones and neighbour lists,
neuron models (Hodgkin-Huxley, FitzHugh-Nagumo, Izhikevich, leaky integrate-and-fire) and spiking networks, reaction-diffusion
patterns as a named model (the method-of-lines machinery is there). The neuron models are the one a learner expects most: small,
visual and a good fit for the tape. DDEs and molecular dynamics follow.

### foundations (`foundations`)

No library lens: it is textbook mathematics (calculus, vector calculus, combinatorics, number theory, complex analysis, linear
algebra by elimination with steps, curves and surfaces, conics, Bezier). SymPy is the closest tool and its area-level comparison
is in `adjacent.md`. No gap list was made beyond the spot checks above.

### text beyond NLTK and gensim

Present: tokenisers of every school, stemmers, lemmatiser, n-gram models with smoothing, TF-IDF, BM25, word2vec, GloVe, fastText,
spelling correction, TextRank and RAKE, CKY, Earley, PCFG training, dependency parsing, record linkage, hashing and MinHash,
Punkt, alignment (Needleman-Wunsch, Smith-Waterman). Not found: part-of-speech taggers (the HMM, perceptron and Brill taggers that
NLTK ships; the CRF and HMM parts for building one exist), word sense disambiguation (Lesk), lexicon sentiment (VADER), language
identification from character n-grams, document embeddings (doc2vec) and word mover's distance, IBM alignment models,
coreference. Priority: a POS tagger and language identification are the two that pair with existing parts; both S.

### audio and vision

Covered by librosa, torchaudio, kornia and scikit-image; see `torch-ecosystem.md` and the first-pass `other.md`.

### omai-render

The chart and view layer was measured only against the chart vocabulary of ArviZ (first pass), yellowbrick and scikit-plot
(`tabular-and-interpretable.md`) and the by-hand list in `repertoire.md`. Matplotlib, seaborn, Plotly, Altair and Bokeh were not
enumerated. The honest statement is that omai-render has about 160 chart builders and 270 view builders and the comparison for it
is weaker than for the engine.

## What a name-level comparison cannot see

The survey compares the names of public functions and classes. These kinds of capability are invisible to it:
- **Behaviour and numbers.** Two functions with the same name can differ in defaults, in numerical accuracy and in what they do
  at edge cases. The engine's oracle fixtures (Python references, golden tests) are the instrument for that, not this survey.
- **Composition.** A library can offer a capability as a configuration of an existing function (omai's `linearModel` covers
  a dozen scikit-learn classes). A name match misses it, and the verdict files carry the concept matches by hand.
- **Depth of options.** `fit` with eight solvers and `fit` with two are one name. "Partial" is a judgement, not a measurement.
- **Documentation, examples and teaching quality.** A library that has an algorithm and no worked example does not teach it.
  omai's `@example` checks and the site pages are where that is judged.
- **Interactivity and traces.** Whether a method can be stepped, probed and replayed is omai's point and no other library
  offers it, so the survey has nothing to compare it with. The tape contracts and probes are not counted.
- **Scale, speed and robustness.** Anything that depends on size (GPU, sparse data, large graphs) or failure behaviour.
- **Ecosystem integration.** File formats, serving, data loaders, hubs of pretrained models and plugin systems.
- **What no library lists.** Textbook material that sits in no single package (the repertoire of `repertoire.md`) and the ideas
  newer than the lists used here. A name-level pass finds gaps against what exists, never against what should exist.
- **Memory lists.** About two thirds of the second-pass libraries were listed from memory. They are checklists, not inventories:
  a function missing from the list is invisible, and a name that exists may have changed in the current release.

For these reasons the survey will never be complete, and a gap list that looks short for a domain means "the lens showed
little", not "nothing is missing".
