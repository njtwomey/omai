# The render audit: what the methods ask of `mdml-render`, and the order to build it

A working document, 2026-10-28, composed from the Rendering sections of the 52 chapter READMEs. Its companion on the
engine side is `LIB.md`. The render layer's job for a method page is fixed by the session layer: a layout of panels
under one step cursor, each panel a view of a record or a path of the tape, the player driving the producer. So
almost everything the chapters ask for is one of four things: a **panel kind** (a view that follows the cursor), an
**overlay** on a panel that exists, a **record kind** the engine must emit for a view to draw (the engine's shadow of
this document), or **infrastructure** (controls, workers, export). The audit groups the asks that way, counts the
chapters behind each, and orders them.

## 1. What exists, and what the audit found

Exists: the session layer (`useSession` and `useSessions`, the player with eager, seeks, reset and held changes, the
comparison of variants under one cursor, the sweep panel), the panel kinds `rows` (with overlays), `probe`,
`curves`, `series`, `trajectory`, `tensor`, `projection`, `board`, `tape`, `sweep`; a view for every exposition
record kind (about 55 kinds, from `decisionSurface` to `episodes` and `boardSteps`); controls from a space including
unions; the dataset and environment selectors; figures with export of one panel.

Found: 52 Rendering sections name about 190 wants. After merging the duplicates they are **48 distinct pieces**, of
which 9 are asked by five or more chapters, and only 11 are specific to one chapter. The asks are not exotic: the
great majority are "a view that exists, following the cursor" or "a mark on a panel that exists".

## 2. The catalogue

Each row: the piece, what it is, the chapters asking (a count, with examples), and what it needs from the engine.

### 2.1 Panel kinds that follow the cursor (views over records that exist)

| piece                      | what it is                                                                                                   | asked by                                                                                         | engine side                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| image grid                 | many small images at the viewed step, with labels; a batch, a pyramid, neighbours of a query, augmented rows  | 7: convolutional, generative, data-centric, representation, multimodal, computer vision, LLM      | `image` records in a list, or an `imageSteps` grid |
| graph that follows the cursor | a structure view whose node and edge attributes come from the tape at the viewed step (colour, size, position from an embedding), with a true-graph overlay | 6: graph networks, causal, exploration, probabilistic inference, search (game trees), recommendation (index graphs) | a `structure` record per step, or per-step attribute tensors keyed by node |
| position-aligned strip     | tokens (or positions) on x with per-position values beneath: probabilities, attention, gates, tags, the chosen path of a trellis | 6: sequence models, transformers, LLM, NLP, information theory (code and interval strips), recommendation (ranking strips) | `tokens` exists; a `strip` record with per-position values |
| table panel                | a small table at the viewed step: cells coloured by status (missing, imputed, outlier, leaked, fold), a profile, test results with the decision rule, a balance table, many metrics with direction arrows, a code table | 7: data preparation, exploration, statistics, evaluation, causal, information, model selection | a `table` record (rows, columns, cell marks)     |
| matrix that follows the cursor | a heatmap of a per-step matrix: a kernel or similarity matrix, a ratings matrix with predicted cells, an affinity matrix filling, a channel matrix, a confusion matrix | 6: supervised, representation, recommendation, multi-task, information, evaluation | `tensor` panel at `[n, m]` exists; add cell marks and row and column labels |
| scene that follows the cursor | a `Scene` per step: agents at the viewed step, a plant with its horizon drawn, the IFS point cloud accumulating | 4: complex systems, control, curiosities, reinforcement learning                                  | a `scene` record per step (exists as `frame` events) |
| vector field at step       | the field recomputed at the viewed step or level, with samples or a trajectory over it: a score field, a phase portrait with nullclines and classified equilibria, a flow over an image | 5: generative, complex systems, scientific, computer vision, mathematics | a `vectorField` record (grid, u, v, optional nullclines) — not in `exposition` today |
| value map over a grid world | the board with values as colour and the policy's arrows, a fixed-state Q panel, a return-distribution panel | 2: reinforcement learning, search                                                                | `valueGrids` exists; add arrows                   |
| dendrogram and reachability | the merge just made highlighted, the cut movable; the reachability plot; the U-matrix                        | 1: unsupervised (3 panels)                                                                       | `tree` with heights; a `reachability` record       |
| small multiples            | one chart per layer, head, client, task or expert, aligned under one cursor                                   | 5: feedforward, transformers, transfer, multi-task, graph networks                                 | none: a layout feature over existing panels       |
| inner session              | a small session inside a step of the outer one (meta-learning's inner loop), with its own cursor             | 1: meta-learning                                                                                 | nested runs on the tape (the trace supports child runs) |

### 2.2 Overlays on panels that exist

| piece                           | what it is                                                                                                 | asked by                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| per-row values on the rows panel | a value per training row as colour or size with a legend scale: scores, weights, data values, attributions, query order, residuals | 8: active learning, anomaly, data-centric, trustworthy, weak supervision, semi-supervised, supervised, online |
| row markers                      | labelled against unlabelled, flagged, chosen at step k, the witness in a bag, a pseudo-label's correctness  | 4: semi-supervised, weak supervision, active learning, anomaly                                 |
| regions on the rows panel        | a rule as a rectangle, a leaf's region, an anchor's box, the ε-ball of an attack                             | 3: logic and rules, supervised, trustworthy                                                   |
| per-point explanation            | the neighbourhood, the isolation path, the reconstruction, the attribution of a chosen row                 | 3: anomaly, trustworthy, representation                                                       |
| graph against truth              | true, found and wrong edges in three styles; blocked and open paths for a query                            | 3: exploration, causal, probabilistic inference                                               |
| messages on edges                | the message just passed, as a label or a thickness, on the structure view                                   | 1: probabilistic inference                                                                     |
| arrows on an embedding or image  | an analogy arrow, a flow field, correspondences between two images (a two-image panel)                     | 3: representation, computer vision, control                                                   |
| a draggable point on an image or a chart | drag a point, the probe updates: the Julia parameter on the Mandelbrot set, an epipolar point, a ranking item | 3: complex systems, computer vision, evaluation                                           |
| step markers on curves           | planted changes, alarms, looks of a sequential test, restarts                                               | 4: shift and drift, online experimentation, time series, optimisation                          |

### 2.3 Curve families from sweeps and studies

These are curves panels over a study's tape with a particular shape; one family of options on the `curves` and
`sweep` panels covers them.

| piece                                   | what it is                                                                                        | asked by                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| many runs overlaid with alpha           | one faint curve per run, the mean in bold, a share curve beneath                                   | 5: learning theory, online experimentation, statistics, optimisation, engineering |
| bands, fans and stacks                   | a curve with its band, a quantile fan (forecast, path bundle), stacked terms or components          | 6: time series, learning theory, probability, supervised, active learning, data-centric |
| estimates against a truth line           | several estimators converging over n on one axis with intervals, the truth drawn                   | 3: causal, statistics, probability                                        |
| log-log with reference slopes            | rates against n with the theoretical slope                                                         | 3: mathematics, probability, learning theory                               |
| interval strip                           | many intervals as horizontal bars against a truth line, the misses marked                          | 2: statistics, online experimentation                                      |
| frontiers and envelopes                  | a Pareto front filling, a cost curve with the trivial envelope, a recall-speed frontier             | 3: trustworthy, decision theory, recommendation                            |
| diagrams of a study                      | a bracket diagram for halving, a critical-difference diagram, a fold board with groups and embargo  | 1: model selection (3 panels)                                              |
| distributions at the viewed step         | a PIT histogram, a sorted p-value panel with step-up thresholds, a QQ and residual pair, a reliability pair before and after | 3: decision theory, statistics, evaluation                |

### 2.4 Specialised panels

| piece                   | what it is                                                                                      | asked by                                    | engine side                              |
| ----------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------- |
| zoomable image          | drag and wheel set the viewport knob; the probe re-renders at that viewport                     | 2: complex systems (fractals), curiosities  | a probe with viewport arguments (exists: probes take args) |
| 3-D point cloud that accumulates | attractors and IFS in three dimensions, orbited by dragging                              | 2: complex systems, curiosities             | `Scene3DView` exists; an accumulating points record |
| complex plane           | poles and zeros moving at the viewed step, a locus's branches, the unit circle                  | 2: control, signal processing               | a `complexPlane` record                  |
| simplex                 | a triangle (or square for two populations) with a trajectory and a field                        | 1: games                                    | a `simplex` record                       |
| loss-landscape slice    | a 1-D line or 2-D surface through the iterate, recomputed at the viewed step; a line-search panel with the Armijo and Wolfe lines | 2: optimisation, feedforward | a probe (exists) over `optim/landscape.ts` |
| stability region        | the region with the current hλ marked                                                           | 1: mathematics                              | a field record                           |
| equation against truth  | TeX with coefficients coloured by error; a function with its derivatives beside the plot        | 2: scientific, mathematics                  | `MathView` exists; a `coefficientTable` with truth |
| audio in sync           | playback of a clip with the cursor on its frame; a signal above its spectrogram with the frame marked | 1: signal processing                   | audio assets through the dataset registry |
| chat transcript         | a step list of an agent loop with tool calls                                                    | 1: LLM                                      | `messages` exist on the tape             |

### 2.5 Controls and infrastructure

| piece                                     | what it is                                                                                              | asked by                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| validity greying                          | an inadmissible option of a union drawn disabled, from the mode table's `admissible`                    | every chapter with modes (the design, 7.1)       |
| the model selector with a gear            | a list of registered models, a settings button opening each one's space                                 | the design, 7.3; supervised, unsupervised        |
| the comparison over a slot                | one variant per option of a slot (every loss, every solver) under one cursor, built from `useSessions`  | 6: loss-functions, supervised, optimisation, decision theory, shift, engineering |
| the worker producer                       | sessions computed off the main thread, the tape streamed to the page                                    | every neural chapter; the design, problem 7      |
| page export and narratives                | a whole page as PNG or GIF; one tape read by several pages                                              | the design, section 5                            |
| samples over training                     | text or images at log-spaced steps as one panel                                                         | 2: LLM, generative                               |

## 3. The engine's shadow: record kinds to add to `lib/exposition`

Most panels above draw records that exist. The ones that do not, in the order the panels need them:
`vectorField` (grid, components, nullclines), `strip` (positions with values and marks), `table` (rows, columns,
cell marks), `structureSteps` (a structure with per-step node and edge attributes, or a per-step attribute map keyed
by node), `pointsSteps` (an accumulating point set, 2-D and 3-D), `complexPlane`, `simplex`, `reachability`, and a
`scene` per step as a first-class record rather than a frame event. Each is a plain serialisable value with a `kind`
tag and a constructor, as the existing kinds are, and each gets its headless view and React view in render. The
`messages` record the probabilistic-inference README defines for message passing joins them.

## 4. The order

1. **The generic panel kinds** that serve the most chapters: the image grid, the graph that follows the cursor, the
   position-aligned strip, the table panel, the matrix with marks, the scene that follows the cursor, the vector
   field. Seven panel kinds, each a small file over an existing or new record, each with a gallery recipe on a toy
   record. With the records of section 3 they cover about two thirds of all asks.
2. **The rows overlays as one mechanism**: per-row values with a legend, markers, regions and a per-point
   explanation, read from a path of the tape or a probe. Eight chapters wait on this and it is one change to the
   rows panel.
3. **The curve families**: many-runs with alpha, bands and fans and stacks, truth lines, log-log slopes, the
   interval strip, as options of `curves` and `sweep`; then the study diagrams.
4. **Small multiples** as a layout feature, and the **comparison over a slot** as a helper over `useSessions`
   (one variant per option of a union), with validity greying in the controls and the model selector.
5. **The specialised panels** (zoomable image, 3-D point cloud, complex plane, simplex, landscape slice, equation
   against truth, audio) as their chapters are built.
6. **Infrastructure**: the worker producer (the one piece that changes how every page performs), page export,
   narratives.

The rule for all of it: a panel is a view of a record or a path plus the cursor, and the layout vocabulary of
`layout.ts` grows by one kind at a time; no chapter gets a bespoke component. A page in the encyclopedia then
composes the same kinds.
