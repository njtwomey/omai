# Complex systems (a methods chapter the encyclopedia has no home for; its pages are curiosities and foundations)

Emergence simulated: cellular automata (elementary rules, Life, Langton's ant, lattice gases), agent-based models (boids, Schelling segregation, an epidemic in a room, traffic, predator and prey on a grid, Sugarscape), criticality (sandpiles, forest fires, percolation), networks growing and carrying processes (random graphs, small worlds, preferential attachment, epidemics and cascades on networks, diffusion and walks, synchronisation), nonlinear dynamics and chaos (phase portraits, iterated maps and bifurcations, Lyapunov exponents, continuation, chaos from a series) and fractals displayed and zoomed in their forms (escape-time sets, iterated function systems, L-systems, strange attractors), as sessions over `dynamics`, `graph`, `stochastic`, `worlds` and `foundations`. The ODE systems page exists (`methods/dynamics/odeSystems.ts`); the automata, the agent-based models beyond boids and the epidemic, and the fractals are the library gap.

## Structure

Everything here is a process simulated and watched, so it lives under `simulate/` (section 3 of `methods/README.md`): `automata/` for the lattice and agent models, `networks/` for the graph ones, `dynamics/` for the continuous and iterated systems and, within it, `fractals/` as a group of four. The `mathematics` README leaves "the behaviour of named dynamical systems (phase portraits, bifurcations, chaos)" to a dynamics chapter that does not exist as a README; it is taken here as `simulate/dynamics/nonlinear/`, beside the fractals that are its pictures, and the solvers' own behaviour stays in `mathematics`.

```
simulate/automata/
  cellular/          elementary rules, Life and life-like rules, Langton's ant, lattice gases and lattice Boltzmann, the cave smoother
  agent-based/       boids, Schelling, people in a room (SIR), Nagel-Schreckenberg traffic, predator and prey on a grid, Sugarscape
  criticality/       the Abelian sandpile, the forest-fire model, site and bond percolation
simulate/networks/
  growth/            random graphs edge by edge, small worlds by rewiring, preferential attachment node by node
  spreading/         epidemics on a network against the mean field, threshold and independent cascades, influence maximisation, diffusion and walks
  synchronisation/   Kuramoto oscillators, consensus
simulate/dynamics/
  nonlinear/         phase portraits, iterated maps and bifurcation diagrams, chaos in flows, continuation, chaos from data
  fractals/
    escape-time/     Mandelbrot, Julia, Burning Ship, multibrot (Newton fractals are `mathematics`'s)
    ifs/             Sierpinski, the Barnsley fern, random iterated function systems by the chaos game
    l-systems/       Koch, dragon, Hilbert, plants, generation by generation
    attractors/      Lorenz, Rössler, Thomas; Hénon, de Jong, Clifford, Ikeda; drawn, coloured by density, zoomed
  shared/            the world slot, the population record, the viewport knob, the point cloud, the named systems (below)
```

The built `methods/dynamics/odeSystems.ts` is the trajectory of `nonlinear/phase-portrait` and the flow of `fractals/attractors`; its named-systems slot is shared with `mathematics` (`simulate/numerics/ode/`) and `curiosities`.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of an automaton is one generation, of an agent model one tick, of a growing network one node or edge, of a map one iterate, of a fractal one iteration over the whole grid or one generation or one batch of points; the viewport is a knob, so zooming is a change of configuration that re-renders the picture probe. Status is about the method alone; what the library has is in the needs column.

### automata/cellular/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `elementary-automaton` | The rule number (0 to 255) as the knob, one row per step from a single cell or a random row: Wolfram's four classes side by side (rule 90's Sierpinski triangle, rule 110's gliders, rule 30's centre column as a random bit stream tested for randomness), the density and the entropy of each row on the tape, the rule's icon as a probe. | `automaton/row` t!, `automaton/density` s, `automaton/entropy` s, `automaton/board` p (the rows so far), `rule/icon` p | `information/discrete.ts`: `entropy`; TODO `lib/dynamics/automata.ts`: `elementaryAutomaton` (an `Algorithm`, one row per step), `wolframClass` | |
| `life` | Life and the life-like rules (the B/S string as the knob), the start as a slot (a random soup at a density, the glider, the R-pentomino, the Gosper gun, a soup that dies): the population and the births and deaths per generation, still lifes and oscillators detected with their periods (events), the board at the viewed step; the cave smoother of `worlds` as the same mechanism with a different rule. | `life/board` t!, `life/population` s, `life/births` s, `life/deaths` s, `life/period` s, `life/stabilised` e | `worlds/levels.ts`: `caveAlgorithm` (a life-like smoothing rule as a traced algorithm); TODO `lib/dynamics/automata.ts`: `lifeLike`, `LIFE_PATTERNS`, `detectPeriod` | |
| `langtons-ant` | The ant on a blank grid: chaos for ten thousand steps, then the highway (the event the page waits for), the rule string (LR, LLRR, LRRRRRLLR) as the knob giving other patterns, several ants interacting; the board with the ant's trail. | `ant/board` t!, `ant/position` t, `ant/heading` s, `ant/blackCells` s, `ant/highway` e | TODO `lib/dynamics/automata.ts`: `langtonsAnt`, `highwayDetected` | |
| `lattice-gas` | HPP and FHP particles colliding on a lattice (the microscopic rules as the mode), the macroscopic density and velocity fields emerging by coarse-graining (the block size as the knob), sound waves crossing; lattice Boltzmann (D2Q9) as the third mode with flow past a cylinder shedding vortices at a Reynolds number (the knob); shared with `curiosities` (the Kelvin-Helmholtz instability). | `lattice/occupancy` t!, `field/density` t!, `field/velocity` t, `field/vorticity` p, `flow/reynolds` s | `dynamics/pde/operators.ts`: `curlOf` (the vorticity); TODO `lib/dynamics/lattice.ts`: `latticeGas` (HPP, FHP), `latticeBoltzmann` (D2Q9 with bounce-back) | |

### automata/agent-based/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `boids` | Reynolds' three rules with their weights and the perception radius as knobs on a torus: the flock forming from random headings, the polarisation (the order parameter) on the tape, the transition from disorder to order as noise falls (a sweep over the noise), obstacles and a predator as modes. | `flock/positions` t!, `flock/headings` t!, `flock/polarisation` s, `flock/clusters` s, `flock/frame` p (an `agentSteps` record), `study/orderCurve` proj | `dynamics/flocking.ts`: `boidsAlgorithm`, `polarisation`; `exposition/agents.ts`: `agentSteps`; `graph/components.ts`: `connectedComponents` (the clusters); ready | |
| `schelling` | Two kinds of agent on a grid with vacancies, each moving when fewer than a share of its neighbours (the tolerance, the knob) are its kind, one round per step: segregation emerging from a mild preference (the segregation index rising, the unhappy count falling to zero), the vacancy rate and the neighbourhood size as the other knobs, the board at the viewed round. | `city/board` t!, `city/unhappy` s, `city/moved` s, `city/segregation` s, `city/settled` e | TODO `lib/dynamics/agent-based.ts`: `schelling` (an `Algorithm` over a `worlds` grid), `segregationIndex` | |
| `epidemic-in-a-room` | People moving in a room, some staying put (the distancing share as the knob), infection on contact at a rate, recovery after a time: the S, I and R counts per tick against the SEIR curves of the mean field with the same rates, the peak lowered and delayed by distancing (the 2020 picture), R0 as a probe; the people as an `agentSteps` record coloured by state. | `room/positions` t!, `room/states` t!, `room/counts` t! (S, E, I, R), `room/peak` s, `room/frame` p, `truth/meanField` p (the SEIR curves) | `dynamics/epidemics.ts`: `epidemicAlgorithm`, `seirCounts`, `seirCurves`, `SEIR_STATES`; `exposition/agents.ts`: `agentSteps`; ready | |
| `traffic` | Nagel-Schreckenberg on a ring road: the density as the knob, cars accelerating, braking and dawdling (the probability as the other knob), the jam forming from nothing and moving backwards (the space-time diagram as a board), the fundamental diagram (flow against density) as a sweep with the capacity drop. | `road/cells` t!, `road/flow` s, `road/meanSpeed` s, `road/jammed` s, `road/spaceTime` p, `study/fundamentalDiagram` proj | TODO `lib/dynamics/agent-based.ts`: `nagelSchreckenberg` | |
| `predator-prey-grid` | Wa-Tor: fish and sharks on a torus with breeding and starvation times as knobs, the two populations oscillating out of phase (the phase plane as a probe) against the Lotka-Volterra cycle of the ODE with fitted rates, extinction as the event when the grid is small. | `world/board` t!, `world/fish` s, `world/sharks` s, `world/phasePlane` p, `truth/lotkaVolterra` p, `world/extinct` e | `dynamics/named-systems.ts`: `lotkaVolterraField` (the comparison, through `methods/dynamics/odeSystems.ts`); TODO `lib/dynamics/agent-based.ts`: `waTor` | |
| `sugarscape` | Agents with vision and metabolism harvesting sugar on a landscape with two peaks: migration to the peaks, the wealth distribution's Gini coefficient rising (the inequality emerging), the carrying capacity; reproduction and trade as modes. | `scape/positions` t!, `scape/wealth` t!, `scape/gini` s, `scape/population` s, `scape/frame` p | TODO `stats/descriptive.ts`: `gini`; TODO `lib/dynamics/agent-based.ts`: `sugarscape` | |

### automata/criticality/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `sandpile` | The Abelian sandpile: one grain dropped per step (at the centre or at random, the mode), topplings cascading, the avalanche's size and duration as the tape's scalars, their distribution a power law (the histogram on log axes as a projection), the board of heights with the current avalanche lit; the sandpile identity as a probe for the curious. | `pile/board` t!, `pile/grains` s, `pile/avalancheSize` s, `pile/avalancheDuration` s, `study/sizeDistribution` proj | TODO `lib/dynamics/automata.ts`: `sandpile` | |
| `forest-fire` | Drossel-Schwabl: trees growing at rate p, lightning at rate f (both knobs), fires burning clusters in one step: the fire sizes' power law when f/p is small, the forest's density oscillating, the clusters as a board. | `forest/board` t!, `forest/density` s, `forest/fireSize` s, `study/fireDistribution` proj | TODO `lib/dynamics/automata.ts`: `forestFire` | |
| `percolation` | Site or bond percolation (the mode) on a square lattice: sites added one per step in a random order (so p is the step), clusters merged by union-find, the largest cluster's share jumping at p_c (0.5927 for sites, 0.5 for bonds), a spanning cluster as the event, the spanning probability sharpening with the lattice size (a sweep over sizes: finite-size scaling), the clusters coloured on the board. | `lattice/p` s, `lattice/largestCluster` s, `lattice/clusters` s, `lattice/spans` e, `lattice/board` t!, `study/spanningCurve` proj | `graph/structures.ts`: `UnionFind`; `graph/generators.ts`: `gridGraph`; TODO `graph/percolation.ts`: `sitePercolationAlgorithm`, `bondPercolationAlgorithm`, `spanningCluster` | |

### networks/growth/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `random-graphs` | The model as the mode: Erdős-Rényi one edge per step (the giant component appearing when the mean degree passes 1, connectivity near ln n / n, the degree distribution against the Poisson), Watts-Strogatz's rewiring probability swept (clustering staying high while the path length collapses: the small-world window), preferential attachment one node per step (the degree distribution's power law forming on log axes against uniform attachment's exponential, the first nodes becoming hubs), the stochastic block model; the graph drawn with the current largest component. | `graph/nodes` s, `graph/edges` s, `graph/giantShare` s, `graph/clustering` s, `graph/pathLength` s, `graph/degrees` t!, `graph/structure` p, `study/degreeDistribution` proj, `study/smallWorld` proj | `graph/generators.ts`: `erdosRenyiGraph`, `gnmRandomGraph`, `wattsStrogatzGraph`, `barabasiAlbertGraph`, `stochasticBlockModel`; `graph/measures.ts`: `averageClustering`, `diameter`, `density`; `graph/components.ts`: `connectedComponents`; `stochastic/random-measures.ts`: `polyaUrnAlgorithm` (the rich-get-richer mechanism alone); the stepped growth forms TODO (`graph/growth.ts`: `edgeByEdgeAlgorithm`, `preferentialAttachmentAlgorithm`, `rewiringAlgorithm`) | |

### networks/spreading/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `network-epidemic` | SIR and SIS (the mode) on a contact network from the slot (random, small world, scale free, the karate club) by event-driven simulation: the infected share against the mean-field curve with the same rates (the network's heterogeneity changing the peak), the epidemic threshold against 1/λ_max of the adjacency, hubs as super-spreaders, vaccinating by degree against at random (the knob) and the herd-immunity threshold; the nodes coloured by state on the structure. | `epidemic/t` s, `epidemic/counts` t!, `epidemic/states` t!, `epidemic/peak` s, `epidemic/threshold` s, `graph/structure` p, `truth/meanField` p | `graph/generators.ts`; `datasets/graphs.ts`: `karateClub`; `graph/measures.ts`: `spectralGap`; `dynamics/epidemics.ts`: `seirCurves` (the mean field); TODO `graph/spreading.ts`: `networkEpidemicAlgorithm` (Gillespie over edges) | |
| `cascades` | Threshold cascades (Watts: each node adopts when a share of its neighbours has, the threshold as the knob; the cascade window in mean degree where a single seed can take the network), the independent cascade one wave per step, and influence maximisation by greedy seeding (one seed per step, the marginal gain diminishing: submodularity); the cascade as node states on the structure. | `cascade/wave` s, `cascade/adopted` s, `cascade/states` t!, `seeding/seed` m, `seeding/gain` s, `graph/structure` p, `study/cascadeWindow` proj | `graph/traversal.ts`: `bfsAlgorithm` (the waves); TODO `graph/spreading.ts`: `linearThresholdAlgorithm`, `independentCascadeAlgorithm`, `greedyInfluence` | |
| `diffusion-and-walks` | Heat diffusing on a graph (the kernel at time t as node colours, the knob), a random walk's occupation converging to the degree-proportional law (the total variation on the tape), the mixing time against the spectral gap across the network slot; PageRank as the walk with teleport is `mathematics`'s (`process/graphs/centrality`). | `walk/position` s, `walk/occupation` t!, `walk/totalVariation` s, `diffusion/t` s, `diffusion/values` t!, `graph/structure` p | `graph/spectral.ts`: `heatDiffusion`, `heatKernel`; `stochastic/random-walks.ts`: `graphRandomWalk`, `graphWalkStationary`; `stochastic/markov-chains.ts`: `mixingTime`, `spectralGap`; ready | |

### networks/synchronisation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `kuramoto` | Oscillators with natural frequencies drawn from a spread (the knob) coupled with strength K (the knob): the order parameter r near zero then jumping at the critical coupling (a sweep over K), the phases on the circle at the viewed step, the locked cluster and the drifters; on a network from the slot against all-to-all; consensus (the Laplacian flow) as the linear cousin with the agreement rate set by the algebraic connectivity. | `oscillators/phases` t!, `oscillators/order` s, `oscillators/locked` s, `oscillators/circle` p, `study/orderCurve` proj | `graph/spectral.ts`: `heatDiffusion` (consensus); `graph/measures.ts`: `algebraicConnectivity`; `dynamics/ode/solve-ivp.ts`: `solveIvpAlgorithm`; TODO `dynamics/named-systems.ts`: `kuramotoField` (all-to-all and on a graph) | |

### dynamics/nonlinear/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `phase-portrait` | A named system's field with its nullclines, its equilibria found and classified (saddle, node, spiral, centre), the stable and unstable manifolds of the saddles, streamlines, and a trajectory integrated from a start that is a knob; the parameter as a knob (the predator-prey cycles around the centre, the damped pendulum's spirals into the wells, Van der Pol's limit cycle attracting from inside and out). | `trajectory/t` s, `trajectory/state` t!, `trajectory/h` s, `field/portrait` p (the field, nullclines and streamlines), `field/equilibria` p, `field/manifolds` p | `dynamics/fields.ts`: `fixedPoints`, `classifyFixedPoint`, `nullclines`, `streamline`, `invariantManifold`, `fieldJacobian`; `dynamics/named-systems.ts`: `namedSystems`, `lotkaVolterraField`, `dampedPendulumField`, `vanDerPolField`; `dynamics/ode/solve-ivp.ts`: `stepperFor`; ready | built `methods/dynamics/odeSystems.ts` (the trajectory; TODO the portrait probes) |
| `iterated-maps` | The logistic map's cobweb at r (the knob) one iterate per step, the orbit settling, doubling or wandering; the bifurcation diagram as r sweeps (one r per step, the attractor's points added), the period-doubling cascade and Feigenbaum's ratio read off the splits, the Lyapunov exponent crossing zero where chaos begins, the periodic windows; the tent, Hénon, standard and circle maps as the slot (Arnold tongues for the circle map as a sweep over two knobs). | `orbit/x` t!, `orbit/n` s, `map/r` s, `map/lyapunov` s, `map/cobweb` p, `study/bifurcation` proj, `study/lyapunovCurve` proj | `dynamics/maps.ts`: `logisticMap`, `tentMap`, `henonMap`, `standardMap`, `circleMap`, `orbitAlgorithm`, `orbit`, `cobweb`, `bifurcationDiagram`, `lyapunovExponents`; ready | |
| `chaos-in-flows` | Sensitivity to initial conditions: two Lorenz trajectories from starts a millionth apart diverging (their distance on a log axis, the slope the largest Lyapunov exponent), the whole Lyapunov spectrum by Benettin's method one renormalisation per step, a Poincaré section accumulating one crossing per step (Rössler's one-dimensional return map), a limit cycle found by shooting with its Floquet multipliers (Van der Pol's), the period-doubling route of Rössler as a parameter sweeps. | `pair/distance` s, `spectrum/exponents` t!, `section/points` t!, `section/returnMap` p, `cycle/orbit` t!, `cycle/multipliers` t, `trajectory/state` t! | `dynamics/nonlinear.ts`: `lyapunovSpectrum`, `poincareSection`, `limitCycle`, `floquetMultipliers`; `dynamics/named-systems.ts`: `lorenzField`, `vanDerPolField`; `dynamics/ode/solve-ivp.ts`: `solveIvpAlgorithm`; TODO `dynamics/named-systems.ts`: `rosslerField` | |
| `continuation-and-bifurcation` | Equilibria followed through a parameter by pseudo-arclength continuation one predictor-corrector step per step, folds and Hopf points detected as events, the branch's stability marked by its eigenvalues (the branch diagram building), the limit cycle born at the Hopf point drawn in the phase plane; the solver's own view is `mathematics` (`roots/continuation`). | `branch/parameter` s, `branch/point` t!, `branch/eigenvalues` t, `branch/stable` s, `branch/fold` e, `branch/hopf` e, `branch/diagram` p | `dynamics/nonlinear.ts`: `continuationAlgorithm`, `continueEquilibria`; `dynamics/fields.ts`: `classifyFixedPoint`; ready | |
| `chaos-from-data` | A measured series (the Lorenz x coordinate, or a real one) reconstructed: the delay embedding with its delay and dimension as knobs (the attractor reappearing), the correlation dimension's scaling region, the largest Lyapunov exponent from divergence of neighbours, recurrence plots and their quantification; a periodic and a noisy series as the controls. | `embedding/points` t!, `dimension/scaling` t!, `dimension/estimate` s, `lyapunov/divergence` t!, `lyapunov/estimate` s, `recurrence/plot` p, `recurrence/measures` p | `dynamics/nonlinear.ts`: `delayEmbedding`, `correlationDimension`, `largestLyapunovExponent`, `recurrenceQuantification`; ready | |

### dynamics/fractals/escape-time/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `escape-time` | The set as the mode (Mandelbrot, Julia with c as a knob dragged on the Mandelbrot set drawn beside it, the Burning Ship, multibrot with its power): one iteration over the whole grid per step, every pixel's escape count updated incrementally so the picture sharpens as the cap grows (the share escaped on the tape), smooth colouring, the viewport (centre, scale, resolution) as a knob so zooming is a change of configuration and the player replays the iterations at the new place; the period of the bulbs as a probe at a clicked point. | `image/escape` t! (the grid's counts), `image/iteration` s, `image/escapedShare` s, `image/picture` p (rendered at the viewport), `point/orbit` p (the orbit of a clicked c or z) | `foundations/complex.ts`: `complexGrid`, `ComplexBox`, `domainColouring` (the grid and a colouring); TODO `lib/dynamics/fractals/escape-time.ts`: `escapeTimeAlgorithm` (incremental iteration over a grid with bailout and smooth counts), `MANDELBROT`, `JULIA`, `BURNING_SHIP`, `multibrot` | |
| `newton-basins` | see mathematics (`simulate/numerics/roots/newton-basins`: the plane coloured by the root reached) | | | |

### dynamics/fractals/ifs/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `iterated-function-system` | The chaos game one batch of points per step, the cloud accumulating (the Sierpinski triangle from three maps, the Barnsley fern from four with their probabilities as knobs: edit a map and the fern bends, random IFS from the slot), the deterministic mode mapping the whole set per step (the collage theorem, the set converging whatever the start), the fractal dimension by box counting against the similarity dimension; each map drawn as what it does to a grid. | `points/cloud` t! (accumulating), `points/count` s, `ifs/maps` t, `ifs/dimension` s, `ifs/mapPictures` p, `cloud/picture` p (at the viewport) | `foundations/linear-algebra.ts`: `gridTransformation` (an affine map on a grid); TODO `lib/dynamics/fractals/ifs.ts`: `chaosGameAlgorithm`, `ifsDeterministicAlgorithm`, `SIERPINSKI`, `BARNSLEY_FERN`, `boxCountingDimension` | |

### dynamics/fractals/l-systems/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `l-system` | The axiom rewritten one generation per step (the string's length growing geometrically on the tape), the turtle drawing it with the angle as the knob (Koch at 60°, the dragon at 90°, Hilbert and Peano filling the square, the Sierpinski arrowhead, plants with brackets), stochastic rules as the mode giving a different plant each run from the same grammar; the curve's length and its box-counting dimension. | `generation/n` s, `generation/length` s, `generation/string` t (the first characters), `generation/curve` t!, `curve/picture` p, `curve/dimension` s | TODO `lib/dynamics/fractals/l-systems.ts`: `lSystemAlgorithm` (the rewriting), `turtle` (the interpretation), `L_SYSTEMS` (the named grammars) | |

### dynamics/fractals/attractors/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `strange-attractor` | The system as the slot (flows: Lorenz, Rössler, Thomas, integrated; maps: Hénon, de Jong, Clifford, Ikeda, iterated) with its parameters as knobs, the point cloud accumulating per step and coloured by density, rotated in three dimensions for the flows, the viewport as a knob for the maps; the butterfly's two lobes and the switching between them counted. | `points/cloud` t! (accumulating), `points/count` s, `trajectory/state` t!, `trajectory/lobeSwitch` e, `cloud/picture` p (at the viewport), `cloud/scene` p (a 3-D scene) | `dynamics/named-systems.ts`: `lorenzField`; `dynamics/maps.ts`: `henonMap`, `orbitAlgorithm`; `scene/three-d.ts`: `scene3d`; TODO `dynamics/named-systems.ts`: `rosslerField`, `thomasField`; TODO `dynamics/maps.ts`: `deJongMap`, `cliffordMap`, `ikedaMap` | built `methods/dynamics/odeSystems.ts` (Lorenz by RK45; TODO the maps, the density picture and the viewport) |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `worlds.ts` | The world slot: lattices with or without wrap and their sizes, the named Life patterns and soups, the contact networks (`graph/generators.ts` models with their knobs, `datasets/graphs.ts`: `karateClub`), each with its truth where one exists (p_c, the mean-field curves, λ_max and the epidemic threshold, the Poisson or power-law degree law) | `automata/*`, `networks/*` |
| `population.ts` | The population-statistics record: counts per state per step, order parameters (polarisation, r, the segregation index, the giant share), the mean-field or exact comparison line, as the `series` of an `agentSteps` record and as curves | `agent-based/`, `criticality/`, `spreading/`, `synchronisation/` |
| `viewport.ts` | The viewport knob (centre, scale, resolution) and the re-rendering probe shared by the fractals and the attractors: a picture at the viewport from the state, so a drag or a wheel sets the knob and the player replays at the new place | `fractals/*` |
| `pointCloud.ts` | The accumulating point-cloud record in two and three dimensions (the chaos game, the attractors, Poincaré sections) with density colouring and a trail | `fractals/ifs/`, `fractals/attractors/`, `nonlinear/` |
| `systems.ts` | The named-systems slot over `dynamics/named-systems.ts` and `dynamics/maps.ts` with their parameters as knobs and their known equilibria, cycles and exponents, shared with `mathematics` (`simulate/numerics/ode/shared/systems.ts`) and `curiosities` | `nonlinear/`, `fractals/attractors/` |

## Examples to make

- Elementary rule 110 unfolding row by row, the rule number as the knob, the four classes side by side.
- Boids with the weights as knobs, the polarisation on the tape, the order-against-noise curve as a sweep.
- Schelling segregation at a tolerance of a third: the segregation index climbing from a mild preference.
- SIR on a small-world network against its mean-field curve, the hubs vaccinated first.
- Percolation on a grid: the giant component appearing as p crosses the threshold, the spanning probability sharpening with size.
- The logistic map's bifurcation diagram building as r sweeps, the Lyapunov exponent beneath it.
- The Mandelbrot set zoomed into a chosen point, the escape-time image sharpening as the iteration cap grows; a Julia set with c dragged on the Mandelbrot set beside it.
- The Barnsley fern appearing point by point with a map's coefficients as knobs; a dragon curve generation by generation.

## Rendering

- Boards and scenes (`boardSteps`, `SceneView`), agents frame by frame (`agentSteps`), structure views with node state, curves against a mean-field truth, trajectories and 3-D scenes (`Scene3DView`), vector fields (`fields`).
- Missing: a scene panel that follows the cursor (agents at the viewed step); a zoomable image panel (drag and wheel set the viewport knob, the probe re-renders at that viewport) for the escape-time sets; a point-cloud panel that accumulates, in 2-D and 3-D, for the iterated function systems and the attractors; a phase-portrait panel (field, nullclines, equilibria classified, a trajectory) over the `fields` panel; a bifurcation-diagram panel built from a sweep.

## Problems

- The library gap is the chapter's own: the automata (`lib/dynamics/automata.ts`), the agent-based models beyond boids and the epidemic (`lib/dynamics/agent-based.ts`), the lattice gases (`lib/dynamics/lattice.ts`), the network growth and spreading loops (`graph/growth.ts`, `graph/spreading.ts`, `graph/percolation.ts`) and all four fractal forms (`lib/dynamics/fractals/`). Each is a traced `Algorithm` over pieces that exist (`worlds` grids, `graph` structures and generators, `foundations/complex.ts` grids), so none is large; the dynamics of `dynamics/` (fields, maps, nonlinear, named systems) is complete.
- The chapter takes nonlinear dynamics and chaos because no `dynamics` chapter README exists while `mathematics` defers to one; if such a chapter is cut later, `simulate/dynamics/nonlinear/` moves whole.
- Keeps its name. The encyclopedia's part 9 (curiosities) or part 1 (mathematics) would link here; the Ising model is `probabilistic-inference`'s (`graphical/markov-random-fields/`), evolutionary game dynamics are `games`', random walks and Markov chains as processes are `probability`'s.
