# Search and planning (a methods chapter the encyclopedia spreads over optimisation, games and reinforcement learning)

Classical AI search watched: state-space search (breadth-first, depth-first, iterative deepening, uniform-cost, bidirectional, greedy, A* and its relatives with heuristics checked), local search (hill climbing, simulated annealing, tabu, local beam, late acceptance), evolutionary and swarm search (genetic algorithms, evolution strategies, genetic programming, particle swarms, ant colonies, quality diversity, neuroevolution), constraint satisfaction (arc consistency, backtracking with its heuristics, min-conflicts), adversarial search (minimax, alpha-beta, expectimax, Monte Carlo tree search) and planning (STRIPS with delete-relaxation heuristics, GraphPlan, partial-order and hierarchical planning, PDDL), as sessions over `search`, `problems`, `worlds`, `games` and `environments`. Two methods exist (`methods/search/gridSearch.ts`, `methods/search/annealing.ts`); the rest of the library's searches are traced algorithms waiting for their rows.

## Structure

Everything here is a procedure over a problem, so it lives under `search/` (section 3 of `methods/README.md`) with its five sections, the problem as the slot and the strategy as the mode: `state-space/` for searches that build paths, `local/` for those that move between complete states, `population/` for those that breed them, `constraint/` for assignments, `adversarial/` for game trees, `planning/` for actions with preconditions and effects. The equilibrium solvers over games are `games` (`search/games/`); the shortest paths on an explicit graph are `mathematics` (`process/graphs/`); the global optimisers of `optim` (CMA-ES, differential evolution, dual annealing, basin hopping, Nelder-Mead) are `optimisation`'s and get a row here; the satisfiability solvers are `logic-and-rule-learning`'s (`search/logic/`).

```
search/
  state-space/
    uninformed/        breadth-first, depth-first, depth-limited, iterative deepening, uniform-cost, bidirectional
    informed/          greedy, A*, weighted A*, IDA*, recursive best-first, beam; the heuristic slot and its admissibility report
    problems/          the problem slot watched for itself: puzzles and route finding, maze carving before the search
  local/               hill climbing with restarts, annealing with its schedules, tabu, local beam, late acceptance, on landscapes and combinatorial problems
  population/
    genetic/           genetic algorithms over bit strings, permutations and real vectors with the operators as slots
    strategies/        self-adaptive evolution strategies (CMA-ES and differential evolution are `optimisation`'s)
    programming/       genetic programming for symbolic regression
    swarm/             ant colony optimisation on the travelling salesman, particle swarms
    quality-diversity/ MAP-Elites, novelty search
    neuroevolution/    NEAT-style evolution of small networks
  constraint/          AC-3, backtracking with ordering and inference, min-conflicts
  adversarial/         minimax, alpha-beta, negamax, expectimax, Monte Carlo tree search on small games
  planning/            STRIPS forward and regression with h-add, h-max and FF, GraphPlan, partial-order planning, HTN, PDDL
  shared/              the problem, heuristic and landscape slots, the boards, the population base, the game adapter (below)
```

The built `methods/search/gridSearch.ts` moves to `state-space/` (it already runs the uninformed and the informed strategies as one mode slot) and `methods/search/annealing.ts` to `local/`; `optimisation` shows annealing as a global optimiser beside the `optim` ones, so the method is shared and the page differs.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a state-space search is one expansion, of a local search one move, of a population one generation, of a constraint solver one assignment or one arc, of a game search one node, of a planner one expansion or one refinement. Status is about the method alone; what the library has is in the needs column.

### state-space/uninformed/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `graph-search` | The strategy as the mode (breadth-first, depth-first, depth-limited with its limit, iterative deepening, uniform-cost, bidirectional): the frontier as a queue, a stack or a priority queue, the explored set flooding the map one expansion per step, the path cost and the nodes expanded as the comparison (breadth-first ignoring terrain, uniform-cost paying for it, depth-first's long wrong path, bidirectional's two small floods meeting); the effective branching factor. Two strategies on one map under one cursor is the page. | `board` t! (cell categories: wall, explored, frontier, the node expanded, the path), `frontier` s, `explored` s, `pathCost` s, `search/node` m (the node expanded, its depth and cost), `search/solved` e, `search/branchingFactor` p | `search/uninformed.ts`: `breadthFirstSearchAlgorithm`, `depthFirstSearchAlgorithm`, `depthLimitedSearchAlgorithm`, `iterativeDeepeningSearchAlgorithm`, `uniformCostSearchAlgorithm`, `bidirectionalSearchAlgorithm`; `search/problem.ts`: `effectiveBranchingFactor`, `solutionPath`, `solutionActions`; `problems/grid.ts`: `gridProblem`; `problems/maps.ts`: `SEARCH_MAPS`, `searchMap`; all modes ready | built `methods/search/gridSearch.ts` (modes: breadth-first, depth-first, uniform-cost; TODO depth-limited, iterative deepening, bidirectional) |

### state-space/informed/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `heuristic-search` | The strategy as the mode (greedy best-first, A*, weighted A* with its weight, IDA*, recursive best-first, beam with its width) and the heuristic as a slot (zero, Manhattan, Euclidean, Chebyshev, octile, an inflated one, a deliberately inadmissible one): the explored set shrinking as the heuristic sharpens, the path cost rising past the optimum when it is inadmissible or the weight grows (the trade is the picture), f-contours over the map, IDA*'s thresholds as events and its re-expansions, beam search losing the goal; the admissibility and consistency report against Dijkstra's true costs as a probe. Compare with uniform-cost on the same map (A* with h = 0). | as `graph-search`, plus `search/f` s, `search/threshold` s (IDA*), `search/reexpansions` s, `heuristic/report` p (`isAdmissible`, `isConsistent`, the worst violation), `heuristic/field` p (h over the map) | `search/informed.ts`: `bestFirstSearchAlgorithm`, `greedyBestFirstSearch`, `aStarSearchAlgorithm`, `weightedAStarSearch`, `idaStarSearchAlgorithm`, `recursiveBestFirstSearchAlgorithm`, `beamSearchAlgorithm`, `isAdmissible`, `isConsistent`; `worlds/analysis.ts`: `heuristicTo`, `cellDistance`, `distanceField` (the true costs); all modes ready | built `methods/search/gridSearch.ts` (modes: greedy, a-star, weighted-a-star; TODO IDA*, recursive best-first, beam, the heuristic slot and its report) |
| `shortest-paths-on-a-graph` | see mathematics (`process/graphs/paths/`: Dijkstra, A*, Bellman-Ford and Floyd-Warshall on an explicit graph from `graph`) | | | |

### state-space/problems/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `puzzle-search` | Any strategy above on a problem that is not a grid: the 8-puzzle (misplaced tiles against Manhattan distance, the explored count falling by orders of magnitude), the Towers of Hanoi, missionaries and cannibals, the Romania road map with straight-line distances (AIMA's route finding); the state drawn as tiles, towers or a map at the viewed step, the search tree as a probe. | `search/node` m, `search/explored` s, `search/frontier` s, `search/pathCost` s, `state/picture` p, `search/tree` p (a `tree` record of the nodes generated) | `search/problem.ts`: `SearchProblem`, `graphProblem`, `searchTree`; the strategies of `search/uninformed.ts` and `search/informed.ts`; TODO `problems/puzzles.ts`: `eightPuzzle`, `hanoi`, `missionariesAndCannibals`, `romania` (the map with straight-line distances as the heuristic) | |
| `maze-generation` | The carver as the mode (recursive backtracker, Prim, Kruskal, Wilson, Aldous-Broder, Eller, binary tree, sidewinder, recursive division), one carve per step on the board: the texture each leaves (long corridors, many short dead ends), braiding adding loops (the knob), the dead-end count and the solution's length; then any search on the maze made. | `maze/board` t!, `maze/carved` s, `maze/deadEnds` s, `maze/solutionLength` s, `maze/perfect` s | `worlds/generators.ts`: `mazeAlgorithm`, `generateMaze`, `braid`, `isPerfectMaze`; `worlds/analysis.ts`: `deadEnds`, `shortestPath`, `distanceField`; `problems/maps.ts`: `mazeWorld`; all modes ready | |

### local/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `local-search` | The strategy as the mode (hill climbing with sideways moves and random restarts, simulated annealing with its schedule as a knob (exponential, geometric, logarithmic) and starting temperature, tabu search with its tenure, local beam search with k states, late acceptance) on the landscape slot (the plane test functions with their contours) or a combinatorial problem (n-queens by conflicts, the travelling salesman by 2-opt over permutations, OneMax over bit strings): the walk over the contours, the energy and the best so far, uphill moves accepted as events, restarts as events, the tabu list, the beam's k walkers; annealing finding the global minimum where hill climbing sits in a local one. | `position` t!, `energy` s, `best` s, `temperature` s, `objective/surface` p, `move/accepted` e (uphill), `search/restart` e, `tabu/list` t, `beam/positions` t! | `search/local.ts`: `hillClimbingAlgorithm`, `randomRestartHillClimbingAlgorithm`, `simulatedAnnealingAlgorithm`, `exponentialSchedule`, `geometricSchedule`, `logarithmicSchedule`, `tabuSearchAlgorithm`, `localBeamSearchAlgorithm`, `lateAcceptanceAlgorithm`, `spaceProblem`, `permutationNeighbours`, `bitFlipNeighbours`; `datasets/objectives.ts`: `rastrigin`, `himmelblau`, `ackley`, `rosenbrock`, `beale`; `problems/puzzles.ts`: `nQueens`; a travelling-salesman instance TODO (`problems/tours.ts`: `randomTour`, `tourLength`, the classic small instances); all strategies ready | built `methods/search/annealing.ts` (modes: simulated-annealing, hill-climbing, late-acceptance on the plane; TODO tabu, local beam, restarts, the combinatorial problems) |
| `global-optimisers` | see optimisation (`optim/minimize/cmaes.ts`: `cmaesAlgorithm`; `optim/minimize/differential-evolution.ts`: `differentialEvolutionAlgorithm`; `optim/minimize/dual-annealing.ts`: `dualAnnealingAlgorithm`; `optim/minimize/global.ts`: `basinHoppingAlgorithm`; `optim/minimize/derivative-free.ts`: `nelderMeadAlgorithm`, on the same landscapes) | | | |

### population/genetic/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `genetic-algorithm` | The representation as the slot (bit strings for OneMax and the knapsack, permutations for the travelling salesman, real vectors for the plane) with selection, crossover and mutation as option slots (tournament, roulette, stochastic universal sampling, rank, truncation; one- and two-point, uniform, order, PMX, cycle, blend, simulated binary, arithmetic; bit flip, swap, inversion, scramble, Gaussian, polynomial), the population size, the rates and elitism as knobs: the population over the contours one generation per step, the best and mean fitness, the diversity collapsing (premature convergence under strong selection pressure, the knob to turn), the best individual drawn (a tour, a knapsack). | `generation/best` s, `generation/mean` s, `generation/diversity` s, `population/fitness` t!, `population/positions` t! (real vectors) or `population/genomes` t!, `best/individual` p, `objective/surface` p, `study/population` proj (a `populationSteps` record) | `search/evolutionary/genetic.ts`: `geneticAlgorithmAlgorithm`, `bitStrings`, `permutations`, `realVectors`; `search/evolutionary/operators.ts`: `tournamentSelection`, `rouletteSelection`, `stochasticUniversalSampling`, `rankSelection`, `truncationSelection`, `onePointCrossover`, `twoPointCrossover`, `uniformCrossover`, `orderCrossover`, `pmxCrossover`, `cycleCrossover`, `blendCrossover`, `simulatedBinaryCrossover`, `arithmeticCrossover`, `bitFlipMutation`, `swapMutation`, `inversionMutation`, `scrambleMutation`, `gaussianMutation`, `polynomialMutation`; `exposition/populations.ts`: `populationSteps`; all ready | |

### population/strategies/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `evolution-strategy` | The self-adaptive (μ/2, λ) strategy: each individual carrying its own step size, the step sizes shrinking as the population closes on the optimum (the log of the mean σ beside the log of the best value), μ and λ as knobs, on the plane and on an ill-conditioned bowl where the comparison with CMA-ES (its covariance adapting to the valley) is the lesson. | `generation/best` s, `generation/meanStepSize` s, `population/positions` t!, `population/stepSizes` t!, `objective/surface` p, `study/population` proj | `search/evolutionary/genetic.ts`: `evolutionStrategyAlgorithm`, `Strategist`; the comparison from `optim/minimize/cmaes.ts`: `cmaesAlgorithm`; `datasets/objectives.ts`: `quadraticBowl` | |

### population/programming/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `genetic-programming` | Symbolic regression: the best expression tree per generation drawn with its curve over the data, subtree crossover and mutation as events on the tree, bloat (the mean tree size growing) against the parsimony penalty as the knob, ramped half-and-half initialisation's depth mix; the winner simplified to a formula against the generating one. | `generation/best` s, `generation/meanSize` s, `generation/meanDepth` s, `population/fitness` t!, `best/tree` p (a `tree` record), `best/curve` p, `best/formula` p (TeX) | `search/evolutionary/programming.ts`: `geneticProgrammingAlgorithm`, `arithmeticPrimitives`, `rampedHalfAndHalf`, `regressionFitness`, `formatTree`, `treeSize`, `treeDepth`, `evaluateTree`; `symbolic`: `simplify`, `toLatex`; all ready | |

### population/swarm/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `ant-colony` | Ant System against MAX-MIN on a travelling-salesman instance: the pheromone matrix as a heatmap over the edges (the strong edges emerging as the tour), evaporation ρ and the α/β balance between pheromone and distance as knobs, the best tour drawn and its length against the known optimum, stagnation when every ant walks the same tour. | `iteration/bestLength` s, `iteration/meanLength` s, `pheromone/matrix` t!, `best/tour` p, `iteration/stagnated` e | `search/swarm.ts`: `antColonyAlgorithm`, `antColonyOptimization`; the instance from `shared/landscapes.ts` (the TSP generator TODO under `problems/tours.ts`); both modes ready | |
| `particle-swarm` | The swarm's positions and velocities over the contours, inertia and the cognitive and social weights as knobs (too much inertia: the swarm never settles; too much social pull: it collapses on a local best), the global best; binary PSO over bit strings as the discrete mode. | `iteration/best` s, `swarm/positions` t!, `swarm/velocities` t, `swarm/personalBests` t, `objective/surface` p | `search/swarm.ts`: `discreteParticleSwarmAlgorithm` (binary, ready); the continuous swarm TODO (`search/swarm.ts`: `particleSwarmAlgorithm`) | |

### population/quality-diversity/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `quality-diversity` | MAP-Elites' archive filling one generation per step (the grid over behaviour descriptors as a board coloured by fitness, coverage and the QD-score rising) against novelty search's archive (each individual's novelty, the archive growing where behaviour is new) as modes, on a deceptive maze where the objective misleads and diversity finds the way. | `generation/coverage` s, `generation/qdScore` s, `generation/best` s, `archive/board` t! (MAP-Elites), `archive/size` s (novelty), `population/novelty` t | `search/evolutionary/quality-diversity.ts`: `mapElitesAlgorithm`, `noveltySearchAlgorithm`; both modes ready | |

### population/neuroevolution/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `neuroevolution` | NEAT-style evolution of small networks on XOR and cart-pole: genomes as networks drawn, speciation keeping innovations alive (the species' sizes as a stacked picture), complexification (nodes and connections added as events), the champion's fitness; against a fixed-topology evolution strategy over the weights. | `generation/best` s, `generation/species` t!, `generation/complexity` s, `champion/network` p (a `diagram` record), `champion/episode` p | `environments/classic-control.ts`: `cartPole`; `rl/rollout.ts`: `evaluatePolicy`; TODO `search/evolutionary/neat.ts`: `neatAlgorithm` (innovation numbers, speciation, the crossover of aligned genes) | |

### constraint/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `arc-consistency` | AC-3 one arc per step on Sudoku or map colouring: the domains as a grid of small sets shrinking, the queue of arcs, the revisions that prune and those that do not, a wipe-out as the event; an easy Sudoku solved by propagation alone, a hard one left for search. | `queue/size` s, `domains/sizes` t!, `domains/board` p, `arc/revised` m, `domains/wipeout` e | `search/csp.ts`: `ac3Algorithm`, `csp`, `allDifferent`, `binaryConstraint`; `problems/puzzles.ts`: `sudoku`, `SUDOKUS`, `mapColouring`, `AUSTRALIA`, `randomMap`; ready | |
| `backtracking` | Variable ordering (as given, minimum remaining values, degree) and value ordering (as given, least constraining) and inference (none, forward checking, maintained arc consistency) as knobs: the partial assignment as a board one assignment per step, backtracks as events, the nodes expanded as the comparison across the knob settings (the orders of magnitude MRV and forward checking save on n-queens and Sudoku). | `search/assignments` s, `search/backtracks` s, `assignment/board` t!, `domains/sizes` t, `search/backtrack` e, `search/solved` e | `search/csp.ts`: `backtrackingAlgorithm`, `backtrackingSearch`, `VariableOrder`, `cspProblem`; `problems/puzzles.ts`: `nQueens`, `sudoku`; ready | |
| `min-conflicts` | Local search over complete assignments: n-queens with hundreds of queens, the conflict count falling one move per step, the plateau where it stalls and the random-walk probability (knob) that breaks it; the board at the viewed step. | `search/conflicts` s, `search/move` m, `assignment/board` t!, `search/solved` e | `search/csp.ts`: `minConflictsAlgorithm`, `minConflicts`, `violations`; ready | |
| `satisfiability` | see logic-and-rule-learning (`search/logic/sat/`: DPLL, CDCL and WalkSAT on a CSP encoded as CNF) | | | |

### adversarial/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `game-tree-search` | The algorithm as the mode (minimax, alpha-beta with move ordering as the knob, negamax, expectimax for chance nodes) on tic-tac-toe, connect four (a depth limit with an evaluation function), Nim and a literal tree: the tree one node per step with values backed up and the subtrees cut (marked), the nodes visited as the comparison (alpha-beta's savings with good ordering), the principal variation, the board at the root. | `search/nodes` s, `search/value` s, `search/depth` s, `search/pv` t, `search/cut` e, `search/tree` p (a `tree` record with cuts), `game/board` p | `search/adversarial.ts`: `minimaxAlgorithm`, `alphaBetaAlgorithm`, `negamaxAlgorithm`, `expectimaxAlgorithm`; `environments/games.ts`: `ticTacToe`, `connectFour`, `nim` through the adapter in `shared/games.ts` (their `GameDefinition` is not `search/adversarial.ts`' `Game`); `games/extensive-form/tree.ts`: `treeGame` for literal trees; all modes ready | |
| `mcts` | UCT on tic-tac-toe and connect four: the tree's visit counts and values at the viewed step, the exploration constant and the simulations per move as knobs, random rollouts against a value function (PUCT with priors as the mode), the chosen move's visit share; against alpha-beta's exact value at the root. The AlphaZero training loop is `games`. | `search/simulations` s, `search/rootValue` s, `search/visits` t!, `search/tree` p, `game/board` p, `truth/value` s | `search/adversarial.ts`: `mctsAlgorithm`, `mcts`; `rl/planning.ts`: `gameMcts`; `agents/planning.ts`: `gameMctsAgent`, `alphaBeta` (the transposition-table version for the truth); both modes ready | |

### planning/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `strips-planning` | Forward (progression) against regression search as modes, the heuristic as a slot (blind, h-add, h-max, FF's relaxed plan) on the blocks world (the Sussman anomaly, random problems of n blocks) or a PDDL domain: the state as towers at the viewed step, the nodes expanded per heuristic as the comparison, the relaxed plan of the current state as a probe, the plan found and then executed block by block, validated. | `search/nodes` s, `search/h` s, `search/frontier` s, `plan/length` s, `plan/actions` t!, `state/towers` p (a board), `heuristic/relaxedPlan` p, `plan/valid` s | `search/planning/strips.ts`: `strips`, `forwardProblem`, `regressionProblem`, `forwardPlan`, `regressionPlan`, `applyAction`, `validatePlan`; `search/planning/heuristics.ts`: `hAdd`, `hMax`, `hFF`, `relaxedPlan`; `search/planning/pddl.ts`: `parseDomain`, `parseProblem`, `ground`; `problems/planning-domains.ts`: `blocksWorld`, `SUSSMAN`, `randomBlocks`, `towersOf`; all modes ready | |
| `graphplan` | The planning graph growing one level per step (fact and action layers, mutex pairs drawn), levelling off, the backward extraction with memoised failures (its attempts as events), the parallel plan's length against the sequential one's; the invariant mutexes read off the graph. | `graph/level` s, `graph/facts` s, `graph/actions` s, `graph/mutexes` s, `graph/levelledOff` e, `extraction/attempt` m, `plan/layers` t!, `graph/picture` p | `search/planning/graphplan.ts`: `graphplanAlgorithm`, `graphplan`, `mutexPairs`; ready | |
| `partial-order-planning` | Least commitment: the partial plan's steps, ordering constraints and causal links as a structure, one refinement per step (an open precondition closed by a new or an existing step, a threat resolved by promotion or demotion), the plan's linearisations at the end; against the total-order forward planner on the same problem. | `plan/steps` s, `plan/openPreconditions` s, `plan/threats` s, `plan/refinement` m, `plan/structure` p (a `diagram` record), `plan/linearisation` p | `search/planning/partial-order.ts`: `partialOrderProblem`, `partialOrderPlan`, `linearize`; the best-first search over plans from `search/informed.ts`; ready | |
| `htn-planning` | SHOP-style decomposition: the task network as a tree growing one decomposition per step (compound tasks opened by methods into subtasks until primitive actions remain), backtracking over method choices as events, the plan read off the leaves. | `plan/tasks` s, `plan/primitive` s, `plan/decomposition` m, `plan/backtrack` e, `plan/tree` p (a `tree` record) | `search/planning/htn.ts`: `htnPlanAlgorithm`, `htnPlan`, `htnTree`; ready | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `problems.ts` | The problem slot over `lib/problems` and `lib/worlds`: the hand-drawn maps (`SEARCH_MAPS`) and generated mazes as `gridProblem`s, the puzzles as CSPs, the planning domains, the textbook and procedural worlds, each with its truth (the optimal path cost from `worlds/analysis.ts`: `distanceField`, the solution count of n-queens, the known optimum of a tour); the puzzles and tours to add | `state-space/*`, `local/`, `constraint/`, `planning/` |
| `heuristics.ts` | The heuristic slot: zero, Manhattan, Euclidean, Chebyshev, octile, inflated and deliberately inadmissible heuristics from `worlds/analysis.ts`: `heuristicTo`, `cellDistance`; the planning heuristics `hAdd`, `hMax`, `hFF`; the admissibility and consistency report as a probe (`search/informed.ts`: `isAdmissible`, `isConsistent`) | `state-space/informed/`, `planning/` |
| `landscapes.ts` | The landscape slot: the plane test functions of `datasets/objectives.ts` with their surface on a grid (as `annealing.ts` builds) and their known minima, and the combinatorial objectives (OneMax, the knapsack, tours, n-queens by conflicts) with their representations and neighbourhoods | `local/`, `population/*` |
| `boards.ts` | Boards as records: the cell categories of a grid search (`BOARD_CELLS` as `gridSearch.ts` has), a CSP's assignment and domains, a planning state as towers, a game tree on a layout, and the `boardSteps` projection over any of them | `state-space/*`, `constraint/`, `adversarial/`, `planning/` |
| `population.ts` | The population base: the representation slot, the operator tables (selection, crossover, mutation as option tables of `methods/shared/options/table.ts`), fitness statistics and diversity per generation, and the `populationSteps` projection | `population/*` |
| `games.ts` | The adapter from `environments/games.ts`: `GameDefinition` (turn-based environments with legal actions and a `render`) to `search/adversarial.ts`: `Game` and `games/extensive-form/tree.ts`: `ExtensiveGame`, so one tic-tac-toe serves the searches, MCTS, CFR and the agents; the game slot shared with `games` | `adversarial/`, `games` |

## Examples to make

- A* against Dijkstra and greedy on the same maze: the explored cells as boards side by side, the heuristic as the knob (exists for A*).
- Simulated annealing on a landscape, the temperature and the acceptance of uphill moves (exists); beside it hill climbing with restarts on the same landscape.
- A genetic algorithm on the plane: the population as points moving, the best fitness curve, the diversity collapsing when selection is too strong.
- Ant colony on a thirty-city tour: the pheromone heatmap sharpening into the tour.
- MCTS on tic-tac-toe: the tree's visit counts at the viewed step against alpha-beta's exact value.
- Arc consistency on Sudoku: domains shrinking cell by cell, then backtracking with MRV finishing a hard one.
- Forward STRIPS planning on the Sussman anomaly with h-add against FF: the towers at each expansion, the plan executed.

## Rendering

- Boards (`boardSteps`), population steps (`populationSteps`), trees with values and counts (`tree`), partial plans as diagrams (`diagram`), expression trees.
- Missing: a game-tree panel with cuts marked and the principal variation highlighted; a domains-grid panel (a grid of small sets); a pheromone heatmap over a tour; a planning-graph panel (layers with mutex arcs).

## Problems

- `methods/search/` (gridSearch, annealing) moves here; the old `evolutionary/` and the search half of `gofai/` named in `docs/structure.md` 5.6 are described by the rows above and never had library code of their own beyond `lib/search`.
- Two game contracts: `search/adversarial.ts` has its own `Game` and `environments/games.ts` its `GameDefinition` (and `agents/planning.ts` plays the latter); the adapter in `shared/games.ts` is the first thing to build, and `games` shares it.
- The problem instances are the library gap, not the algorithms: the 8-puzzle, Hanoi, missionaries and cannibals, the Romania map, travelling-salesman instances, random k-SAT (`logic-and-rule-learning`'s) all go in `lib/problems`.
- The encyclopedia has no such chapter; its derivative-free optimisation (`optimisation`), games (`games`) and planning sections link here.
