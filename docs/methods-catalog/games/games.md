# Games (part 7, application domains)

Game theory computed and played: normal-form games and their equilibria (dominance, zero-sum by linear programming, support and vertex enumeration, Lemke-Howson, correlated equilibria), learning in games (fictitious play, best-response dynamics, regret matching, multiplicative weights, replicator dynamics), extensive-form games (backward induction, counterfactual regret minimisation and its sampled forms, the sequence form, exploitability), self-play (AlphaZero on small boards, with `reinforcement-learning`), cooperative games (the Shapley value, the core, the nucleolus) and mechanisms (auctions, voting), as sessions over `games`, `environments` (the two-player games), `agents` and `optim`. The game-tree searches themselves are `search-and-planning`'s.

## Structure

An equilibrium is found by a procedure over a problem (the game), so the chapter lives under `search/` (section 3 of `methods/README.md`) in a `games/` section of its own beside `adversarial/`, which keeps the tree searches: `equilibria/` for the normal-form solvers, `learning/` for the dynamics that converge (or cycle) to them, `extensive-form/` for the solvers over trees with information sets, `self-play/` for the trained players, `cooperative/` and `mechanisms/` for the rest of the field. The learning dynamics could as well be `simulate/dynamics/` (replicator dynamics is a vector field); they stay here so that one game slot and one simplex record serve every page, which the Problems record.

```
search/games/
  equilibria/        dominance, zero-sum LP, support and vertex enumeration, Lemke-Howson, correlated equilibria
  learning/          fictitious play and best-response dynamics, regret matching and multiplicative weights, replicator dynamics
  extensive-form/    backward induction, CFR and its variants, MCCFR, the sequence form; exploitability as the measure
  self-play/         AlphaZero on small boards; neural fictitious self-play and minimax-Q are `reinforcement-learning`'s
  cooperative/       the Shapley value and its sampling, Banzhaf, the core, the least core, the nucleolus
  mechanisms/        auctions (first and second price, all-pay, VCG), voting rules and their paradoxes
  shared/            the game slot, the simplex record, the profile probes, payoff matrices as pictures (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a learning dynamic is one round of play, of Lemke-Howson one pivot, of CFR one iteration, of backward induction one node, of the Shapley sampler one permutation, of a mechanism one auction or one elimination round. Status is about the method alone; what the library has is in the needs column.

### equilibria/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `dominance` | Iterated elimination of dominated strategies one elimination per step (strict against weak as the mode: the order dependence of weak elimination, two orders reaching two games), the game shrinking as a payoff picture, what survives against the Nash equilibria; the security levels of pure strategies. | `elimination/action` m, `game/actions` t, `game/payoffs` p, `game/survivors` p | `games/normal-form/dominance.ts`: `iteratedElimination`, `isDominated`, `dominatedActions`, `restrictGame`, `pureMaxmin`; `games/normal-form/classic.ts`: the named games; ready | |
| `zero-sum` | Von Neumann's minimax theorem as a linear programme: the row and column players' optimal mixes and the value, the saddle point marked on the payoff heatmap (or its absence, hence mixing), the maximin security strategy of a player in a general game; rock-paper-scissors with payoffs as knobs tilting the mix. | `solution/rowStrategy` t!, `solution/columnStrategy` t!, `solution/value` s, `game/payoffs` p, `game/bestResponses` p | `games/normal-form/zero-sum.ts`: `solveZeroSum`, `zeroSumEquilibrium`, `maximin`; `games/normal-form/game.ts`: `bestResponses`, `nashConv`, `isNash`; ready (one LP; a probe page) | |
| `nash-enumeration` | Every equilibrium of a bimatrix game: support enumeration one support pair per step (the candidate solved and checked, the ones that are equilibria kept) against vertex enumeration walking the best-response polytopes as the mode; a degenerate game (the knob) where the count is wrong; the equilibria as points in the simplex. | `search/support` m, `search/candidates` s, `search/found` s, `result/equilibria` t!, `result/simplex` p | `games/normal-form/enumeration.ts`: `supportEnumeration`, `vertexEnumeration`, `distinctEquilibria`; `games/normal-form/game.ts`: `pureNashEquilibria` (whole functions; TODO per-support step records) | |
| `lemke-howson` | Complementary pivoting one pivot per step from an initial label (the knob: different labels reach different equilibria, and the enumeration from every label finds several), the tableau and the labels carried, the path drawn on the best-response polytopes of a 2-by-2 or 3-by-3 game, the equilibrium reached and checked by NashConv. | `pivot/label` m, `pivot/tableau` t!, `pivot/dropped` s, `path/vertex` t!, `result/equilibrium` p, `result/nashConv` s | `games/normal-form/lemke-howson.ts`: `lemkeHowsonAlgorithm`, `lemkeHowson`, `lemkeHowsonEnumeration`; ready | |
| `correlated-equilibrium` | A distribution over joint actions from which no player wants to deviate when told its part: found by linear programming with the objective as the knob (welfare, a player's payoff, the uniform), the deviation gains as the check, against the product distribution of a Nash equilibrium on chicken (the traffic light that beats mixing); coarse correlated equilibria beside it. | `solution/distribution` t!, `solution/welfare` s, `solution/deviationGains` t, `game/payoffs` p | `games/normal-form/correlated.ts`: `correlatedEquilibrium`, `correlatedGains`, `isCorrelatedEquilibrium`, `jointDistribution`; ready | |

### learning/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `fictitious-play` | Each player best-responding to the others' empirical frequencies one round per step: the beliefs and the empirical strategies spiralling into the mixed equilibrium of rock-paper-scissors or matching pennies, cycling without converging on Shapley's game, the NashConv of the empirical profile on a log axis; stochastic fictitious play with perturbed utilities (the smoothing as the knob) and best-response dynamics (converging to a pure equilibrium on coordination games or cycling) as modes. | `round/actions` m, `round/beliefs` t!, `round/strategies` t!, `round/nashConv` s, `strategy/simplex` p (the trajectory in the simplex or the square) | `games/learning.ts`: `fictitiousPlayAlgorithm`, `stochasticFictitiousPlayAlgorithm`, `bestResponseDynamicsAlgorithm`; `games/normal-form/classic.ts`: `rockPaperScissors`, `matchingPennies`, `shapleyGame`, `coordinationGame`, `stagHunt`; `games/normal-form/game.ts`: `nashConv`; all modes ready | |
| `no-regret` | Regret matching and multiplicative weights (Hedge, with its step size as the knob) in self-play: each player's cumulative regret growing sublinearly (the bound drawn), the average play converging to a coarse correlated equilibrium (and to the Nash equilibrium in a zero-sum game) while the current play does not; the regrets per action as bars. | `round/regrets` t!, `round/strategy` t!, `round/averageStrategy` t!, `round/regretBound` s, `round/nashConv` s, `strategy/simplex` p | `games/no-regret.ts`: `regretMatchingAlgorithm`, `regretMatching`, `multiplicativeWeightsAlgorithm`, `multiplicativeWeights`; both modes ready | |
| `replicator-dynamics` | The population's shares flowing in the simplex (symmetric games) or the unit square (bimatrix games, two populations), the vector field as a probe, the payoff matrix as knobs (the hawk-dove mix, the stag hunt's basins, the rock-paper-scissors cycle), evolutionarily stable strategies as the resting points; the projected replicator dynamics with exploration as the mode that converges where the plain one cycles. | `flow/t` s, `flow/shares` t!, `flow/fitness` t, `field/simplex` p (the field with the trajectory), `result/ess` p | `games/evolutionary.ts`: `replicatorField`, `replicatorDynamics`, `asymmetricReplicatorDynamics`, `projectedReplicatorDynamicsAlgorithm`; `dynamics/ode/solve-ivp.ts`: `solveIvpAlgorithm` (the stepped flow); `games/normal-form/classic.ts`: `hawkDove`; ready | |

### extensive-form/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `backward-induction` | A perfect-information tree solved from the leaves one node per step, the values backed up and the chosen branch marked; subgame perfection against a Nash equilibrium that relies on an incredible threat (the ultimatum and the centipede games), the same game converted to normal form to see both equilibria. | `induction/node` m, `induction/value` s, `tree/values` t!, `tree/picture` p (a `tree` record), `game/normalForm` p | `games/extensive-form/tree.ts`: `backwardInduction`, `gameTree`, `treeGame`, `toNormalForm`, `expectedReturns` (whole; TODO the node-per-step form) | |
| `cfr` | Counterfactual regret minimisation on Kuhn (and Leduc) poker with the variant as the mode (CFR, CFR+, linear CFR, discounted CFR; alternating updates as the knob): the regrets and the current and average strategies at each information set, exploitability on a log axis against the variants under one cursor, the strategy at the viewed iteration against Kuhn's known equilibrium family (the bluffing frequency). | `iteration/exploitability` s, `iteration/nashConv` s, `iteration/regrets` t!, `iteration/averagePolicy` t!, `infoset/strategy` p (one information set's strategy over iterations), `game/tree` p, `truth/equilibrium` p | `games/extensive-form/cfr.ts`: `cfrAlgorithm`, `cfr`, `cfrPlus`; `games/extensive-form/best-response.ts`: `exploitability`, `extensiveNashConv`, `bestResponse`; `games/extensive-form/poker.ts`: `kuhnPoker`, `kuhnEquilibrium`, `leducPoker`; `games/extensive-form/tree.ts`: `gameTree`; all modes ready | |
| `mccfr` | External against outcome sampling as the mode: the parts of the tree touched each iteration lit on the tree, exploitability against full CFR at equal nodes touched (the variance against the cost), on Leduc where full CFR is slow. | `iteration/exploitability` s, `iteration/nodesTouched` s, `iteration/sampled` t (the information sets visited), `game/tree` p | `games/extensive-form/mccfr.ts`: `mccfrAlgorithm`, `mccfr`; both modes ready | |
| `sequence-form` | Realisation plans over action sequences: the sequence form built (its size linear in the tree against the normal form's exponential, the step adds a betting round) and the zero-sum game solved by one linear programme; the plan read back as a behaviour strategy against CFR's average. | `form/sequences` s, `form/normalFormSize` s, `solution/realisationPlan` t!, `solution/value` s, `solution/behaviour` p | `games/extensive-form/sequence-form.ts`: `sequenceForm`, `solveSequenceForm`; ready | |

### self-play/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `alphazero` | Self-play games generating policy and value targets, the network trained on them, PUCT search guided by its priors; the policy and value losses, the win rate against alpha-beta and a random player as the evaluation (one match series per step), the search tree at a chosen position, the board; tic-tac-toe, connect four and Go on a five-by-five board, each at a cost the Problems record. | `training/policyLoss` s, `training/valueLoss` s, `selfPlay/games` s, `eval/winRate` t (against each opponent), `search/tree` p, `game/board` p | `agents/planning.ts`: `alphaZeroAlgorithm`, `alphaZero`, `policyValueNetwork`, `gameMctsAgent`, `minimaxAgent`, `randomLegalPolicy`; `environments/games.ts`: `ticTacToe`, `connectFour`; `environments/go.ts`: `go`; `rl/rollout.ts`: `compareAgents`; ready (the mechanics; the budget is the problem) | |
| `neural-fictitious-self-play` | see reinforcement-learning (`act/deep/multi-agent/competitive-learning`: NFSP on Kuhn poker, minimax-Q on Markov games, with this chapter's exploitability as the measure) | | | |

### cooperative/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `shapley-value` | The average marginal contribution over every order of arrival: the sampled estimate one permutation per step converging on the exact value (the error on a log axis), on weighted voting (a player with many votes and no power), the glove game, bankruptcy and unanimity games; the Banzhaf index beside it and the interaction index of a pair; convexity checked. Shapley values as feature attributions are `trustworthy-machine-learning`'s. | `sample/permutation` m, `sample/estimate` t!, `sample/error` s, `truth/shapley` p, `truth/banzhaf` p, `game/interactions` p | `games/cooperative.ts`: `shapleyValue`, `sampledShapley`, `banzhafIndex`, `shapleyInteractionIndex`, `weightedVotingGame`, `gloveGame`, `bankruptcyGame`, `unanimityGame`, `isConvex`, `isSuperadditive`; ready | |
| `the-core` | Allocations no coalition can improve on: membership of a proposed allocation (the knob: drag it in the simplex of a three-player game), every coalition's excess, the least core's ε, the nucleolus minimising the largest excesses lexicographically one level per step; an empty core (the glove game) and a large one. | `allocation/values` t!, `allocation/excesses` t!, `allocation/inCore` s, `leastCore/epsilon` s, `nucleolus/level` s, `core/simplex` p | `games/core.ts`: `isInCore`, `excesses`, `leastCore`, `corePoint`, `nucleolus`; ready | |

### mechanisms/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `auctions` | The rule as the mode (first price, second price, all-pay; the reserve as the knob) with bidders' values drawn each auction (one per step): the winning bid, who wins and what they pay, the revenue's running mean against the expected revenue (revenue equivalence across the rules), the symmetric equilibrium bid function drawn against truthful bidding; VCG on combinatorial bundles as the fourth mode. | `auction/values` t, `auction/bids` t, `auction/winner` s, `auction/payment` s, `revenue/mean` s, `revenue/expected` s, `bids/function` p | `games/auctions.ts`: `firstPriceAuction`, `secondPriceAuction`, `allPayAuction`, `equilibriumBid`, `expectedRevenue`, `vcg`, `combinatorialAuction`; ready | |
| `voting` | One profile of ranked ballots under every rule as modes (plurality, Borda, Condorcet, Copeland, the Smith set, instant runoff one elimination round per step), the winners disagreeing, the paradoxes found (a majority cycle, a Condorcet winner losing, the spoiler); the Tennessee capital and the Condorcet paradox profiles, and a random profile (the knob). | `round/eliminated` m, `round/tallies` t!, `rule/winner` t, `profile/pairwise` p, `profile/paradoxes` p | `games/voting.ts`: `profile`, `plurality`, `borda`, `pairwiseMatrix`, `condorcetWinner`, `copeland`, `smithSet`, `instantRunoff`, `socialChoiceParadoxes`, `tennesseeProfile`, `condorcetParadoxProfile`; ready | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `games.ts` | The game slot: the textbook normal-form games of `games/normal-form/classic.ts` with payoffs as knobs and random games, the extensive-form pokers and literal trees, the cooperative examples, and the two-player environments of `environments/games.ts` through the adapter of `search-and-planning`'s `shared/games.ts`; each with its known equilibria as the truth | every group |
| `simplex.ts` | The simplex record: strategies as points, a trajectory and a vector field over the simplex (three actions) or the unit square (two players with two actions), with the equilibria marked | `learning/`, `equilibria/`, `cooperative/the-core` |
| `profiles.ts` | Probes over any strategy profile: NashConv and exploitability (`games/normal-form/game.ts`: `nashConv`; `games/extensive-form/best-response.ts`: `exploitability`), best responses, deviation gains; shared with `reinforcement-learning`'s multi-agent rows | every group |
| `payoffs.ts` | Payoff matrices as pictures: the bimatrix as a heatmap with best responses marked, the tree with values, an information set's strategy as bars | `equilibria/`, `extensive-form/`, `mechanisms/` |

## Examples to make

- Fictitious play on rock-paper-scissors: the strategies spiralling into the mixed equilibrium, NashConv on a log axis; Shapley's game cycling beside it.
- Replicator dynamics in the simplex with the payoff matrix as knobs, the field as a probe.
- CFR against CFR+ on Kuhn poker: exploitability on a log axis, the bluffing frequency at the viewed iteration against the equilibrium family.
- Lemke-Howson's path on a bimatrix game, one pivot per step, from each initial label.
- The sampled Shapley value converging on the exact one for a weighted voting game.
- Instant runoff eliminating candidates round by round on the Tennessee profile, the Condorcet winner losing.

## Rendering

- Simplex plots with trajectories and fields, matrices as heatmaps, trees with values and information sets (`tree`), strategy bars per information set, exploitability curves, boards for the played games.
- Missing: a simplex panel kind (a triangle with a trajectory and a field, the square for two populations); an information-set strategy panel that follows the cursor; the polytope path of Lemke-Howson.

## Problems

- The library has the algorithms; the chapter is thin wrappers plus the simplex record and the profile probes. The only stepped forms missing are small: support enumeration's supports and backward induction's nodes.
- Where the learning dynamics live (`search/games/learning/` here, or `simulate/dynamics/` as vector fields) is a lookup either way; decide with the first built page.
- AlphaZero at any board size is a cost problem (the worker producer of `methods/README.md`, section 6, problem 7); tic-tac-toe is the only size that runs live today.
- The game slot is shared with `search-and-planning` (the tree searches) and `reinforcement-learning` (self-play, NFSP, minimax-Q); the adapter between the `Game`, `ExtensiveGame` and `GameDefinition` contracts is built once, in `search-and-planning`'s `shared/`.
