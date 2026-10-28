# Logic and rule learning (part 5, learning paradigms)

Rules and logic, deduced and learned: propositional logic and satisfiability (truth tables, normal forms, DPLL, CDCL, WalkSAT, MaxSAT), logic programming (unification, a small Prolog by SLD resolution, Datalog, answer sets), first-order theorem proving by resolution, the classical knowledge-based systems (production rules, certainty factors, fuzzy and belief-function reasoning, argumentation, description logics), rule induction from data (sequential covering, CN2, RIPPER, rule lists and sets, RuleFit, version spaces), inductive logic programming (FOIL, Golem, Progol, Metagol, Popper), statistical relational learning (Markov logic, probabilistic logic programs) and neuro-symbolic learning, as sessions over `logic`, `search` (refinement search), `explain`, `pgm` and `datasets`. Explanation by rules (anchors, surrogate rules) is `trustworthy-machine-learning`'s and gets a row here.

## Structure

The chapter has two halves that the layout of `methods/README.md` (section 3) separates: deduction and satisfiability are procedures over a problem (a formula, a program, a query), so they live under `search/logic/`; induction is a task, so rule learning from attribute-value data lives under `fit/classify/rules/` (a rule set is a classifier) and learning from relations, facts and background knowledge under `fit/relational/`. The constraint solvers of `search/constraint/` (`search-and-planning`'s) are the same problems in another language; the two share the puzzle encodings.

```
search/logic/
  propositional/     truth tables, normal forms, Tseitin, Quine-McCluskey, entailment
  sat/               DPLL, CDCL, WalkSAT on random and encoded formulas; MaxSAT, model counting, unsatisfiable cores
  logic-programming/ unification, SLD resolution (a Prolog), Datalog, answer sets
  theorem-proving/   clausal form, the given-clause resolution prover, finite models
  knowledge/         production systems, certainty factors, fuzzy inference, belief functions, argumentation, description logic (the classical AI of `docs/structure.md` 5.6, all library TODO)
fit/classify/rules/
  covering/          sequential covering, CN2, PRISM, RIPPER
  lists-and-sets/    Bayesian rule lists and CORELS, OneR and decision tables, RuleFit
  version-spaces/    Find-S and candidate elimination
fit/relational/
  ilp/               FOIL, Golem's RLGG, Progol and Aleph, Metagol, Popper
  statistical/       Markov logic networks, probabilistic logic programs
  neuro-symbolic/    differentiable logic over a network's outputs
analyse/explain/rules/  see trustworthy-machine-learning
shared/              knowledge bases, formulas, the rule record, proofs as trees (below)
```

The alternative is one `search/logic/` root with `learn/` beneath it, keeping every logic page together; it loses only that a rule set then sits apart from the other classifiers of `fit/classify/`, whose registry and probes it shares.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Every inference algorithm of `logic` is traced step by step already (a resolution, a decision, a round, a proof event), so the step of a deduction is the library's; the step of a rule learner is one literal added or one rule finished; the step of a version space is one example. Status is about the method alone; what the library has is in the needs column.

### search/logic/propositional/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `propositional-logic` | A formula parsed and taken apart: the truth table one row per step with the models marked, satisfiability, validity and entailment decided (entailment by CDCL on the Tseitin encoding, whose auxiliary variables are the knob to show), the normal forms (NNF, CNF, DNF) and their sizes (CNF blowing up on a long disjunction of conjunctions, Tseitin staying linear), Quine-McCluskey's prime implicants with the covering step to a minimal DNF. | `table/row` m (the valuation and the value), `table/models` s, `formula/cnfClauses` s, `formula/tseitinVariables` s, `implicants/prime` t!, `implicants/cover` t!, `formula/forms` p (the forms as text and TeX) | `logic/propositional.ts`: `parseFormula`, `truthTable`, `models`, `isSatisfiable`, `isValid`, `entails`, `toNnf`, `toCnf`, `toDnf`, `cnfClauses`, `tseitin`, `primeImplicants`, `minimalDnf`, `minimalCnf`, `simplifyFormula`, `formulaToString` (whole functions; the row-by-row and the cover are a few lines in the method) | |

### search/logic/sat/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `sat-solver` | The solver as the mode (DPLL with its branching heuristic as a knob, CDCL in MiniSat's shape with watched literals, first-UIP learning and Luby restarts, WalkSAT with its noise) on random 3-SAT (the clause-to-variable ratio as the knob: the phase transition near 4.26, the hardest instances at the threshold) and on encoded puzzles (Sudoku, n-queens, graph colouring as CNF): decisions, propagations and conflicts on the tape, learnt clauses accumulating, restarts as events, the implication graph of the current conflict as a probe, WalkSAT's unsatisfied-clause count falling and stalling; the assignment as a board for the puzzles. | `search/decisions` s, `search/propagations` s, `search/conflicts` s, `search/learnt` s, `search/restart` e, `search/unsatisfied` s (WalkSAT), `search/flip` m, `assignment/values` t!, `search/implicationGraph` p, `assignment/board` p | `logic/sat.ts`: `dpllAlgorithm`, `cdclAlgorithm`, `walksatAlgorithm`, `cnf`, `parseDimacs`, `satisfies`; all three modes ready; TODO the instances (`problems/sat.ts`: `randomKSat`, `sudokuCnf`, `queensCnf`, `colouringCnf`) | |
| `maxsat-and-counting` | Weighted partial MaxSAT by branch and bound one node per step (the bound and the incumbent closing, the soft clauses falsified), model counting of a small formula, a minimal unsatisfiable core extracted clause by clause (which clauses cannot all hold); the same puzzle with soft preferences. | `search/bound` s, `search/incumbent` s, `search/nodes` s, `search/falsified` t!, `count/models` s, `core/clauses` t! | `logic/maxsat.ts`: `maxsatAlgorithm`, `maxsat`, `countModels`, `allModels`, `unsatCore`; ready | |

### search/logic/logic-programming/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `unification` | Two terms unified by the Martelli-Montanari rules one rule per step (decompose, eliminate, swap, the clash, the occurs check failing on `X = f(X)`: the knob that switches the check off and lets a cyclic term through), the substitution growing; Robinson's algorithm as the one-shot check. | `step/rule` m (the rule and the equation), `step/substitution` t!, `step/equations` s, `step/failed` e | `logic/unify.ts`: `unificationAlgorithm`, `martelliMontanari`, `unify`, `mgu`, `substitutionToString`; ready | |
| `prolog` | SLD resolution watched through Byrd's box model (call, exit, redo, fail, one port per step) on family relations, `append`, n-queens and a DCG: the goal stack, the depth, the choice points, cut pruning them, the answers as they are found, the SLD tree growing as a probe; left recursion looping as the trap (the depth climbing without bound). | `proof/port` m (port, goal, depth), `proof/depth` s, `proof/choicePoints` s, `proof/answers` s, `proof/answer` m, `proof/cut` e, `proof/tree` p (a `tree` record) | `logic/resolution.ts`: `consult`, `sldAlgorithm`, `solve`, `prove`, `sldTree`, `formatTerm`; `logic/parse.ts`: `parseProgram`, `parseQuery`, `translateDcg`; `logic/terms.ts`: `termToString`; ready | |
| `datalog` | Naive against semi-naive evaluation as the mode, one round per step: the facts derived per round (semi-naive touching only the new ones), the least model reached, stratified negation's strata evaluated in order, a fact's derivation tree as a probe; transitive closure on a graph, the same-generation program. | `round/derived` s, `round/total` s, `round/stratum` s, `round/facts` t!, `fact/derivation` p (a `tree` record), `model/relations` p | `logic/datalog.ts`: `datalog`, `datalogAlgorithm`, `evaluateDatalog`, `leastModel`, `queryModel`, `stratify`, `derivationTree`, `checkSafety`; both modes ready | |
| `answer-sets` | Guess and check: candidate models one per step, each reduced by the Gelfond-Lifschitz reduct and kept if it reproduces itself (the stable ones), the integrity constraints cutting candidates; graph colouring and n-queens as answer-set programs, the solutions as boards. | `search/candidate` m, `search/candidates` s, `search/stable` s, `search/reduct` p (the program reduced by the candidate), `model/board` p | `logic/answer-sets.ts`: `normalProgram`, `groundProgram`, `reduct`, `isStableModel`, `stableModelsAlgorithm`, `stableModels`; ready | |

### search/logic/theorem-proving/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `resolution-prover` | A first-order formula to clausal form step by step (negation normal form, Skolemisation, the clauses), then the given-clause loop one given clause per step (the usable set and the set of support, the resolvents generated, subsumption pruning them, the empty clause as the event), the proof drawn as a structure; a non-theorem shown by a finite model found. | `prover/given` m, `prover/usable` s, `prover/setOfSupport` s, `prover/resolvents` s, `prover/subsumed` s, `prover/proved` e, `proof/structure` p (a `diagram` record), `model/finite` p | `logic/first-order.ts`: `parseFol`, `clausalForm`, `resolutionAlgorithm`, `proveFol`, `checkProof`, `proofStructure`, `subsumes`, `hasFiniteModel`, `clauseText`; ready | |

### search/logic/knowledge/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `production-system` | Forward chaining over if-then rules with conflict resolution as the mode (recency, specificity, rule order; the same rule base reaching different conclusions), the Rete network's matches updated incrementally as facts arrive, backward chaining as goal-driven (a Prolog query) beside it, "why" and "how" explanations of a conclusion as a derivation tree. | `cycle/matched` s, `cycle/fired` m, `cycle/facts` t!, `rete/network` p (a `diagram` record), `conclusion/explanation` p (a `tree` record) | `logic/datalog.ts`: `datalogAlgorithm`, `derivationTree` (the bottom-up core); `logic/resolution.ts`: `sldAlgorithm` (the backward mode); TODO `logic/production.ts`: `productionSystem` with the conflict-resolution strategies, `rete` (alpha and beta memories as a structure) | |
| `uncertain-reasoning` | One diagnosis under four calculi as modes: MYCIN's certainty factors combined along the rule chain, Dempster-Shafer belief and plausibility with the combination rule (the conflict mass as the knob that breaks it), possibility theory, and fuzzy inference (Mamdani against Sugeno, the membership functions as knobs, the defuzzified output); the evidence added one item per step and the belief interval narrowing. | `evidence/item` m, `belief/values` t!, `belief/interval` t, `belief/conflict` s, `fuzzy/membership` p, `fuzzy/output` s | `foundations/probability.ts`: `bayesRule` (the probabilistic baseline); TODO `logic/uncertain.ts`: `certaintyFactors`, `dempsterShafer`, `possibility`, `fuzzyInference` (Mamdani, Sugeno, the defuzzifiers) | |
| `argumentation` | An abstract argumentation framework (arguments and attacks as a graph): the grounded extension computed one labelling step per step (in, out, undecided), the preferred and stable extensions as modes (enumerated, the framework with none), an argument added as the knob that flips the outcome. | `labelling/in` s, `labelling/out` s, `labelling/undecided` s, `labelling/step` m, `framework/structure` p (a `diagram` record), `extensions/list` p | `graph/graph.ts` for the framework; TODO `logic/argumentation.ts`: `groundedAlgorithm`, `preferredExtensions`, `stableExtensions` | |
| `description-logic` | An ALC tableau: a concept's satisfiability by expansion rules one rule per step (the completion graph growing, a clash closing a branch, blocking stopping infinite expansion), subsumption as unsatisfiability of the difference, a small ontology classified (the hierarchy as a tree). | `tableau/rule` m, `tableau/nodes` s, `tableau/branches` s, `tableau/clash` e, `tableau/graph` p (a `diagram` record), `ontology/hierarchy` p (a `tree` record) | TODO `logic/description.ts`: `alcTableauAlgorithm`, `subsumes`, `classify` | |

### fit/classify/rules/covering/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `sequential-covering` | Rules grown literal by literal by refinement search (greedy against beam as the mode, the beam width as the knob, the heuristic as a knob: accuracy, Laplace, m-estimate, entropy as CN2 does) and the covered examples removed, one rule per step with its literals as sub-steps: each rule a rectangle on a 2-D dataset with its covered points fading, the rule set's accuracy and size, the last rules covering noise (the stopping threshold as the knob); PRISM's one-class rules as the third mode; the rule set as the classifier (the standard probes of the kind). | `rule/literal` m, `rule/coverage` s, `rule/precision` s, `ruleset/rules` s, `ruleset/accuracy` s, `ruleset/covered` s, `model/rules` p (a `rules` record), `model/regions` p (a decision surface), `model/params` p | `search/refinement.ts`: `refinementBeamSearchAlgorithm`, `refinementBestFirstAlgorithm`, `refinementExhaustiveAlgorithm`, `conjunctions` (the rule-growing search over selectors); `exposition/rules.ts`: `ruleSet`; TODO the covering loop and its heuristics (`lib/rules/covering.ts`: `sequentialCovering`, `cn2`, `prism` as iterative estimators on the `learn` contract) | |
| `ripper` | Grow (FOIL's gain) and prune (reduced error on a held-out third) per rule, the MDL-based stop, then the optimisation pass revisiting each rule with a replacement and a revision (the one that wins as an event); IREP as the mode without optimisation; the rule set's size against a tree's on the same data. | `rule/grown` t!, `rule/pruned` t!, `rule/descriptionLength` s, `pass/revision` m, `ruleset/accuracy` s, `ruleset/size` s, `model/rules` p, `model/regions` p | `logic/ilp/foil.ts`: `foilGain` (the growth criterion); `learn/model-selection/splitters.ts`: `holdout` (the pruning split); TODO `lib/rules/ripper.ts`: `ripper`, `irep` | |

### fit/classify/rules/lists-and-sets/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `rule-lists` | Bayesian rule lists by MCMC over ordered lists of pre-mined rules (the prior on length and rule width as knobs, the chain's lists on the tape, the posterior mode as the model, with `probabilistic-inference`'s samplers) and CORELS's branch and bound (the objective bound and the incumbent, the prefix tree explored, the optimal list certified) as modes; the list read as if-then-else against a tree of the same depth. | `chain/list` t!, `chain/logPosterior` s, `chain/accepted` s, `search/bound` s, `search/incumbent` s, `search/nodes` s, `model/rules` p, `model/regions` p | `search/refinement.ts`: `conjunctions` (the pre-mined rules); `sampling`: the Metropolis kernel; TODO `lib/rules/lists.ts`: `bayesianRuleList`, `corels` | |
| `rule-sets` | OneR (the single best feature's rule per value), decision tables (the feature subset as the knob, the table's cells), and the rules a surrogate tree's paths define, as modes: how far the simplest rule gets on a dataset (OneR's surprising accuracy on the UCI classics). | `model/rules` p, `model/table` p, `ruleset/accuracy` s, `features/chosen` t | `explain/rules.ts`: `treeRules`; `explain/surrogates.ts`: `surrogateTree`; TODO `lib/rules/sets.ts`: `oneR`, `decisionTable` | |
| `rulefit` | Rules harvested from an ensemble's tree paths plus the linear terms, a lasso picking a sparse set: the coefficients' path as λ moves (the knob, or the step over a grid), each rule's support and importance, the model as a `rules` record. | `path/lambda` s, `path/activeRules` s, `path/coefficients` t!, `model/rules` p, `model/importances` p | `explain/rules.ts`: `ruleFit`, `ruleFitPredict`, `treeRules`; `linear/paths.ts`: `lassoPath` (the path view); ready, the tree ensemble being the `trees` the library lacks (`methods/README.md` section 6, problem 12) | |

### fit/classify/rules/version-spaces/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `version-space` | Find-S and candidate elimination over attribute-value examples one example per step: the specific and general boundaries moving towards each other (S generalised by positives, G specialised by negatives), the version space's size shrinking, a query classified by the vote of its members, a noisy example collapsing the space (the event). | `example/label` m, `boundary/specific` t!, `boundary/general` t!, `space/size` s, `space/collapsed` e, `query/vote` p | `logic/ilp/version-spaces.ts`: `findS`, `candidateEliminationAlgorithm`, `candidateElimination`, `versionSpaceMembers`, `classifyByVersionSpace`, `moreGeneral`; ready | |

### fit/relational/ilp/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `foil` | Learning `grandparent` (and the other planted relations of the kinship data) from `parent` facts: one literal per step, the candidate literals scored by information gain (drawn as a ranked list), the clause growing until no negative is covered, the covered positives removed, the next clause; the learnt theory as Prolog clauses against the planted rules. | `clause/literal` m, `clause/candidates` t (the gains), `clause/positives` s, `clause/negatives` s, `theory/clauses` s, `theory/text` p, `theory/accuracy` s | `logic/ilp/foil.ts`: `foilAlgorithm`, `foil`, `foilGain`, `foilClause`; `datasets/graphs.ts`: `kinshipTriples` (family trees with the relations' rules as the truth); ready | |
| `bottom-clause-ilp` | Progol and Aleph: mode declarations restricting the language (the knob), each seed example saturated into its bottom clause (the clause drawn, its length), the lattice between the empty clause and the bottom one searched for the best compression one clause per step, the covering loop; Golem's relative least general generalisation as the bottom-up mode (the lgg of two examples, reduced). | `seed/example` m, `seed/bottomClause` t!, `search/candidate` m, `search/compression` s, `theory/clauses` s, `theory/text` p, `lgg/clause` p | `logic/ilp/progol.ts`: `progolAlgorithm`, `progol`, `bottomClause`, `parseMode`, `canonicalClause`; `logic/ilp/generalisation.ts`: `lgg`, `lggClauses`, `rlgg`, `reduceClause`, `thetaSubsumes`; both modes ready | |
| `meta-interpretive-learning` | Metagol: metarules (chain, inverse, tail recursion) instantiated during the proof of the positive examples, the program growing clause by clause with invented predicates, the smallest program that proves every positive and no negative (the size bound as the step); Popper's learning from failures as the other mode: each tested hypothesis failing as too general or too specific and the constraints it adds pruning the space (the space's size on the tape). | `search/hypothesis` m, `search/size` s, `search/failures` s (Popper), `search/constraints` s, `program/clauses` t!, `program/text` p, `program/invented` s | `logic/ilp/meta-interpretive.ts`: `metagolAlgorithm`, `metagol`, `METARULES`, `learnFromFailuresAlgorithm`, `learnFromFailures`; both modes ready | |

### fit/relational/statistical/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `markov-logic` | Weighted first-order formulas (smokes, friends, cancer) grounded into a Markov network over a small world (the ground network's size as objects are added: the knob), the weights learnt by pseudo-likelihood gradient one step per step, inference by MC-SAT (WalkSAT inside a slice sampler) or Gibbs as the mode, a query's marginal against exact inference on the tiny world. | `ground/atoms` s, `ground/cliques` s, `training/pseudoLikelihood` s, `training/weights` t!, `inference/marginal` s, `inference/exact` s, `ground/network` p (a `diagram` record) | `logic/first-order.ts`: `parseFol`, `clausalForm`; `pgm/models/markov-network.ts`: `markovNetwork`; `pgm/inference/sampling.ts`: `gibbsSamplingAlgorithm`; `pgm/inference/elimination.ts`: `variableElimination` (the exact check); `logic/sat.ts`: `walksat`; TODO `logic/markov-logic.ts`: `groundMln`, `mcSat`, `pseudoLikelihoodGradient` | |
| `probabilistic-logic-program` | ProbLog: probabilistic facts and rules, a query's success probability by weighted model counting over its proofs (the SLD proofs collected as a probe, the DNF they form, its probability by inclusion-exclusion or a BDD), the parameters learnt by EM from observed queries one iteration per step. | `query/proofs` s, `query/probability` s, `training/logLikelihood` s, `training/parameters` t!, `query/proofTree` p | `logic/resolution.ts`: `sldAlgorithm`, `sldTree` (collecting the proofs); TODO `logic/problog.ts`: `successProbability` (weighted model counting), `problogEm` | |

### fit/relational/neuro-symbolic/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `neuro-symbolic` | A differentiable semantics of formulas (t-norms as the knob: product, Łukasiewicz, Gödel) over a network's outputs: the formulas' satisfaction as a loss term rising on the tape beside the data loss, a classifier obeying a rule it was not shown data for (the rule's weight as the knob), and the DeepProbLog form (the network's outputs as probabilistic facts of the program above) as the other mode; where the rule helps and where it fights the data. | `training/dataLoss` s, `training/ruleSatisfaction` s, `training/ruleLoss` s, `eval/accuracy` s, `eval/ruleViolations` s, `model/params` p | `nn/training/fit.ts`: `fitAlgorithm`; `logic/propositional.ts`: `parseFormula`; TODO `logic/differentiable.ts`: `tNorms`, `formulaSatisfaction` (formulas as tensor operations over `nn`), the ProbLog bridge of the row above | |

### analyse/explain/rules/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `explanation-by-rules` | see trustworthy-machine-learning (`interpretability/`: anchors by beam search with their precision and coverage, `explain/surrogates.ts`: `anchors`, `surrogateTree`; `explain/rules.ts`: `treeRules`; the `rules` record) | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `knowledge-bases.ts` | The knowledge-base slot: named programs and fact bases with their truths (family relations from `datasets/graphs.ts`: `kinshipTriples` with the planted rules, `append` and `member`, n-queens and graph colouring as Prolog, Datalog and answer-set programs, transitive closure on a small graph, the smokers world of Markov logic), parsed by `logic/parse.ts` | `logic-programming/`, `ilp/`, `statistical/`, `knowledge/` |
| `formulas.ts` | The formula slot: named propositional and first-order formulas (the textbook tautologies and the pigeonhole principle), random k-SAT with its ratio, the puzzle encodings (Sudoku, queens, colouring) as CNF with their board layouts for the assignment board | `propositional/`, `sat/`, `theorem-proving/` |
| `rules.ts` | The rule record over `exposition/rules.ts`: `ruleSet` (a rule with its precision, coverage, coefficient), coverage probes on the training rows, the rule-as-region overlay (a rule's conjunction as a rectangle on the two drawn features) and the rule set as a classifier with the standard probes of the kind | `fit/classify/rules/*`, `ilp/` |
| `proofs.ts` | Proofs and derivations as `tree` and `diagram` records (an SLD tree, a derivation tree, a resolution proof, a tableau, a Rete network), and the Byrd-box event record of a Prolog run | `logic-programming/`, `theorem-proving/`, `knowledge/`, `statistical/` |

## Examples to make

- Sequential covering on a 2-D dataset: each rule as a rectangle appearing and its covered points fading, the beam width as the knob.
- FOIL learning `grandparent` from `parent` facts, the candidate literals scored each step, the theory read out as Prolog.
- CDCL on random 3-SAT at the threshold: decisions, conflicts and learnt clauses on the tape, restarts as events; WalkSAT on the same formula beside it.
- A Prolog query stepped through Byrd's boxes, the SLD tree growing, cut pruning a branch.
- Candidate elimination on a small concept, the S and G boundaries meeting.
- Metagol inventing a predicate to learn a recursive definition in a handful of clauses.

## Rendering

- Rules (`rules` view), rectangles over rows, clause text (`tokens` or TeX), coverage curves, trees (`tree`) for proofs and derivations, structures (`diagram`) for implication graphs, Rete networks and proofs, boards for the encoded puzzles.
- Missing: a rule-as-region overlay on the rows panel; a clause view with variables bound at the viewed step; a boundary view for version spaces (S and G as two frontiers in the hypothesis lattice).

## Problems

- The deduction half is entirely in the library and traced; the induction half is the gap: rule induction (covering, RIPPER, rule lists and sets) has its search in `search/refinement.ts` and nothing else, and goes to a `lib/rules/` domain on the `learn` contract so a rule set is a classifier like any other; Markov logic, ProbLog and the differentiable semantics are a `logic/` file each.
- The classical knowledge-based systems of `docs/structure.md` 5.6 (production systems with Rete, certainty factors, belief functions, fuzzy inference, argumentation, description logics, case-based reasoning) have never been written; the rows under `knowledge/` name them so the chapter is complete, and each is a small `logic/` module.
- `rules/` of the old plan moves here; `gofai/` splits between here (logic, knowledge) and `search-and-planning` (planning, CSP); the puzzle encodings as CNF are shared with its `constraint/` and belong in `lib/problems`.
- Explanation by rules is split by the truth it compares with: a rule set fitted to data is here, a rule explaining a black box is `trustworthy-machine-learning`'s.
