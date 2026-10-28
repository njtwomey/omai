# Logic and rule learning (part 5, learning paradigms)

Rules and logic learned from data: rule induction (sequential covering, RIPPER, rule lists and rule sets, decision tables), inductive logic programming (FOIL, inverse entailment on small knowledge bases), statistical relational learning (Markov logic networks), neuro-symbolic learning, and explanation by rules (anchors, surrogate rules), as sessions over `logic`, `learn`, `explain` and the tree algorithms.

## Capabilities

- Sequential covering: rules grown literal by literal and the covered examples removed, the rule set's accuracy on the tape; options: the search (greedy, beam), the stopping and pruning.
- RIPPER: the grow and prune phases, the optimisation pass.
- Rule lists: Bayesian rule lists by MCMC over lists (with `probabilistic-inference`), CORELS's branch and bound.
- ILP: FOIL's clause search on a family-relations base, the hypothesis clause by clause.
- Markov logic: weights learned by gradient on a small world, inference by MC-SAT.
- Explanation by rules: anchors found by beam search for a prediction of any classifier.

## Examples to make

- Sequential covering on a 2-D dataset: each rule as a rectangle appearing and its covered points fading.
- FOIL learning `grandparent` from `parent` facts, the candidate literals scored.
- An anchor for one prediction: the candidate anchors and their precision and coverage.

## Rendering

- Rules (`rules` view), rectangles over rows, clause text (`tokens` or TeX), coverage curves.
- Missing: a rule-as-region overlay on the rows panel; a clause view with variables bound.

## Shared

- `shared/`: the rule record and coverage probes, the knowledge-base slot (`logic` programs).

## Problems

- `rules/` moves here; `gofai/` splits between here (logic) and `search-and-planning` (planning, CSP).
