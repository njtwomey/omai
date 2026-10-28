# Trustworthy machine learning (part 8, practice)

Models examined and made trustworthy: label noise (its effect, robust losses, cleaning), interpretability (Shapley values, LIME, anchors, counterfactuals and recourse, partial dependence and ICE, attributions for networks, concept activation, prototypes, explanation quality), fairness (group, individual and intersectional measures, bias in data, mitigation before, during and after training, the frontier and the impossibility), robustness (adversarial examples and training, certified bounds, corruption robustness), and privacy (differential privacy in training, membership inference), as sessions over `explain`, `fairness`, `metrics`, `losses`, `nn` and the models of `supervised-learning`. Sections as subfolders: `label-noise/`, `interpretability/`, `fairness/`, `robustness/`, `privacy/`.

## Capabilities

- Interpretability: every explainer in `explain` as a slot over any fitted model, with the explanation of a chosen point at the viewed step where the explainer iterates (kernel SHAP's samples, LIME's perturbations, a counterfactual's search, anchors' beam); explanation quality (faithfulness, stability) on the tape.
- Fairness: audits across groups as the threshold moves, mitigation methods stepped (reweighing, the exponentiated-gradient reduction's iterations, threshold optimisation), the accuracy-parity frontier traced, the impossibility shown on planted base rates.
- Robustness: an adversarial example found by projected gradient descent step by step, adversarial training's effect on the decision surface, corruption robustness curves, certified radii.
- Privacy: DP-SGD's clipping and noise as knobs with the privacy budget accounted on the tape, membership-inference attacks' success against the budget.
- Label noise: planted noise rates, robust losses and co-teaching, confident learning's cleaning (with `data-centric-learning`).

## Examples to make

- Kernel SHAP's coalitions sampled one at a time, the Shapley values converging for one row.
- A counterfactual walking across the decision boundary step by step with the recourse cost.
- The fairness frontier traced by the exponentiated-gradient reduction, each iteration a point.
- PGD finding an adversarial example on a two-moons network, the perturbation as the knob.
- DP-SGD: accuracy against epsilon, noise and clipping as knobs.

## Rendering

- Attributions (`featureAttributions`), dependence (`dependence`), counterfactuals, group reports, Pareto fronts, decision surfaces with the adversarial path, curves against the budget.
- Missing: the frontier as a step record that follows the cursor; a per-row explanation probe drawn on the rows panel.

## Shared

- `shared/`: the fitted-model slot (any method's fitted state as a probe target), the explainer slot table, the protected-attribute data modifiers with truth, the budget accountant record.

## Problems

- `explain/`, `fairness/`, `trust/` (robustness, model cards), `privacy/` and `adversarial/` move here as sections; calibration, conformal and uncertainty go to `decision-theory-and-calibration`, shift to `shift-and-drift`.
