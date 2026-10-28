# Data-centric learning (part 3, data)

Learning that changes the data rather than the model: labelling and annotation, augmentation and synthetic data, data selection and valuation (coresets, Shapley values of data, influence), and error analysis, as sessions over `learn`, `explain` (influence) and the model slots.

## Capabilities

- Labelling: label models over noisy annotators (Dawid-Skene by EM, majority vote) stepped; the connection to `weak-supervision`.
- Augmentation: each augmentation's effect on a model's score and on its decision surface; mixup, cutout, noise, as knobs.
- Data valuation: leave-one-out and Shapley values of data points estimated by sampling, influence functions; coresets and the model fitted on them.
- Error analysis: slices with the worst error, confusion by subgroup, label errors found by confident learning.

## Examples to make

- Dawid-Skene's confusion matrices per annotator converging, the true labels recovered.
- Data Shapley by Monte Carlo: the values settling, the most valuable and most harmful points marked on the rows.
- Mixup as the knob: the decision surface smoothing.
- Confident learning: the suspected label errors ranked, against the planted noise.

## Rendering

- Rows with per-point values as colour, confusion matrices (`ContingencyTable`), decision surfaces, bars by slice.
- Missing: a per-point value overlay on the rows panel with a legend scale.

## Shared

- `shared/`: the planted label noise and annotator models with truth (shared with `weak-supervision` and `trustworthy-machine-learning/label-noise`).

## Problems

- New chapter. Data Shapley nests thousands of fits: a budgeted estimator and the summarising sweep base.
