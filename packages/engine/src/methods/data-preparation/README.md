# Data preparation (part 3, data)

Preparation watched: cleaning and imputation, scaling and transforms, encoding of features and targets, resampling and weighting, pipelines fitted on training data only, and splitting without leakage, as sessions over `learn` (preprocessing, pipelines, splits) and `datasets` (modifiers that plant the problem).

## Capabilities

- Imputation: mean, k-nearest-neighbour and iterative (MICE) imputation stepped, under each missingness mechanism; options: the mechanism, the rate, the imputer; log the imputed values against the hidden truth.
- Outliers and errors: detection on columns, the effect on a model downstream.
- Scaling and transforms: each scaler's effect on a model's fit and on gradient descent's path; Box-Cox and Yeo-Johnson's parameter found.
- Encoding: one-hot, ordinal, target (cross-fitted), hashing, with the leakage of naive target encoding shown.
- Resampling and weighting: oversampling, SMOTE, class weights, on an imbalanced problem.
- Splitting and leakage: random, stratified, grouped, temporal splits; leaks planted and detected (adversarial validation, duplicates across splits).

## Examples to make

- A planted leak: a feature derived from the target, the validation score too good, adversarial validation catching it.
- Target encoding with and without cross-fitting, the train and test gap.
- Scaling as the knob: gradient descent's path on unscaled against standardised features.
- MICE converging, the imputations' distribution against the truth.

## Rendering

- Rows with an overlay (missing cells, imputed values), the learning-curve and score panels, pipeline stages (`pipelineStages`).
- Missing: a table panel for small datasets with cells coloured by status (missing, imputed, leaked).

## Shared

- `shared/`: planted problems (a modifier catalogue: missingness, leaks, duplicates, imbalance, shift) with their truths, shared with `shift-and-drift` and `data-centric-learning`.

## Problems

- New chapter. Most steps are one-shot transforms; the sessions are studies with a model downstream, so each needs a model slot filled from `supervised-learning`.
