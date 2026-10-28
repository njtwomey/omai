# Weak supervision (part 5, learning paradigms)

Labels that are indirect: multiple-instance learning, learning from label proportions, positive-unlabelled and partial labels, crowdsourced labels and label models (majority vote, Dawid-Skene), and programmatic weak supervision (labelling functions and the label model), as sessions over `learn`, `mixture` (EM) and the models of `supervised-learning`.

## Capabilities

- Label models: majority vote, Dawid-Skene by EM, the generative label model over labelling functions; options: the annotators' or functions' accuracies and overlaps as planted knobs; log the estimated accuracies and the recovered labels' quality.
- Multiple-instance: bags with a positive instance, mi-SVM and MIL pooling stepped.
- Positive-unlabelled: the class prior estimated, the unbiased risk estimator trained.
- Label proportions: a model fitted to bag-level proportions.
- The downstream model trained on the weak labels against clean labels.

## Examples to make

- Labelling functions with planted accuracies: the label model's estimates converging, the weak labels against the truth.
- Dawid-Skene with one bad annotator: its confusion matrix found.
- Positive-unlabelled on two moons: the prior as the knob, the boundary moving.

## Rendering

- Rows with weak labels as colour and disagreement as a marker, confusion matrices per annotator, curves of accuracy.
- Missing: a labelling-function matrix panel (functions against points, votes as cells).

## Shared

- `shared/`: planted annotators and labelling functions with truth, shared with `data-centric-learning`.

## Problems

- Keeps its name. The label model and its EM belong in `lib` (`learn` or `pgm`) before the methods wrap them.
