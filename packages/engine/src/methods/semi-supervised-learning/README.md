# Semi-supervised learning (part 5, learning paradigms)

Learning from a few labels and many unlabelled points: generative models with unlabelled data (EM over a mixture with labels), low-density separation (transductive SVM, entropy minimisation), self-training and co-training, consistency regularisation (Π-model, mean teacher, FixMatch), graph-based methods (label propagation and spreading), as sessions over the models of `supervised-learning`, `mixture`, `nn` and `graph`.

## Capabilities

- Every method with the label fraction as the knob and the unlabelled points' assignments as a probe over steps.
- Self-training: the confidence threshold, the pseudo-labels added each round.
- Label propagation and spreading on a kNN graph: the labels diffusing iteration by iteration, the graph's bandwidth as a knob.
- Consistency regularisation: a small network trained with augmentations, the decision surface smoothing between the clusters.
- The failure cases: the cluster assumption broken, pseudo-labels reinforcing an error.

## Examples to make

- Two moons with four labels: label propagation step by step, against a supervised fit on the four alone.
- Self-training's pseudo-labels appearing round by round, with a planted error spreading.
- EM with labels on a mixture: the labelled points pinning components.

## Rendering

- Rows with assignments and confidence as colour, decision surfaces, the kNN graph with label mass as colour (structure view).
- Missing: a labelled-against-unlabelled marker style on the rows panel; the graph overlay on points.

## Shared

- `shared/`: the partial-label data modifier (which points keep labels) with truth, shared with `weak-supervision` and `active-learning`; the kNN graph builder.

## Problems

- `semi-supervised/` moves here (renamed for consistency with the encyclopedia's chapter).
