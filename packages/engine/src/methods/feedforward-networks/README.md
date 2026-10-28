# Feedforward networks (part 6, neural architectures)

The multilayer perceptron and its training: perceptrons and the XOR problem, activations, initialisation, backpropagation watched, optimisers and schedules, regularisation (weight decay, dropout, early stopping, data augmentation), normalisation layers, the loss landscape, and energy-based networks (Hopfield, Boltzmann machines), as sessions over `nn`, `optim`, `losses`, `autodiff` and the trainer base.

## Capabilities

- The trainer base with every slot open: the architecture (widths, depth, activation, normalisation), the objective (any loss), the optimiser and schedule, the batch size, the initialisation, regularisation; log the loss and metrics per step or epoch, the parameter norms, the gradient norms per layer; probes: the decision surface or fit, each layer's activations for the rows, the weights as images, the loss landscape slice.
- Backpropagation stepped: one forward and backward pass on a tiny network with every intermediate value shown.
- Initialisation studies: activation and gradient statistics by depth at step 0 for each scheme.
- Regularisation studies: the fit with and without each regulariser, on a small noisy dataset.
- Energy-based: a Hopfield network recalling patterns, a restricted Boltzmann machine trained by contrastive divergence with its filters.

## Examples to make

- A two-layer network on two moons: the decision surface forming, the hidden units' boundaries as a probe.
- Backpropagation on a three-neuron network, one value per step.
- Dying ReLUs: the fraction of dead units against the learning rate.
- Weight decay against dropout against early stopping on a noisy fit, one page, three sessions.
- Hopfield recall of corrupted patterns, energy falling step by step.

## Rendering

- Decision surfaces and fits, curves per layer, images of weights (`image`), computation graphs (`diagram`), activation histograms by layer.
- Missing: a per-layer small-multiples panel (one chart per layer, aligned); the architecture diagram with live shapes.

## Shared

- `shared/`: the trainer base is this chapter's and every other part-6 chapter's: modules, optimiser, batches, the step, evaluation at a schedule, parameter checkpoints for probes.

## Problems

- `neural/` splits across part 6; this chapter takes its foundations and training studies. Training at a watchable rate on the main thread: the worker producer.
