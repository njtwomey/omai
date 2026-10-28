# Sequence models (part 6, neural architectures; with efficient and state-space sequence models)

Recurrent and sequence-to-sequence networks, and their efficient successors: recurrent networks (Elman, LSTM, GRU) stepped through a sequence, backpropagation through time, vanishing and exploding gradients, sequence-to-sequence with attention, and the efficient family (linear attention, linear recurrent models, structured and selective state-space models) compared and combined, as sessions over `nn`, `text` and the trainer base.

## Capabilities

- A recurrent cell stepped over a sequence: the hidden state as a bar at each position, the gates of an LSTM as probes.
- Training on small sequence tasks (copying, adding, parity, a tiny language) with the loss on the tape and the hidden state trajectories as probes.
- Gradient flow through time: the gradient norm per position for each cell type.
- Sequence to sequence: an encoder-decoder with attention on a toy translation, the attention matrix at the viewed step.
- State-space models: a linear recurrence as a convolution, the kernel drawn; a selective state-space model's gates.

## Examples to make

- An LSTM on the adding problem: the gates over the sequence for one example at the viewed step.
- Vanishing gradients: gradient norms by position for the Elman cell against the LSTM.
- Attention on a toy sequence-to-sequence task as the alignment forms over training.
- A linear recurrent model's kernel against the LSTM's hidden-state trajectory on the same task.

## Rendering

- Tokens and state sequences (`tokens`, `stateSequence`), heatmaps of attention and gates, hidden-state trajectories, curves.
- Missing: a position-aligned panel (tokens on x with per-position values beneath).

## Shared

- `shared/`: the sequence task slot (named synthetic tasks with truth), the per-position probes; the trainer base.

## Problems

- New chapter; `neural/` sequence parts move here; `efficient-sequence-models` folded in.
