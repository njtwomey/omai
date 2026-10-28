# Sequence models (part 6, neural architectures; with efficient and state-space sequence models)

Recurrent and sequence-to-sequence networks, and their efficient successors: recurrent cells (Elman, LSTM, GRU) stepped through a sequence, backpropagation through time with vanishing and exploding gradients, training on the algorithmic tasks (copy, reverse, sort, parity, addition, Dyck, associative recall), sequence-to-sequence with attention and beam search, temporal convolutions, and the state-space family (a linear recurrence as a convolution, S4-style diagonal models, the linear recurrent unit, selective scans, the delta rule, linear attention) compared on one task, as sessions over `nn`, `datasets/sequence-tasks` and the trainer base. "Sequence model" means two things: the probabilistic chains (HMMs, CRFs) are `probabilistic-inference`'s; this chapter is the neural one.

## Structure

Problem methods of the torch world (section 7.2 of `methods/README.md`) under `train/sequence/`, one folder per family, with the cell or block as the mode of one method rather than a method per cell; the stepped cell and the gradient-flow study are probe-only methods of the same tree. The efficient family is one folder, `state-space/`, since every member is a linear recurrence seen two ways. The transformer and the language models are this tree's other two folders and their own chapters (`transformers`, `large-language-models`); neural forecasters are `time-series`' `fit/forecast/neural/`.

```
train/sequence/
  recurrent/        a cell stepped, the recurrent network on the algorithmic tasks (the cell as a mode), gradient flow through time, bidirectional and stacked
  seq2seq/          encoder-decoder with attention, teacher forcing against free running, beam search, CTC alignment
  convolutional/    temporal convolutions and WaveNet, dilation and the receptive field
  state-space/      the linear recurrence as a convolution, diagonal state spaces and the LRU, selective scan, the delta rule, linear attention, the family compared
  shared/           the sequence task slot with truth, per-position probes, windows and padding (below)
  transformers/     the transformers chapter
  language/         the large-language-models chapter
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Trained rows step one optimiser update on the trainer base (`train/loss`, `train/gradNorm`, `train/lr`, `eval/<metric>`, `model/params` as checkpoints; `TODO: methods/shared/bases/trainer.ts` over `nn/training/fit.ts`: `fitAlgorithm`); the stepped cell steps one position. Per-position values of one example at the viewed step (hidden states, gates, attention, gradients) are the `hidden` t! family, drawn under the tokens. Nothing in this chapter is built.

### recurrent/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `recurrent-cell` | One cell stepped over a sequence, one position per step: the hidden state as a bar at each position, the gates of an LSTM (input, forget, output) and a GRU (reset, update) as probes, the cell as the mode (Elman, LSTM, GRU); a hand-set cell that counts or remembers one bit shows what the gates are for. A probe-only page with random or trained weights. | `position/t` s, `hidden` t!, `cell/gates` t! (per position), `cell/picture` p (tokens with the state beneath) | `nn/functional/recurrent.ts`: `rnnCell` (elman ready), `lstmCell` (lstm ready), `gruCell` (gru ready), `scan`; `exposition/sequences.ts`: `stateSequence` | |
| `recurrent-network` | The cell as the mode on the algorithmic tasks (copy, reverse, sort, parity, addition, Dyck as the task slot; the length as the knob): the loss on the tape, the accuracy by position (the end of a long sequence fails first), the hidden-state trajectory of one example as a projection, the gates of one example at the viewed step; the Elman cell failing on the long copy where the LSTM and GRU succeed; truncated BPTT with the window as a knob. | `train/loss` s, `train/gradNorm` s, `eval/accuracy` s, `eval/accuracyByPosition` t!, `hidden` t!, `hidden/trajectory` p (a projection of one example's states), `cell/gates` t!, `model/params` p, `model/predictions` p | `nn/module/recurrent.ts`: `RNN`, `LSTM`, `GRU`; `nn/functional/recurrent.ts`: `packPaddedSequence`, `padPackedSequence`, `lengthsFromMask`; `datasets/sequence-tasks.ts`: `copyTask`, `reverseTask`, `sortTask`, `parityTask`, `additionTask`, `dyckTask`; `losses/classification.ts`: `crossEntropy`; `optim/transform.ts`: `clipByGlobalNorm` | |
| `gradient-flow` | Backpropagation through time watched: the gradient norm of the loss at the last position with respect to the hidden state at each earlier position, per cell type (the Elman cell's vanishing or exploding by the spectral radius of its recurrent weight as the knob, the LSTM's near-constant error carousel); gradient clipping's effect as an event; at initialisation and at the viewed training step. | `gradient/byPosition` t!, `gradient/ratio` s (first against last), `weight/spectralRadius` s, `train/clipped` e, `gradient/picture` p | `autodiff/transforms.ts`: `vjp`, `grad` (with respect to the per-position states, through `scan`); `nn/functional/recurrent.ts`: `scan`; `numerics/linalg/eig.ts`: `powerIteration` (the spectral radius); `optim/transform.ts`: `clipByGlobalNorm` | |
| `bidirectional-and-stacked` | A tagger (a label per position) with directions and depth as knobs: the backward direction's gain on tasks that need the future (the bracket's match), the layers' hidden states stacked as a probe, residual connections between layers at depth. | `train/loss` s, `eval/accuracy` s, `layer/<i>/hidden` t!, `model/predictions` p | `nn/blocks/sequence.ts`: `StackedRecurrent`; `nn/module/recurrent.ts`: `LSTM`, `GRU` (bidirectional); `datasets/sequence-tasks.ts`: `dyckTask`, `parityTask` (prefix mode) | |
| `character-language-model` | see large-language-models | | | |
| `neural-forecaster` | see time-series | | | |

### seq2seq/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `encoder-decoder` | An encoder summarising the source into one vector and a decoder unrolling the target (reverse and sort as the toy translations): the bottleneck failing as the length grows (the knob) and attention fixing it (modes: none, dot-product, additive); the alignment matrix at the viewed step forming the diagonal (reverse) or the permutation (sort); teacher forcing against free running with scheduled sampling as the knob (exposure bias as the gap between them). | `train/loss` s, `eval/accuracy` s, `eval/sequenceAccuracy` s, `attention/alignment` t! (one example), `hidden` t!, `model/predictions` p, `model/params` p | `nn/module/recurrent.ts`: `GRU`, `LSTM`; `nn/functional/attention.ts`: `attention` (dot-product ready; TODO the additive scorer of Bahdanau as an option of `attention` or a line in `shared/`); `datasets/sequence-tasks.ts`: `reverseTask`, `sortTask`, `additionTask` | |
| `beam-search` | Decoding the trained model at the viewed training step: greedy against beam search with the width as the knob, the beam's hypotheses and scores per step as a board, the length penalty, when a wider beam hurts; sampling as the third mode (`large-language-models` owns the sampling knobs). | `decode/beam` t! (hypotheses and scores per position), `decode/finished` e, `decode/output` p, `eval/sequenceAccuracy` s | `nn/decoding/beam.ts`: `beamSearchAlgorithm`, `beamSearch`; `nn/decoding/generate.ts`: `decodingAlgorithm`; `nn/language-model.ts`: `languageModelFromModule` (the decoder as a `LanguageModel`) | |
| `ctc` | Alignment-free sequence labelling: the output lattice with blanks, the forward-backward over alignments (the trellis as a board), the collapse rule, the loss on the tape; on the parity-of-prefixes and a stretched-copy task where the input is longer than the target. | `train/loss` s, `ctc/trellis` t!, `ctc/alignment` p (the best path), `eval/editDistance` s, `model/predictions` p | TODO `losses/sequence.ts`: `ctc` (the forward-backward over the blank-augmented lattice, with the trellis available for the probe); `metrics/text.ts`: `characterErrorRate` | |

### convolutional/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `temporal-convolution` | A TCN (causal, dilated residual blocks) against the LSTM on the same task: the receptive field growing exponentially with depth (drawn on the sequence for one output position), the dilation and depth as knobs, the per-position accuracy, the parallel cost against the recurrent one; WaveNet's gated blocks as the mode. | `train/loss` s, `eval/accuracy` s, `conv/receptiveField` s, `conv/field` p (over the input), `layer/<i>/activations` t!, `model/params` p | `nn/blocks/sequence.ts`: `TCN` (`receptiveField`), `TemporalBlock`, `WaveNet`, `WaveNetBlock`, `causalPad`; `nn/functional/conv.ts`: `conv1d` | |

### state-space/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `linear-recurrence` | A continuous state-space model discretised (zero-order hold against bilinear as modes, the step Δ as the knob) and run two ways: as a recurrence (one position per step) and as a convolution with its kernel drawn (the same output, the check on the tape); the kernel's shape from the eigenvalues of A (decaying, oscillating); diagonal A as the structure S4 and its successors keep. A probe-only page. | `kernel/values` t!, `recurrence/state` t!, `output/recurrent` t, `output/convolutional` t, `check/difference` s, `system/eigenvalues` p | `nn/functional/state-space.ts`: `discretise`, `discretiseDiagonal`, `ssmKernel`, `causalConvolution`, `ssmRecurrence`, `linearRecurrence` | |
| `linear-recurrent-unit` | The LRU trained on the task slot: the eigenvalues (its parametrisation in the unit disc) moving during training (drawn in the disc at the viewed step), the initialisation's ring (the radius range as a knob) and why it matters for long memory, the kernel each eigenvalue contributes; against the LSTM on the long copy. | `train/loss` s, `eval/accuracy` s, `lru/eigenvalues` t! (in the disc), `lru/kernel` p, `hidden` t!, `model/params` p | `nn/functional/state-space.ts`: `lru`, `lruInit`, `lruEigenvalues`; `datasets/sequence-tasks.ts`: `copyTask` (long), `associativeRecall` | |
| `selective-scan` | Mamba's input-dependent Δ, B and C: the gates per position of one example (where the state is written and where it is reset) at the viewed step, against the time-invariant model on associative recall (which a fixed kernel cannot solve and selection can); the state as a probe over positions. | `train/loss` s, `eval/accuracy` s, `scan/delta` t!, `scan/gates` t!, `scan/state` t!, `model/params` p | `nn/functional/state-space.ts`: `selectiveScan`; `datasets/sequence-tasks.ts`: `associativeRecall`, `copyTask` | |
| `linear-attention-and-delta-rule` | Softmax attention against linear attention (a running sum of key-value outer products: the state as a matrix drawn per position) against the delta rule (the state corrected rather than summed, so a key's value can be overwritten) as modes on associative recall: the recall accuracy against the number of pairs (the knob), the state's rank; the costs linear against quadratic in length. | `train/loss` s, `eval/accuracy` s, `state/matrix` t! (per position), `state/rank` s, `attention/weights` t! (softmax mode), `study/pairsCurve` proj | `nn/functional/attention.ts`: `linearAttention`, `attention`; `nn/functional/state-space.ts`: `deltaRule`; `datasets/sequence-tasks.ts`: `associativeRecall` | |
| `family-comparison` | One task, one budget, every member as a session on the same keys: LSTM, TCN, transformer (`transformers`'), S4-style diagonal, LRU, selective scan, linear attention, delta rule: the loss curves overlaid, accuracy against sequence length (the sweep), the cost per position (recurrent, convolutional, quadratic); the page the chapter's title promises. | `study/curves` proj, `study/lengthCurve` proj, `study/cost` t, `eval/accuracy` s (per session) | the rows above; the comparison of runs on one cursor (`methods/README.md` section 5, render); `nn/describe.ts`: `functionFlops` (the cost) | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `tasks.ts` | The sequence task slot: the eight algorithmic tasks of `datasets/sequence-tasks.ts` (copy, reverse, sort, Dyck, addition, associative recall, parity, modular arithmetic) with their truths, the length, alphabet and number of pairs as knobs, the special tokens, and a stretched-copy variant for CTC; shared with `transformers` and `large-language-models` | every group |
| `positions.ts` | The per-position record: tokens on x with values beneath (hidden states, gates, gradients, attention rows) for one example at the viewed step, built on `exposition/sequences.ts`: `stateSequence`; the accuracy-by-position curve | every group |
| `windows.ts` | Padding, masks and lengths (`packPaddedSequence`, `lengthsFromMask`), truncated-BPTT windows, the causal padding of temporal convolutions | `recurrent/`, `convolutional/`, `state-space/` |
| `decoding.ts` | The decoder as a `LanguageModel` (`nn/language-model.ts`: `languageModelFromModule`) so beam search and sampling run on any trained seq2seq model; the decoding slot table is `large-language-models`' | `seq2seq/` |

## Examples to make

- An LSTM on the adding problem: the gates over the sequence for one example at the viewed step, the hidden trajectory as a projection.
- Vanishing gradients: gradient norms by position for the Elman cell against the LSTM, the spectral radius as the knob.
- Attention on reverse as the alignment forms over training, the bottleneck model beside it failing at length 20.
- A linear recurrence run as a recurrence and as a convolution, the kernel drawn, the two outputs equal.
- Associative recall: selective scan and the delta rule succeeding where a fixed kernel and linear attention saturate.
- The family comparison: six sessions on the long copy, loss curves overlaid, accuracy against length.

## Rendering

- Tokens and state sequences (`tokens`, `stateSequence`), heat maps of attention, gates and kernels, hidden-state trajectories (`trajectory`, `embeddingSteps`), eigenvalues in the unit disc, curves.
- Missing: a position-aligned panel (tokens on x with per-position values beneath, following the cursor); a beam board; the comparison panel for the family study.

## Problems

- Efficient sequence models are folded in as `state-space/` and should stay so: every member is a linear recurrence with a different parametrisation or selection rule, and the family comparison is the chapter's point. A split would separate the comparison from the LSTM it is measured against.
- The library has every cell, scan and state-space kernel; it lacks CTC, the additive attention scorer and nothing else named here.
- Three chapters share the task slot and the per-position record (`transformers`, `large-language-models`); one `shared/tasks.ts` and `shared/positions.ts` should serve all three, which argues for a `train/sequence/shared/` above the chapters' own `shared/` folders.
