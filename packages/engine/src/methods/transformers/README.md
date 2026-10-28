# Transformers (part 6, neural architectures)

The transformer watched: the original encoder-decoder, positional encodings, the architecture families (encoder-only, decoder-only, encoder-decoder), attention variants (multi-head, multi-query, sparse, local), the modern decoder block (pre-norm, rotary positions, gated MLPs), training and analysis (loss curves, learning-rate warm-up, scaling at toy size), and interpreting transformers (attention maps, induction heads, logit lens), as sessions over `nn`, `text` and the trainer base.

## Capabilities

- Attention stepped: queries, keys and values for a short sequence, the scores, the softmax, the weighted sum, each a probe; options: heads, masking, positions.
- A tiny decoder trained on a small corpus or a synthetic task (sorting, induction): the loss on the tape, the attention maps per head at the viewed step, the next-token distribution for a prompt.
- Positional encodings compared: sinusoidal, learned, rotary; extrapolation beyond the training length.
- Interpretation: the logit lens over layers, an induction head forming over training.
- Decoding: greedy, beam, sampling with temperature and top-k as knobs (with `large-language-models`).

## Examples to make

- Attention on an eight-token sequence: one head's scores and the weighted values, mask as a knob.
- An induction head forming: the attention map on a repeated sequence over training.
- The next-token distribution for a prompt as training proceeds.
- Rotary against sinusoidal positions on a length-extrapolation task.

## Rendering

- Heatmaps of attention per head, tokens with per-token values, distributions over a vocabulary as bars, diagrams of the block.
- Missing: a heads-grid panel (every head's map at the viewed step), a token-probability strip.

## Shared

- `shared/`: the token task slot, the attention probes; the trainer base.

## Problems

- New chapter. Toy sizes only on the main thread; the worker producer again.
