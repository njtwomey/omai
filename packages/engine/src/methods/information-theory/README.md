# Information theory (part 2, signals, systems and information)

Entropies, divergences and mutual information estimated and watched, and coding stepped: source coding (Huffman, arithmetic, Lempel-Ziv), channel capacity, rate-distortion and error-correcting codes, as sessions over `information` and `dists`.

## Capabilities

- Measures: entropy, cross-entropy, KL and f-divergences, mutual information of named distributions and of samples (plug-in, k-nearest-neighbour, kernel estimators) converging with the sample size; options: the distributions, the estimator.
- Source coding: Huffman tree construction step by step, arithmetic coding's interval narrowing, Lempel-Ziv's dictionary growing on a text; options: the source, the alphabet.
- Channels: capacity by the Blahut-Arimoto iteration, binary symmetric and erasure channels, a simple code's error rate against noise.
- Rate-distortion: the curve traced by the same iteration.
- Minimum description length and the information bottleneck as objectives (with `representation-learning`).

## Examples to make

- The Huffman tree growing from leaf frequencies, the code lengths against entropy.
- Arithmetic coding one symbol at a time, the interval and the bits so far.
- Mutual-information estimators against the truth as n grows, on a Gaussian pair with correlation as the knob.
- Blahut-Arimoto converging on a channel matrix, the input distribution updating.

## Rendering

- Trees (`tree` view), interval strips, curves against truth, a heatmap of a channel matrix.
- Missing: a code table panel (symbol, probability, code, length).

## Shared

- `shared/`: the source slot (a named distribution, a text, a sample), the estimator-against-truth study base shared with `probability`.

## Problems

- `information/` moves here. Little of it is iterative; most pages are a one-shot computation with a knob, which the session player handles as a one-step run.
