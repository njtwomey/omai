# Information theory (part 2, signals, systems and information)

Entropies, divergences and mutual information computed and estimated, and coding stepped: the measures of named distributions, their estimators from samples converging with n, directed information and complexity; source coding (Huffman, arithmetic and range coding, Lempel-Ziv, integer codes, deflate), channel capacity by Blahut-Arimoto, error-correcting codes over a noisy channel (block, convolutional, LDPC, Reed-Solomon), rate-distortion, and the information bottleneck and minimum description length as objectives, as sessions over `information`, `dists` and `learn/model-selection`.

## Structure

Little of the chapter fits a model: a measure is computed or estimated, a code is run, a channel is simulated, an iteration (Blahut-Arimoto, belief propagation, Huffman's merges, Viterbi's stages) is watched. That is `simulate/` (section 3 of `methods/README.md`), beside `simulate/stochastic/` where the probability chapter's estimators and processes go; the section's list of `simulate/` children has no `information/`, so this chapter asks for one (Problems). The estimator-against-truth base is shared with `probability`.

```
simulate/information/
  measures/           entropies, divergences and mutual information of named distributions; the estimators from samples; directed information; complexity and entropy rate
  source-coding/      Huffman and Shannon-Fano, arithmetic and range coding, Lempel-Ziv and the Burrows-Wheeler transform, integer codes, deflate
  channels/           channel capacity by Blahut-Arimoto, the binary channels, the Gaussian channel and water-filling
  error-correction/   block codes with syndrome decoding, convolutional codes by Viterbi, LDPC by belief propagation, Reed-Solomon, the bounds
  rate-distortion/    the curve by the Blahut-Arimoto iteration, the Gaussian and binary truths, reverse water-filling
  objectives/         the information bottleneck, minimum description length
  shared/             the source slot, the channel slot, the estimator-against-truth base, the code-table and trellis records (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of an estimator is one batch of samples; of a coder, one symbol, phrase or merge; of a channel simulation, one block; of an iteration, one iteration.

### measures/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `entropy-and-divergences` | Named distributions with their knobs (a Bernoulli's p, a Gaussian pair's correlation, a mixture's separation): entropy, cross-entropy, KL in both directions (its asymmetry), Jensen-Shannon, total variation, Rényi's α as a knob, the f-divergence family with f as the mode; mutual information as the KL from the product; the chain rule's terms stacked. | `measure/value` s, `measure/knob` s, `measure/terms` t (the chain-rule decomposition), `distribution/picture` p, `study/curve` proj (the measure against the knob) | `information/discrete.ts`: `entropy`, `binaryEntropy`, `jointEntropy`, `conditionalEntropy`, `crossEntropy`, `relativeEntropy`, `jensenShannonDivergence`, `totalVariation`, `mutualInformation`, `conditionalMutualInformation`, `interactionInformation`; `information/divergences.ts`: `fDivergence`, `renyiDivergence`, `renyiEntropy`, `alphaDivergence`, `bregmanDivergence`; `dists/divergences.ts`: `klDivergence`, `hellinger`, `wasserstein2Gaussian` | |
| `estimators` | Mutual information of a Gaussian pair (its correlation as the knob, the truth in closed form) estimated as n grows one batch per step, the estimator as a mode: plug-in on a histogram (its bins), KDE, Kozachenko-Leonenko and KSG (k as the knob), the mixed estimator for a discrete-continuous pair, the neural bounds (InfoNCE, NWJ, MINE) trained by the optimiser slot; the bias and variance of each against n. | `estimate/value` s, `estimate/error` s, `estimate/n` s, `estimate/k` s, `training/loss` s (the neural modes), `truth/value` p, `study/convergence` proj | `information/estimators.ts`: `plugInEntropy`, `plugInMutualInformation`, `kdeEntropy`, `kdeMutualInformation`, `infoNce`, `nwjBound`, `mineBound`; `information/neighbour-estimators.ts`: `kozachenkoLeonenkoEntropy`, `ksgMutualInformation`, `ksgConditionalMutualInformation`, `knnKlDivergence`, `mixedMutualInformation`; `nn/training/fit.ts`: `fitAlgorithm` for the bounds | |
| `directed-information` | Transfer entropy from one series to another with a planted coupling (its strength and lag as knobs), one batch per step, against the reverse direction and against Granger causality; the lag at which it peaks. | `estimate/transferEntropy` s, `estimate/reverse` s, `estimate/lag` s, `study/byLag` proj | `information/estimators.ts`: `transferEntropy`; `datasets/timeseries.ts`: `armaProcess` (a coupled pair TODO in `shared/sources.ts`) | |
| `complexity` | A text or a symbol stream: Lempel-Ziv complexity growing phrase by phrase, block entropies against the block length converging to the entropy rate (a Markov source's rate known), approximate and sample entropy of a series (regular against chaotic), the normalised compression distance clustering texts by what compresses them together. | `stream/position` s, `lz/phrases` s, `lz/complexity` s, `entropy/byBlock` t!, `entropy/rate` s, `truth/rate` p, `ncd/matrix` p | `information/complexity.ts`: `lempelZivComplexity`, `blockEntropy`, `entropyRate`, `markovEntropyRate`, `approximateEntropy`, `sampleEntropy`, `normalisedCompressionDistance`, `compressionDistanceMatrix`; `datasets/corpus.ts`: `nurseryRhymes`, `tinyShakespeare` | |

### source-coding/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `huffman` | The tree built one merge per step from the leaf frequencies (the two lightest joined, the queue drawn), the code lengths against −log p, the expected length between H and H + 1, Kraft's sum exactly one; Shannon-Fano and Shannon codes as modes losing a little; the canonical form; a text encoded and decoded with its bits counted. | `merge/step` s, `merge/pair` m, `tree/structure` p, `code/lengths` t!, `code/expectedLength` s, `code/entropy` s, `code/kraft` s, `code/table` p | `information/source-coding.ts`: `huffmanAlgorithm`, `huffmanTree`, `huffman`, `shannonFano`, `shannonCode`, `canonicalCode`, `kraftSum`, `expectedLength`, `isPrefixFree`, `prefixEncode`, `prefixDecode` | |
| `arithmetic-coding` | One symbol per step: the interval narrowing by the symbol's probability (a strip whose width falls as 2^(−bits)), the bits emitted so far against the ideal −Σ log p, the decoder walking the same intervals; an adaptive model (the frequency table updating) as a mode; range coding as the integer form. | `symbol/index` s, `interval/low` s, `interval/high` s, `interval/strip` p, `code/bits` s, `code/ideal` s, `model/frequencies` t! | `information/stream-coding.ts`: `arithmeticEncode`, `arithmeticDecode`, `rangeEncode`, `rangeDecode`, `frequencyTable` (whole; TODO a stepped `arithmeticEncoderAlgorithm`, one symbol per step with the interval) | |
| `dictionary-coding` | The algorithm as a mode (LZ77 with its window as the knob, LZ78, LZW): the dictionary growing on a text one phrase per step, the match drawn back into the window, the compression ratio against the text's entropy rate; the Burrows-Wheeler transform and move-to-front as the pre-processing that makes runs. | `phrase/index` s, `phrase/match` m (offset, length, next), `dictionary/size` s, `code/ratio` s, `code/bits` s, `bwt/matrix` p, `mtf/stream` t! | `information/dictionary-coding.ts`: `lz77Encode`, `lz78Encode`, `lzwEncode`, their decoders, `bwt`, `inverseBwt`, `moveToFront` (whole; TODO stepped forms, one phrase per step) | |
| `integer-codes` | Elias γ and δ, Golomb-Rice with its parameter chosen from the mean, run-length coding: code lengths against the integer, which code wins for which distribution (the source's parameter as the knob). | `code/lengths` t!, `code/expectedLength` s, `code/parameter` s, `study/byDistribution` proj | `information/integer-coding.ts`: `eliasEncode`, `eliasDecode`, `golombEncode`, `golombDecode`, `golombParameter`, `runLengthEncode` | |
| `deflate` | The real thing: LZ77 then Huffman on the literals and lengths, the blocks of a gzip stream, the ratio against the parts alone; round-tripped through inflate and checked by CRC. | `block/index` s, `block/ratio` s, `code/ratio` s, `code/crc` s, `code/literals` t! | `information/deflate.ts`: `deflate`, `inflate`, `gzipCompress`, `gzipDecompress`, `crc32` | |

### channels/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `channel-capacity` | Blahut-Arimoto one iteration per step on a channel matrix (the binary symmetric, erasure and Z channels with their parameter as the knob, or any matrix): the input distribution updating, the lower and upper bounds closing on the capacity, against the closed forms (1 − H(p), 1 − ε); the Gaussian channel's ½ log(1 + SNR) and water-filling over parallel channels with the power as the step. | `solver/input` t!, `solver/lower` s, `solver/upper` s, `solver/capacity` s, `truth/capacity` p, `channel/matrix` p, `waterfilling/levels` t! | `information/channels.ts`: `blahutArimotoAlgorithm`, `channelCapacity`, `channelMutualInformation`, `binarySymmetricChannel`, `binaryErasureChannel`, `zChannel`, `gaussianCapacity`, `waterFilling` | |

### error-correction/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `block-codes` | The code as a mode (repetition, Hamming, a linear code from its generator): bits through a BSC with its crossover as the knob, the syndrome decoded, the bit error rate after decoding against before, the minimum distance and the bounds (Hamming, Singleton, Gilbert-Varshamov, Plotkin) framing what rate is possible; one block per step, the running error rate. | `block/index` s, `block/syndrome` t, `block/corrected` s, `ber/before` s, `ber/after` s, `code/distance` s, `code/bounds` p | `information/error-correction.ts`: `repetitionCode`, `hammingCode`, `linearCode`, `linearEncode`, `syndrome`, `syndromeDecoder`, `syndromeDecode`, `minimumDistance`, `hammingBound`, `singletonBound`, `gilbertVarshamovBound`, `plotkinBound`; `information/channels.ts`: `binarySymmetricChannel` | |
| `convolutional-codes` | The encoder's shift register one input bit per step and the trellis it draws; Viterbi decoding one trellis stage per step: the path metrics, the survivors, the traceback; the error rate against the uncoded channel. | `encoder/state` s, `trellis/metrics` t!, `trellis/survivors` t!, `decode/path` p, `ber/after` s | `information/convolutional.ts`: `convolutionalCode`, `convolutionalEncode`, `viterbiAlgorithm`, `viterbiDecode` | |
| `ldpc` | A Gallager parity-check matrix as a bipartite graph; belief propagation one iteration per step: the log-likelihood ratios on the edges, the hard decision's syndrome going to zero (an event), the share of blocks decoded against the channel's noise (the waterfall); the message overlay on the graph as the probe. | `bp/iteration` s, `bp/llr` t!, `bp/syndromeWeight` s, `bp/decoded` e, `ber/after` s, `graph/messages` p | `information/ldpc.ts`: `gallagerParityCheck`, `ldpcCode`, `ldpcEncode`, `bpskLlr`, `bscLlr`, `beliefPropagationAlgorithm`, `ldpcDecode` | |
| `reed-solomon` | Symbols over a Galois field: a burst of errors (its length as the knob) corrected up to t, the syndromes, the error locator; the field's tables as a board. | `errors/planted` s, `decode/syndromes` t, `decode/corrected` s, `decode/failed` e, `field/table` p | `information/reed-solomon.ts`: `galoisField`, `reedSolomon`, `rsEncode`, `rsDecode` | |

### rate-distortion/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `rate-distortion` | The Blahut-Arimoto iteration for R(D) at one slope per step, the curve traced; the Gaussian source's ½ log(σ²/D) and the Bernoulli's H(p) − H(D) as the truths; reverse water-filling for a Gaussian vector with the distortion as the step; the quantiser of `signal-processing` placed against the curve. | `solver/rate` s, `solver/distortion` s, `solver/slope` s, `curve/points` t!, `truth/curve` p, `waterfilling/components` t! | `information/rate-distortion.ts`: `rateDistortionAlgorithm`, `rateDistortion`, `rateDistortionCurve`, `reverseWaterFilling`, `binaryRateDistortion`, `gaussianRateDistortion` | |

### objectives/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `information-bottleneck` | A joint distribution over (X, Y) with a planted structure: the bottleneck's iterative algorithm at one β per step (the compression I(T; X) against the relevance I(T; Y), the curve traced, the clusters of X merging as β falls); the deterministic agglomerative mode. | `solver/beta` s, `solver/compression` s, `solver/relevance` s, `solver/encoder` t!, `curve/points` t!, `encoder/picture` p | `information/discrete.ts`: `mutualInformation`, `relativeEntropy`; TODO `information/bottleneck.ts`: `informationBottleneckAlgorithm` (Tishby's self-consistent updates), `agglomerativeBottleneck` | |
| `minimum-description-length` | Model selection as compression: for a polynomial's degree or a histogram's bins as the step, the code length of the model plus the data given it, its minimum against BIC's asymptotic form and against held-out error; the two-part code as a stacked bar. | `model/complexity` s, `code/modelBits` s, `code/dataBits` s, `code/total` s, `study/curve` proj | `learn/model-selection/criteria.ts`: `bic`, `aic`, `gaussianLogLikelihood`; `information/source-coding.ts`: `expectedLength`; TODO `information/mdl.ts`: `twoPartCodeLength`, `nmlCodeLength` (the normalised maximum likelihood for small families) | |
| `see representation-learning` | The bottleneck as a training objective of a network (the information plane over epochs) is the representation chapter's. | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `sources.ts` | The source slot: named distributions from `dists` with their entropies and divergences in closed form, Markov sources with their entropy rate, texts (`nurseryRhymes`, `tinyShakespeare`) with their symbol statistics, a Gaussian pair with its mutual information, coupled series for transfer entropy; the sample as the step's batch | `measures/`, `source-coding/`, `objectives/` |
| `channels.ts` | The channel slot: BSC, BEC, Z, a general matrix with knobs, the AWGN channel with its SNR; the capacities known; a bit stream pushed through with its errors marked | `channels/`, `error-correction/`, `rate-distortion/` |
| `estimates.ts` | The estimator-against-truth base: n growing by a schedule, the estimate with its error and a band over repeats, the convergence projection; shared with `probability` | `measures/estimators`, `measures/directed-information` |
| `records.ts` | The code-table record (symbol, probability, code, length; sortable), the interval strip of arithmetic coding, the trellis board, the parity-check graph with messages (a `Structure` as `probabilistic-inference`'s `messages.ts`) | `source-coding/`, `error-correction/` |

## Examples to make

- The Huffman tree growing from the leaf frequencies, the code lengths against −log p.
- Arithmetic coding one symbol at a time, the interval and the bits so far.
- Mutual-information estimators against the truth as n grows, on a Gaussian pair with correlation as the knob.
- Blahut-Arimoto converging on a channel matrix, the input distribution updating; the same iteration tracing R(D).
- LDPC belief propagation on a small code: the messages on the graph, the syndrome going to zero.

## Rendering

- Trees (`tree` view), curves against truth, heatmaps of a channel matrix, structures with messages.
- Missing: a code-table panel (symbol, probability, code, length); an interval strip for arithmetic coding; a trellis board.

## Problems

- `information/` of the old plan moves here. Section 3's `simulate/` lists no `information/`; this chapter asks for it as a sibling of `stochastic/`. The coders on text could equally live under `process/text/coding/`: decide when the first page is built.
- Most of the chapter is a one-shot computation with a knob or a sweep over n; the iterations (Blahut-Arimoto, belief propagation, Huffman's merges, Viterbi's stages) are the sessions proper. The coders run whole: stepped forms of the arithmetic and dictionary coders are the library work; the bottleneck and MDL are missing.
- The neural estimators train a critic: they take the trainer base of `train/` and the optimiser slot, and are the only rows here that cost more than a second.
