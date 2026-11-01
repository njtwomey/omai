# Representation learning (part 5, learning paradigms)

Representations learned and evaluated: self-supervision by contrastive objectives (SimCLR-style, triplet and margin losses, two-tower CLIP at toy scale), the non-contrastive methods (BYOL, SimSiam, Barlow Twins, VICReg) and what stops their collapse, masked prediction (masked autoencoders), embeddings of words (count-based vectors, skip-gram and CBOW with negative sampling, GloVe, fastText subwords) and of graphs (random-walk embeddings, spectral embeddings, contrastive graph encoders, knowledge-graph embeddings), and the evaluation of representations (linear probes and nearest neighbours, alignment and uniformity, representation similarity, mutual-information bounds), as sessions over `nn`, `losses/contrastive`, `losses/large-output`, `text`, `graph`, `stochastic/random-walks`, `kernels/matrices` and `information`, each training watched on data small enough to run in the page. Text and sequence representations are `natural-language-processing`'s, multimodal ones `multimodal-learning`'s, foundation models as feature extractors `large-language-models`' and `transfer-learning`'s, autoencoders `generative-models`'; each has a `see` row.

## Structure

Every method here trains a network or an embedding table against an objective on fixed data, which is the torch world, so they live under `train/representation/` (section 3 of `methods/README.md`), one folder per family of objective or of structured data; the evaluation of a representation is done to a trained model and is an `analyse/evaluate/` candidate, kept here as `evaluation/` because every page of this chapter runs a probe at a schedule. The trainer base is `methods/shared/bases/`; the augmentation slot, the encoder registry and the probe harness are the chapter's shared pieces.

```
train/representation/
  contrastive/        SimCLR-style NT-Xent and InfoNCE, triplet and margin losses with their mining, a two-tower model at toy scale
  non-contrastive/    BYOL, SimSiam, Barlow Twins and VICReg: no negatives, the collapse and what prevents it
  masked/             masked autoencoders on small images, masked prediction on sequences
  words/              count-based vectors (PPMI, SVD), skip-gram and CBOW with negative sampling, GloVe, fastText subwords; analogies
  graphs/             random-walk embeddings (DeepWalk, node2vec), spectral node embeddings, contrastive graph encoders, knowledge-graph embeddings
  autoencoders/       see generative-models
  evaluation/         linear probes and nearest neighbours, alignment and uniformity, representation similarity (CKA), mutual-information bounds
shared/               the augmentation slot, the encoder registry, the probe harness, the pair samplers, the embedding metrics (below)
```

Nothing is built; the old plan's `embedding/` folder (word2vec, fastText, GloVe, graph, knowledge-graph, evaluation) maps onto `words/`, `graphs/` and `evaluation/` above.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of every trainer is one minibatch (or one epoch at the page's choice); every method on the trainer base logs `training/loss` s, `training/gradNorm` s and `model/params` p, and every method of this chapter logs `model/embedding` p (the training rows mapped at the viewed step; the `embeddingSteps` projection reads it over the tape), `model/neighbours` p (the nearest rows of a query in the embedding) and `eval/probeAccuracy` s (a linear probe at a schedule, from `shared/probes.ts`); the tables name what a method adds. The status is `built <path>`, `partly` or empty; library readiness is in the needs column (a function named there exists; `TODO:` names what is missing and its proposed home).

### train/representation/contrastive/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `simclr` | Two augmentations of each example through one encoder and a projection head, NT-Xent with the temperature as the knob and the batch as the negatives (the batch size as the other knob: the lesson that it matters), on planted 2-D clusters or small images: the embedding's clusters forming step by step, alignment and uniformity on the tape, the probe accuracy rising before the loss settles; the projection head's output against the encoder's (which to probe); the augmentation strength as the knob that decides what is invariant. | `training/alignment` s, `training/uniformity` s, `training/temperature` s, `model/similarityMatrix` p (a batch's), `model/augmentedPairs` p, `study/embeddingSteps` proj | `losses/contrastive.ts`: `ntXent`, `infoNce`, `supervisedContrastive`; `nn/module/containers.ts`: `mlp`, `Sequential`; `nn/blocks/vision.ts`: `BasicBlock`; `nn/training/fit.ts`: `fitAlgorithm`; `signal/image/geometry.ts`: `shift`, `rotate`, `zoom`, `resize`; `nn/functional/regularisation.ts`: `gaussianNoise`, `cutout`; `exposition/embeddings.ts`: `embeddingSteps`; `explain/concepts.ts`: `linearProbe`; `datasets/images.ts`: `shapes`, `digitGlyphs`, `fashionMnistTest`; TODO alignment and uniformity (a few lines in `shared/metrics.ts`) and the augmentation slot | |
| `triplet-and-margin` | The triplet margin loss with the mining as the mode (random, semi-hard, hardest; the fraction of active triplets falling on the tape), the margin as the knob, an anchor with its positive and negative drawn in the embedding; the margin-softmax heads (ArcFace, CosFace) as the mode that needs no triplets, the angular margin drawn on the sphere. | `training/activeTriplets` s, `training/margin` s, `model/triplet` p (an anchor, its positive and negative), `model/sphere` p | `losses/contrastive.ts`: `tripletMargin`, `contrastive`, `cosineEmbedding`, `hingeEmbedding`, `arcFace`, `cosFace`, `arcFaceLogits`; `numerics/spatial/brute.ts`: `kNearest` (the mining); TODO the miners in `shared/pairs.ts` | |
| `two-tower` | CLIP at toy scale: two encoders over two views of one synthetic object (an image of a shape and a description from a tiny grammar), the symmetric InfoNCE, the image-text similarity matrix as a heatmap converging to the diagonal, zero-shot classification by prompts at the viewed step; the real thing is `multimodal-learning`. | `model/similarityMatrix` p, `eval/zeroShotAccuracy` s, `model/textEmbedding` p | `losses/contrastive.ts`: `clipLoss`; `nn/module/layers.ts`: `Embedding`; `nn/module/containers.ts`: `mlp`; `datasets/images.ts`: `shapes`, `geometricScenes`; TODO the tiny caption grammar in `shared/` | |

### train/representation/non-contrastive/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `non-contrastive` | The method as the mode: BYOL (an exponential-moving-average target network and a predictor), SimSiam (the stop-gradient alone), Barlow Twins (the cross-correlation of the two views driven to the identity), VICReg (variance, invariance and covariance terms); the collapse when the asymmetry is removed (a knob: no stop-gradient, no predictor) as the picture, the embedding's variance per dimension on the tape going to zero; the same probe accuracy as SimCLR without negatives. | `training/embeddingVariance` s, `training/crossCorrelation` p (Barlow Twins), `training/collapsed` e, `study/embeddingSteps` proj | `optim/first-order.ts`: `ema` (the target network); `nn/training/fit.ts`: `fitAlgorithm`; `nn/module/containers.ts`: `mlp`; `losses/regression.ts`: `mse`; TODO the Barlow Twins and VICReg losses in `losses/contrastive.ts` (a few lines) and the stop-gradient in the loop | |

### train/representation/masked/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `masked-autoencoder` | Patches of a small image masked at random (the ratio as the knob), a transformer encoder on the visible patches, a light decoder reconstructing the rest: the reconstruction at the viewed step as a picture (what the model fills in), the loss on masked patches only, the probe accuracy against the mask ratio (a sweep); masked token prediction on sequences as the mode (BERT-style at toy scale; the full story is `transformers`). | `training/maskRatio` s, `model/reconstruction` p (masked, reconstructed, original), `model/attention` p, `study/ratioCurve` proj | `nn/module/vision.ts`: `PatchEmbedding`; `nn/module/transformer.ts`: `TransformerEncoder`; `nn/blocks/vision.ts`: `toTokens`; `losses/regression.ts`: `mse`; `datasets/images.ts`: `digitGlyphs`, `fashionMnistTest`; `datasets/sequence-tasks.ts`: `copyTask` (the sequence mode's data); TODO the masking as an augmentation of `shared/augmentations.ts` | |

### train/representation/words/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `count-based-vectors` | The co-occurrence matrix (the window as the knob), PPMI with its context smoothing, the SVD (the dimension as the knob): nearest words and analogies on a tiny corpus, the step being the window or the rank; LSA as the document-level mode; the baseline every neural embedding is measured against. | `model/cooccurrence` p, `model/topNeighbours` p (for a word), `model/analogy` p (a : b :: c : ?), `eval/analogyAccuracy` s | `text/representations.ts`: `cooccurrence`, `pmi`, `svdWordVectors`, `lsa`, `nearestWords`, `analogy`; `text/tokenise.ts`: `wordTokenize`; `text/vocabulary.ts`: `buildVocabulary`; `datasets/corpus.ts`: `nurseryRhymes`, `tinyShakespeare` | |
| `skip-gram` | Skip-gram and CBOW as the input mode, negative sampling (k as the knob, the unigram-to-the-0.75 table) against hierarchical softmax (the Huffman tree, the path of a word) as the output mode, subsampling of frequent words, one batch of (centre, context) pairs per step: the embedding moving, the analogy vector drawn as it forms, the nearest words of a query at the viewed step; GloVe (weighted least squares on the log co-occurrence) as the third mode on the same corpus; fastText's subword n-grams as an input mode (vectors for unseen words, the n-grams of a word as a probe). | `training/negatives` s, `model/huffmanPath` p (the hierarchical mode), `model/analogy` p, `model/topNeighbours` p, `model/subwords` p (fastText), `eval/analogyAccuracy` s, `study/embeddingSteps` proj | `losses/large-output.ts`: `hierarchicalSoftmax`, `sampledSoftmax`, `huffmanCoding`; `nn/module/layers.ts`: `Embedding`, `EmbeddingBag`; `text/shingles.ts`: `skipgrams`, `fastTextNgrams`, `fastTextBuckets`; `text/vocabulary.ts`: `buildVocabulary`; `text/representations.ts`: `cooccurrence` (the GloVe mode), `analogy`; `nn/training/fit.ts`: `fitAlgorithm`; TODO `lib/embedding/word2vec.ts` (the pair sampler with the unigram table and subsampling, the training loop as an `Algorithm`) and `glove.ts` | |

### train/representation/graphs/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `random-walk-embeddings` | DeepWalk's uniform walks and node2vec's biased ones (p and q as knobs: breadth-like against depth-like neighbourhoods) sampled one batch per step and fed to skip-gram: the walks drawn on the graph as they are sampled, the embedding against the communities (the planted blocks of a stochastic block model, or Louvain's partition as the reference), link prediction on held-out edges as the evaluation; p and q changing which nodes end up near. | `walk/batch` p (the walks drawn), `model/communityAgreement` s, `eval/linkAuc` s, `model/graph` p (nodes placed by the embedding), `study/embeddingSteps` proj | `stochastic/random-walks.ts`: `graphRandomWalk`; `graph/community.ts`: `louvain`, `modularity`; `graph/generators.ts`: `stochasticBlockModel`; `datasets/graphs.ts`: `karateClub`, `lesMiserables`, `contextualSbm`; `metrics/curves.ts`: `rocAucScore`; the skip-gram above; TODO the second-order biased walk of node2vec beside `graphRandomWalk` | |
| `spectral-node-embedding` | The Laplacian's eigenvectors as node coordinates (as `fit/reduce/manifold/spectral-embedding`, here on a graph), the positional encodings GNNs use (Laplacian and random-walk), NetMF as the closed form of DeepWalk (the matrix it factorises); the same communities found without training. | `solver/eigenvalues` t!, `model/graph` p, `model/communityAgreement` s | `graph/spectral.ts`: `spectralEmbedding`, `spectralEmbeddingAlgorithm`; `graph/matrices.ts`: `laplacianPositionalEncoding`, `randomWalkPositionalEncoding`, `transitionMatrix`; `numerics/linalg/svd.ts`: `svd`; TODO NetMF (a few lines) | |
| `graph-contrastive` | A graph encoder (GCN, GraphSAGE, GIN as the mode) trained with a contrastive objective between two augmented views of the graph (edge dropping and feature masking as knobs) or by local-global mutual information (DGI) as the other mode: the node embedding against the communities, the probe accuracy on node labels; the architectures themselves are `graph-neural-networks`'. | `training/views` p, `model/graph` p, `model/communityAgreement` s | `nn/functional/graph.ts`: `gcnConv`, `sageConv`, `ginConv`, `globalPool`; `nn/module/graph.ts`: `GCNConv`, `SAGEConv`, `GINConv`; `losses/contrastive.ts`: `infoNce`; `datasets/graphs.ts`: `contextualSbm`; TODO the graph augmentations in `shared/augmentations.ts` | |
| `knowledge-graph-embedding` | TransE (head plus relation near tail, a margin loss against corrupted triples), DistMult, ComplEx and RotatE as modes on the kinship triples: entities and relations drawn as points and arrows, the ranking metrics (mean reciprocal rank, hits at k) on held-out triples in the filtered setting, the relations a model cannot express (symmetry for TransE, antisymmetry for DistMult) as the comparison. | `training/corrupted` s, `model/relationArrows` p, `eval/mrr` s, `eval/hitsAtK` s, `model/triplesRanked` p (for a query) | `datasets/graphs.ts`: `kinshipTriples`; `metrics/ranking.ts`: `meanReciprocalRank`, `hitRate`; `losses/contrastive.ts`: `tripletMargin`; `nn/module/layers.ts`: `Embedding`; TODO `lib/embedding/knowledge-graph.ts` (the scoring functions, the corruption sampler, filtered ranking) | |

### train/representation/autoencoders/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `autoencoder-representation` | The bottleneck as the knob, the latent as the representation, the VAE's posterior as the probabilistic one. | | | see `generative-models` |

### train/representation/evaluation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `linear-probe` | A linear classifier on frozen features at a schedule of any trainer above (the probe accuracy on the tape, rising before the training loss settles), k-nearest-neighbour accuracy as the cheaper mode, the comparison of two encoders by their probes at the same step; which layer to probe as the knob. | `eval/probeAccuracy` s, `eval/knnAccuracy` s, `model/probeWeights` p, `study/byLayer` proj | `explain/concepts.ts`: `linearProbe`; `numerics/spatial/brute.ts`: `kNearest`; `learn/model-selection/cross-validation.ts`: `crossValScore` | |
| `alignment-and-uniformity` | The two quantities on the hypersphere as the temperature changes (a sweep): alignment (positives close) against uniformity (everything spread), their trade-off, the embedding on the circle as the picture; the two quantities as probes of any contrastive trainer. | `sweep/temperature` s, `sweep/alignment` s, `sweep/uniformity` s, `model/sphere` p, `study/tradeOff` proj | TODO `alignment`, `uniformity` in `shared/metrics.ts` (a few lines over `kernels/kernel.ts`: `gram`) | |
| `representation-similarity` | Centred kernel alignment (linear and kernel) and HSIC between two encoders or two layers of one network, the layer-against-layer similarity matrix as a heatmap at the viewed step (which layers change during training), SVCCA as the mode. | `model/ckaMatrix` p, `training/ckaToInit` s (per layer) | `kernels/matrices.ts`: `kernelAlignment`, `centerKernel`, `hsic`; TODO SVCCA over `fit/reduce/cross/cca` (`unsupervised-learning`) | |
| `mutual-information-bounds` | InfoNCE as a lower bound on the mutual information between two views (its log-batch ceiling as the lesson), MINE and NWJ as modes, on a pair of variables with a known mutual information (the Gaussian case): the bound against the truth as the critic trains, the variance of each. | `training/bound` s, `training/truth` s, `training/gap` s, `study/byBatch` proj | `information/estimators.ts`: `infoNce`, `mineBound`, `nwjBound`; `information/neighbour-estimators.ts`: `ksgMutualInformation` (the non-parametric check) | |

### cross-chapter

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `text-and-sequence-representations` | Contextual representations from language models, sentence embeddings. | | | see `natural-language-processing` and `sequence-models` |
| `multimodal-representations` | CLIP and its kin at scale, cross-modal retrieval. | | | see `multimodal-learning` |
| `foundation-models-as-features` | Frozen features from a pretrained model, fine-tuning against probing. | | | see `large-language-models` and `transfer-learning` |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `augmentations.ts` | The augmentation slot: noise and feature dropout for points, shifts, rotations, crops, flips and cutout for images (`signal/image/geometry.ts`, `nn/functional/regularisation.ts`), edge dropping and feature masking for graphs, word dropping and masking for text, each with a strength knob; shared with `semi-supervised-learning` and `computer-vision` | `contrastive/`, `non-contrastive/`, `masked/`, `graphs/` |
| `encoders.ts` | The encoder registry: an MLP, a small convolutional encoder (`nn/blocks/vision.ts`), a graph encoder (`nn/module/graph.ts`), a transformer encoder (`nn/module/transformer.ts`), each with its projection head and its space; shared with `feedforward-networks` | every trainer |
| `probes.ts` | The probe harness: a linear probe or k-nearest neighbours on frozen features at a schedule, `eval/probeAccuracy` and `eval/knnAccuracy` on the tape, the layer as a knob; shared with `feedforward-networks` and `transfer-learning` | every trainer |
| `pairs.ts` | The pair samplers: positives by augmentation or by label and negatives from the batch, the triplet miners, (centre, context) pairs with the unigram table and subsampling, walk windows, corrupted triples | `contrastive/`, `words/`, `graphs/` |
| `metrics.ts` | Embedding metrics at the viewed step: alignment and uniformity, the embedding's per-dimension variance (collapse), CKA, community agreement of a node embedding, analogy accuracy | every trainer, `evaluation/` |

## Examples to make

- SimCLR on a toy image set: the embedding's clusters forming, the linear-probe accuracy rising; the temperature and the batch size as knobs.
- Skip-gram on a tiny corpus: the words' positions settling, an analogy's vector drawn; the count-based vectors beside it.
- node2vec on a small graph: the walks as they are sampled, the embedding and the communities; p and q as knobs.
- BYOL with and without the stop-gradient: the collapse on the tape.
- Alignment against uniformity as the temperature changes.

## Rendering

- Embedding steps with labels, neighbour lists, curves, graphs with node positions from the embedding, tokens (`tokens`), image grids for reconstructions.
- Missing: an analogy-arrow overlay on an embedding; an image grid for the neighbours of a query (`image`); a similarity-matrix heatmap that follows the cursor; a sphere panel for the hypersphere pictures.

## Problems

- `embedding/` moves here. Training on the main thread at a watchable rate needs the worker producer (problem 7 of `methods/README.md`); until then the data are tiny (planted clusters, glyphs, nursery rhymes, the karate club), which is also what makes the pictures legible.
- The library has the losses, the layers, the count-based vectors and the walks; the gaps are the samplers and loops (`lib/embedding/`: word2vec, GloVe, the knowledge-graph scorers), the node2vec bias, and a few short losses and metrics (Barlow Twins, VICReg, alignment, uniformity).
- `evaluation/` is done to a trained model and may belong to `analyse/evaluate/`; it is kept here because the probe runs at a schedule inside every trainer of the chapter, and the standalone studies (CKA across layers, the mutual-information bounds) are the ones that would move.
- Whether the trainer base and the probe harness are this chapter's or `feedforward-networks`' is the same question for every chapter of the torch world; `methods/shared/bases/` holds the base and this chapter's `shared/probes.ts` the harness until a second chapter needs it.
