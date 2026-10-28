# Representation learning (part 5, learning paradigms)

Representations learned and evaluated: self-supervision (contrastive, masked and predictive objectives), embeddings of structured data (words, graphs, knowledge graphs), text and sequence representations, multimodal representations, foundation models as feature extractors, and the evaluation of representations (linear probes, alignment and uniformity), as sessions over `nn`, `losses`, `graph` and `text`.

## Capabilities

- Contrastive training on 2-D or small-image data: the embedding moving step by step, alignment and uniformity on the tape; options: the objective (InfoNCE, triplet, SimCLR-style), the augmentations, the temperature.
- Word embeddings: skip-gram with negative sampling trained on a small corpus, the analogy arithmetic as a probe; GloVe's co-occurrence factorised.
- Graph embeddings: DeepWalk and node2vec's walks and their skip-gram, spectral embeddings; knowledge-graph embeddings (TransE) with triples scored.
- Autoencoders as representation learners (with `generative-models`): the bottleneck as the knob.
- Evaluation: a linear probe fitted on frozen features at each step, its accuracy on the tape; nearest neighbours in the embedding as a probe.

## Examples to make

- SimCLR on a toy image set: the embedding's clusters forming, the linear-probe accuracy rising.
- Skip-gram on a tiny corpus: the words' positions settling, an analogy's vector drawn.
- node2vec on a small graph: the walks as they are sampled, the embedding and the communities.
- Alignment against uniformity as the temperature changes.

## Rendering

- Embedding steps with labels, neighbour lists, curves, graphs with node positions from the embedding, tokens (`tokens`).
- Missing: an analogy-arrow overlay on an embedding; an image grid for neighbours of a query (`image`).

## Shared

- `shared/`: the trainer base (modules, optimiser, batches) shared with every part-6 chapter; the probe-evaluation harness (a linear probe at a schedule) shared with `feedforward-networks`.

## Problems

- `embedding/` moves here. Training on the main thread at a watchable rate needs the worker producer; until then the data are tiny.
