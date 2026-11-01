# Graph neural networks (part 6, neural architectures; with geometric deep learning)

Learning on graphs: message-passing layers (GCN, GraphSAGE, GAT, GIN, ChebNet) stepped on a small graph, their spectral view, the graph tasks (node classification, link prediction, graph classification and regression), expressiveness (the Weisfeiler-Leman test, positional encodings), depth (over-smoothing, over-squashing), scaling (sampling, precomputation, partitions) and relational graphs, and the geometric view (symmetry and equivariance on grids, sets and graphs, graph transformers), as sessions over `nn`, `graph` and the trainer base, on the karate club, planted partitions, random-graph families and small molecules.

## Structure

Problem methods of the torch world (section 7.2 of `methods/README.md`) under `train/graphs/`, one folder per group; the stepped layer and the Weisfeiler-Leman test are probe-only methods of the same tree. Geometric deep learning is folded in as `geometric/`: the symmetry blueprint (grids, sets, graphs) with the group-equivariant image instance owned by `convolutional-networks` and the set models here. Graph algorithms watched (traversal, paths, centrality) are `mathematics`' `process/graphs/`; node embeddings by random walks are `representation-learning`'s.

```
train/graphs/
  layers/           message passing stepped, the layer as a mode, the spectral view
  tasks/            node classification, link prediction, graph classification and regression
  expressivity/     Weisfeiler-Leman, GIN against GCN on indistinguishable graphs, positional encodings, over-smoothing, over-squashing
  scaling/          neighbour sampling, simplified propagation, partition training, relational graphs
  geometric/        the symmetry blueprint, set models, graph transformers, equivariance on graphs
  shared/           the graph slot with truth, the message probes, edge splits, layouts (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Trained rows step one optimiser update on the trainer base (`train/loss`, `train/gradNorm`, `train/lr`, `eval/<metric>`, `model/params` as checkpoints; `TODO: methods/shared/bases/trainer.ts` over `nn/training/fit.ts`: `fitAlgorithm`); the node embeddings at the viewed step are `embedding` t! and drawn over the graph; messages on edges are `messages/sent` m, the record `probabilistic-inference` uses for belief propagation. Nothing in this chapter is built.

### layers/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `message-passing` | One layer stepped on a small graph: the messages on the edges (one node's incoming set lit), the aggregation at the node (sum, mean, max as modes: what each loses), the update; self-loops and the symmetric normalisation of GCN as knobs; the layer as a matrix product (Â X W) as the other view. A probe-only page that explains every row below. | `messages/sent` m, `node/aggregate` t, `embedding` t!, `graph/structure` p (with the message overlay), `layer/matrix` p | `nn/functional/graph.ts`: `messagePassing`, `aggregate`, `edgeList`, `addSelfLoops`, `gcnNorm`; `graph/matrices.ts`: `normalisedAdjacency`; `datasets/graphs.ts`: `karateClub` | |
| `graph-layer` | The layer as the mode (GCN, GraphSAGE with its aggregator, GAT with heads, GIN with its ε, ChebNet with its order) on node classification: the node embeddings moving over the graph's drawing epoch by epoch, the attention weights on a node's edges at the viewed step (GAT), the parameter counts; the same page as `tasks/node-classification` with the layer as the thing compared. | `train/loss` s, `eval/accuracy` s, `embedding` t!, `attention/edges` t! (GAT), `model/params` p, `model/predictions` p | `nn/functional/graph.ts`: `gcnConv`, `sageConv`, `gatConv` (returns its attention), `ginConv`; `nn/module/graph.ts`: `GCNConv`, `SAGEConv`, `GATConv`, `GINConv`; `graph/spectral.ts`: `chebyshevFilter`, `chebyshevFilterCoefficients` (ChebNet) | |
| `spectral-view` | A graph signal in the Laplacian's eigenbasis (the graph Fourier transform), low-pass filtering as what GCN's propagation does (the filter's response against the eigenvalues drawn), heat diffusion on the graph one step per step, the Chebyshev approximation of a filter against the exact one as the order grows (the knob). | `signal/spectrum` t!, `filter/response` p, `diffusion/signal` t!, `filter/error` s, `graph/structure` p | `graph/spectral.ts`: `graphFourierTransform`, `inverseGraphFourierTransform`, `spectralFilter`, `heatKernel`, `heatDiffusion`, `chebyshevFilter`, `propagate`; `graph/matrices.ts`: `laplacian`, `normalisedLaplacian` | |

### tasks/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `node-classification` | Semi-supervised: a few labelled nodes per class (the knob), the rest predicted by propagation; the embedding separating the communities over the graph's drawing, the accuracy on the unlabelled nodes, the feature-only MLP and label propagation as the baselines it must beat; on the contextual stochastic block model the signal-to-noise of features against structure (knobs) decides which wins. | `train/loss` s, `eval/accuracy` s, `embedding` t!, `model/predictions` p (node colours), `model/params` p, `study/labelCurve` proj | `datasets/graphs.ts`: `karateClub`, `contextualSbm`; `nn/module/graph.ts`: `GCNConv`, `GATConv`; `graph/community.ts`: `labelPropagation` (the baseline); `nn/module/containers.ts`: `mlp` | |
| `link-prediction` | Edges held out (the split as a probe), a graph autoencoder scoring pairs by the inner product of embeddings against negatives sampled at random; the variational mode with its KL; the ROC curve on held-out edges at the viewed step, the predicted edges drawn on the graph; the decoder as the mode (inner product, an MLP on pairs, DistMult on typed triples). | `train/loss` s, `train/kl` s, `eval/auc` s, `eval/averagePrecision` s, `embedding` t!, `model/scores` p (over held-out pairs), `graph/split` p | `nn/module/graph.ts`: `GCNConv`; `losses/classification.ts`: `binaryCrossEntropyWithLogits`; `losses/generative.ts`: `gaussianKl` (VGAE); `metrics/curves.ts`: `rocAucScore`; `metrics/precision-recall.ts`: `averagePrecisionScore`; the edge split and negative sampling in `shared/splits.ts`; `datasets/graphs.ts`: `kinshipTriples` (typed) | |
| `graph-classification` | Whole graphs labelled by family (or molecules by a property): message passing then readout (sum, mean, max, top-k pooling as modes), GIN's sum readout as the one that counts; the accuracy on the tape, the graph embeddings as a projection coloured by family, which families confuse. | `train/loss` s, `eval/accuracy` s, `graphs/embedding` t!, `model/predictions` p, `model/confusion` p, `pool/kept` p (top-k) | `datasets/graphs.ts`: `graphFamilies`, `familyGraph`, `smallMolecules`; `nn/functional/graph.ts`: `globalPool`, `topKPool`, `ginConv`; `metrics/classification.ts`: `confusionMatrix` | |
| `graph-regression` | Molecules from SMILES with a computed target (molecular weight as the exact one, a planted function of structure as the hard one): the error on the tape, atom-level contributions as a probe (which atoms the readout weights), the residuals against size. | `train/loss` s, `eval/mae` s, `model/predictions` p, `model/atomContributions` p (over the molecule's drawing) | `datasets/graphs.ts`: `smallMolecules`; `losses/regression.ts`: `mae`, `mse`; `metrics/regression.ts`: `meanAbsoluteError` | |

### expressivity/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `weisfeiler-leman` | Colour refinement one round per step on two graphs (the colours as a board over both), stopping when the partition is stable; two regular graphs it cannot tell apart (the hash equal, the graphs not isomorphic, which the backtracking search proves); the number of rounds against the graph's diameter. | `wl/colours` t!, `wl/classes` s, `wl/stable` e, `wl/hash` m, `graph/board` p | `graph/isomorphism.ts`: `weisfeilerLehmanAlgorithm`, `wlHash`, `isIsomorphic`; `graph/generators.ts`: `cycleGraph`, `petersenGraph`, `completeBipartiteGraph` (the regular pairs); `exposition/boards.ts`: `boardSteps` | |
| `gin-against-gcn` | The two layers trained to tell apart pairs of graphs WL distinguishes (GIN can, mean aggregation cannot): the embeddings of the pair collapsing together or not, the injectivity of sum against mean and max on multisets drawn as the explanation; ε as the knob. | `train/loss` s, `eval/accuracy` s, `graphs/embedding` t!, `pair/distance` s, `aggregation/multisets` p | `nn/functional/graph.ts`: `ginConv`, `gcnConv`, `aggregate`; `datasets/graphs.ts`: `graphFamilies`; `graph/isomorphism.ts`: `wlHash` (the truth) | |
| `positional-encodings` | Laplacian eigenvectors and random-walk encodings as node features (modes) that break the symmetry message passing cannot: a task that needs position (which node is the centre of a cycle), accuracy with and without; the sign ambiguity of eigenvectors and the flip augmentation. | `train/loss` s, `eval/accuracy` s, `node/encoding` t!, `encoding/eigenvectors` p (over the graph) | `graph/matrices.ts`: `laplacianPositionalEncoding`, `randomWalkPositionalEncoding`; `graph/spectral.ts`: `fiedlerVector`, `spectralEmbedding` | |
| `over-smoothing` | The depth as the knob (one layer added per step of the study): the node embeddings converging to one point (the Dirichlet energy and the mean pairwise distance falling), the accuracy peaking at two or three layers; residual connections, jumping knowledge and normalisation as the fixes (modes); the heat-diffusion view (propagation is a low-pass filter, depth is time). | `layer/<i>/dirichletEnergy` s, `layer/<i>/meanDistance` s, `eval/accuracy` s, `embedding` t!, `study/depthCurve` proj | `graph/spectral.ts`: `dirichletEnergy`, `heatDiffusion`; `nn/module/containers.ts`: `Residual`; `nn/module/normalisation.ts`: `LayerNorm` | |
| `over-squashing` | A barbell graph with the target on one side and the signal on the other: the gradient of the output with respect to the far input vanishing through the bottleneck as the bridge lengthens (the knob), the influence per hop drawn on the graph; rewiring (an edge added across the bottleneck) and a global readout as the fixes. | `influence/perNode` t!, `influence/far` s, `bottleneck/length` s, `graph/structure` p (influence as node colour) | `graph/generators.ts`: `barbellGraph`, `pathGraph`; `autodiff/transforms.ts`: `jacobian` (output against input features); `graph/paths.ts`: `dijkstra` (hops); TODO `graph/spectral.ts`: `effectiveResistance` (the bottleneck measure) | |

### scaling/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `neighbour-sampling` | Minibatches of target nodes with k sampled neighbours per hop (GraphSAGE's recipe): the sampled subgraph at the viewed step drawn inside the full graph, the receptive field's size against the fan-out (knobs), the variance of the stochastic estimate against the full-batch gradient. | `train/loss` s, `batch/subgraph` p, `batch/size` s, `train/gradVariance` s, `eval/accuracy` s | `nn/functional/graph.ts`: `sageConv`; TODO `graph/`: `neighbourSample` (k-hop sampling of a target set with the induced subgraph; proposed in `graph/traversal.ts`) | |
| `simplified-propagation` | SGC and APPNP as modes: propagation precomputed (Â^k X, k as the knob, the features smoothed before any learning) then a linear model; personalised PageRank propagation against plain powers (the teleport probability keeps locality); the same accuracy as a GCN at a fraction of the cost on these graphs. | `train/loss` s, `eval/accuracy` s, `features/propagated` t!, `propagation/k` s, `study/kCurve` proj | `graph/spectral.ts`: `propagate`; `graph/matrices.ts`: `normalisedAdjacency`; `graph/centrality.ts`: `pagerankAlgorithm` (the personalised propagation); `nn/module/layers.ts`: `Linear` | |
| `partition-training` | The graph cut into clusters (Louvain or spectral bisection) and a batch as a cluster (Cluster-GCN's recipe): the edges lost between clusters (the share as the knob's cost), training on clusters against the full graph, the partition drawn. | `train/loss` s, `eval/accuracy` s, `partition/clusters` p, `partition/edgesCut` s | `graph/community.ts`: `louvain`, `spectralBisection`, `modularity` | |
| `relational-graphs` | Typed edges (the kinship triples): a relational layer with one weight per relation (basis decomposition as the knob against the parameter count), node classification and the typed link prediction of `tasks/`; the relation weights as a heat map. | `train/loss` s, `eval/accuracy` s, `relation/weights` t!, `embedding` t!, `model/params` p | `datasets/graphs.ts`: `kinshipTriples`; TODO `nn/functional/graph.ts`: `rgcnConv` (per-relation weights with basis decomposition, beside `gcnConv`) | |

### geometric/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `symmetry-blueprint` | One page, three domains: a signal on a grid, a set and a graph, each with its symmetry group (translations, permutations, graph automorphisms), the layer that respects it (convolution, Deep Sets, message passing) and a layer that does not (a dense one); the equivariance error of each under a transformed input at the viewed training step, the invariance of the pooled output. A probe-only page that frames the chapter. | `equivariance/<domain>` s, `invariance/<domain>` s, `domain/picture` p (input, transformed input, outputs) | `nn/equivariant.ts`: `equivarianceError`, `invarianceError`, `rotate90`; `nn/functional/sets.ts`: `setPool`; `nn/functional/graph.ts`: `messagePassing`; `nn/functional/conv.ts`: `conv2d` | |
| `set-models` | Deep Sets (per-element network, pooling, a network on the pool) against the Set Transformer (attention among elements, induced attention through m learnt points, pooling by attention) as modes on set tasks (the maximum, the count of a kind, a point cloud's class): permutation invariance checked by shuffling, accuracy against set size, the attention among elements at the viewed step. | `train/loss` s, `eval/accuracy` s, `set/attention` t!, `invariance/error` s, `model/predictions` p | `nn/functional/sets.ts`: `setPool`, `mab`, `sab`, `isab`, `pma`, `maskSet`; `nn/module/sets.ts`: `SetAttentionBlock`, `InducedSetAttentionBlock`, `PoolingByMultiheadAttention`; the set tasks in `shared/sets.ts` | |
| `graph-transformer` | Full attention among all nodes with the structure given only by positional encodings (Laplacian or random walk) or as an attention bias from the adjacency (modes): against message passing on the same task, the attention matrix over the graph at the viewed step (how much it follows the edges), the cost quadratic in nodes. | `train/loss` s, `eval/accuracy` s, `attention/nodes` t!, `attention/edgeShare` s (mass on true edges), `embedding` t! | `nn/blocks/transformer.ts`: `TransformerBlock`; `nn/functional/attention.ts`: `attention` (an additive bias from the adjacency); `graph/matrices.ts`: `laplacianPositionalEncoding`, `randomWalkPositionalEncoding`, `adjacencyMatrix`; assembled in `shared/` (TODO a `GraphTransformerBlock` in `nn/blocks/` once stable) | |
| `equivariant-graph-network` | Nodes with coordinates (a molecule, a point cloud): an E(n)-equivariant layer updating features by distances and coordinates by relative vectors, the output rotating with the input (the equivariance error at zero against a plain layer's), a geometric target (the centre of mass, a force field) learned. | `train/loss` s, `equivariance/error` s, `coords/trajectory` t!, `model/predictions` p | `datasets/graphs.ts`: `smallMolecules` (no coordinates: TODO 2-D layouts or a planted point-cloud generator in `shared/`); TODO `nn/functional/graph.ts`: `egnnLayer` (features and coordinates); `nn/equivariant.ts`: `equivarianceError` | |
| `group-equivariant-convolution` | see convolutional-networks | | | |
| `node-embeddings` | see representation-learning | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `graphs.ts` | The graph slot with its truth: the karate club (factions), Les Misérables, planted partitions (`contextualSbm` with its communities and feature signal-to-noise as knobs), random-graph families with their labels, small molecules, the kinship triples; small named graphs from `graph/generators.ts` (cycles, barbells, grids, the Petersen graph) for the expressivity rows | every group |
| `probes.ts` | The message-on-edge record (`messages/sent` m with the edge and its value, shared with `probabilistic-inference`), the node-embedding steps drawn over the graph (`exposition/embeddings.ts`: `embeddingSteps`), the attention-on-edges record, node colours from predictions | `layers/`, `tasks/`, `expressivity/`, `geometric/` |
| `splits.ts` | Edge splits with negatives for link prediction, labelled-node splits per class for semi-supervised node classification, graph-level splits by family | `tasks/`, `scaling/` |
| `layouts.ts` | A fixed drawing per graph (spring or spectral from `graph/spectral.ts`: `spectralEmbedding`) so every step draws nodes in the same place, and the embedding's projection aligned to it | every group |
| `sets.ts` | The set tasks (maximum, counting, point-cloud class) with their truths and the shuffle test | `geometric/` |

## Examples to make

- GCN on the karate club: the two-dimensional embedding separating the factions epoch by epoch, over the graph's drawing.
- A GAT's attention on a node's edges at the viewed step, the heads side by side.
- Over-smoothing: the Dirichlet energy and accuracy as the depth knob grows, residual connections as the fix.
- Weisfeiler-Leman refinement colour by colour on two regular graphs it cannot tell apart, then GIN against GCN on them.
- Link prediction on the karate club: the ROC on held-out edges and the predicted edges drawn.
- Deep Sets against the Set Transformer on the maximum of a set, the attention among elements.

## Rendering

- Structure views with node colour and edge weight from the tape, embedding steps, curves, colour-refinement boards, the message overlay.
- Missing: a graph panel kind that follows the cursor (node and edge attributes per step, the embedding as positions or as colour); a two-graph board for WL; small multiples per head for GAT.

## Problems

- Geometric deep learning is folded in as one group; if it grows (gauges, meshes, more equivariant layers) it becomes its own chapter under `train/graphs/geometric/` and the set models go with it. Splitting now would leave two thin chapters.
- The library has the layers, the spectral tools and WL; it lacks the plumbing of scale (neighbour sampling, relational layers) and the geometric layers (EGNN, a graph-transformer block), each named above.
- Node embeddings by random walks (`representation-learning`) and belief propagation (`probabilistic-inference`) share the graph slot and the message record; the three chapters should import one `shared/graphs.ts`.
