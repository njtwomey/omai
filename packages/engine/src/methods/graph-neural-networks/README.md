# Graph neural networks (part 6, neural architectures; with geometric deep learning)

Learning on graphs: message-passing layers (GCN, GraphSAGE, GAT, GIN) stepped on a small graph, their expressiveness (the Weisfeiler-Leman test), depth and over-smoothing, the graph tasks (node classification, link prediction, graph classification), scalable and heterogeneous graphs, and the geometric view (equivariance, symmetries), as sessions over `nn`, `graph` and the trainer base.

## Capabilities

- One message-passing layer stepped: messages on edges, aggregation at a node, the update; options: the layer type, the aggregation.
- Training on a small graph (karate club, planted partitions, Cora-sized subsets): node embeddings moving, accuracy on the tape, attention weights as a probe.
- Over-smoothing: the embeddings' similarity against depth.
- Weisfeiler-Leman: colour refinement on two graphs that it cannot tell apart.
- Link prediction: scores on held-out edges; graph classification by pooling.

## Examples to make

- GCN on the karate club: the two-dimensional embedding separating the factions epoch by epoch, over the graph's drawing.
- A GAT's attention on a node's edges at the viewed step.
- Over-smoothing: embeddings collapsing as the depth knob grows.
- Weisfeiler-Leman refinement colour by colour on two regular graphs.

## Rendering

- Structure views with node colour and edge weight from the tape, embedding steps, curves, colour-refinement boards.
- Missing: a graph panel kind that follows the cursor (node and edge attributes per step).

## Shared

- `shared/`: the graph data slot with truth (planted partitions, named small graphs), the message-passing probes.

## Problems

- New chapter, with geometric deep learning folded in. Graph layers are in `nn` partly; the rest first.
