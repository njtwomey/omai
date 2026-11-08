# A decision tree and the regions it cuts

A decision tree classifies a flower by asking a few questions about its measurements, one after another. This example
fits a tree to Fisher's iris data using only two measurements, petal length and petal width, and shows the library's own
pictures of the result.

## What the program does

- It takes the 150 iris flowers (three species) and keeps petal length and petal width.
- It grows a classification tree (CART, depth at most 3) as a session, one node decided per step. It took 9 steps.
- It asks the session for two records at the end: `model/structure`, the tree, and `model/decision-surface`, the regions
  of the plane. The render package draws both. Nothing is drawn by the example itself.

## What to look for

- On the left is the tree. The top box asks "petal length <= 2.45". Flowers for which this is true go left, to a leaf
  that holds only setosa (50 flowers). The others go right to a second question, and so on. Each box shows how many
  flowers reached it, the share of each species as a coloured bar, and its gini impurity (0 means only one species).
- On the right is the same tree as a picture of the plane. Every leaf is a rectangle, and every question is one of the
  straight lines. The flowers are drawn on top in their species colours, so you can see which rectangle each one fell in.
- 146 of the 150 flowers are classified correctly. The four that are not sit where versicolor and virginica overlap.

The tape of the run is `tape.json`. To run it again: `make readme-examples ARGS="decision-tree"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
