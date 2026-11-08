# A decision tree and the regions it cuts

A decision tree asks yes/no questions about one feature at a time. This example grows a small one on a real dataset and
draws it twice, as the tree of questions and as the rectangles those questions cut out of the plane, with the same
numbers and colours in both.

![A decision tree on iris petal length and width, and the three rectangles it cuts the plane into](figure.svg)

## The data and the method

The data is Fisher's iris set (`iris` in `mdml-engine/datasets`): 150 flowers, 50 of each of three species, using two of
the four measurements, petal length and petal width in cm. The model is `decisionTreeClassifier` from
`mdml-engine/trees` (gini impurity, depth at most 2), run as a session that decides one node per step.

## What the program does

1. Take the two petal columns and fit the tree. It stops after 5 steps (five nodes).
2. Ask the session for its `model/structure` record: every node, its rows, its class counts and its split.
3. Draw the record as boxes and arrows. A box is a question (S1, S2) or a leaf (L1, L2, L3). Each shows the number of
   rows of each class (in the class colours), and is filled with its majority class's colour, paler when the box is less pure.
4. Read the leaf rectangles off the same splits (each split narrows the box of rows that reach it), and draw them as
   shaded regions with the rows on top. Each split is one line. The program checks that every rectangle holds the rows
   its leaf counts and that the fitted tree predicts the leaf's class at the rectangle's centre.

## What to look for

S1 asks whether petal length is at most 2.45 cm: that is the vertical line, and everything left of it is setosa (L1, 50
of 50). S2 asks, of the other 100 flowers, whether petal width is at most 1.75 cm: the horizontal line. Below it is L2,
versicolor, 49 of 54; above it is L3, virginica, 45 of 46. The six flowers on the wrong side of the line are the only
mistakes, so training accuracy is 0.960. Printed values: leaves hold 50, 54 and 46 rows, in the tree and in the plane.
Every split is parallel to an axis, so the regions are rectangles. The tree has no way to draw a diagonal boundary.

## Run it

`make readme-examples ARGS="decision-tree"` rewrites `figure.svg` and `tape.json`.
The program is [main.tsx](main.tsx); the boxes are drawn by [tree-diagram.tsx](tree-diagram.tsx).
