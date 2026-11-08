# A rolled-up sheet, flattened

A Swiss roll is a flat sheet rolled up in three dimensions. Points that are far apart along the sheet can be close in
space, where the layers touch. This example looks at the roll in two dimensions, first with PCA and then with t-SNE.

![A Swiss roll in two dimensions by PCA and by t-SNE](figure.svg)

## The data and the method

The data is 120 points from the Swiss roll generator, which keeps each point's true position along the sheet. PCA is the
linear projection onto the two directions of largest variance. t-SNE (`tsne` in `mdml-engine/manifold`) places the points in
the plane so that points near each other in three dimensions stay near each other, by gradient descent on a divergence between
two sets of neighbour probabilities (perplexity 30, 300 steps).

## What the program does

1. Generate the roll; fit PCA with two components and transform the data.
2. Run t-SNE as a session and read its embedding and its objective back at ten evenly spaced steps. A checkpoint is kept every
   100 steps so the tape stays small (about 90 KB).
3. Draw both with the render package's embedding view, each point coloured by its true position along the sheet.

## What to look for

A good layout is a smooth gradient of colour. PCA can only look at the roll from one side, so colours from different layers mix. The t-SNE panel also
shows how the objective fell: the KL divergence is 1.599 after 30 steps and 0.317 at the end. As a number: the mean distance
along the sheet between a point and its 5 nearest neighbours in the plane is 1.63 for PCA and 0.98 for t-SNE, on a sheet that spans
4.8 to 14.1. t-SNE keeps the sheet's neighbours together better, but it does not unroll the whole sheet.

## Run it

`make readme-examples ARGS="embedding"` rewrites `figure.svg` and `tape.json`.
The program is [main.tsx](main.tsx).
