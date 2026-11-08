# Handwritten digits in two dimensions

Each image in the digits data is 8 by 8 pixels, so it is a point in 64 dimensions, and nobody can look at that. An embedding
places every point on a plane so that we can look at it. Different methods keep different things.

## What the program does

- It takes 60 examples of each digit, 600 in all, from the 1,797 handwritten digits that the library ships.
- It places them on a plane with three methods: PCA, a linear projection that keeps the directions of greatest variation;
  t-SNE, which tries to keep each point's neighbours near it (fitted as a session of 300 steps, so it has a tape); and
  UMAP, which also keeps neighbours together and works from a graph of nearest neighbours.
- It measures each layout: for each point, the share of its 5 nearest neighbours *on the plane* that are the same digit.

## What to look for

- Each dot is one digit, coloured by which digit it really is. The methods were not given the colours.
- PCA mixes the digits: only 59% of a point's neighbours show the same digit.
- t-SNE and UMAP both give separate groups, one for most digits: 98% of neighbours show the same digit. Two features of
  the neighbour methods are worth knowing: the distances between the groups mean little, and the size of a group
  does not tell you much about the data. They show which points belong together.

The tape of the t-SNE run is `tape.json`. To run it again: `make readme-examples ARGS="embedding"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
