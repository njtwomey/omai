# Principal components of the iris data

Each of the 150 iris flowers is described by four measurements: sepal length, sepal width, petal length and petal width.
Principal component analysis (PCA) finds new directions in that four-dimensional space, ordered by how much of the
variation between flowers they carry, so that the first two directions can stand in for all four.

## What the program does

- It fits PCA to the iris data as a session, with the power-iteration solver, which finds one component per step. It took
  4 steps, one for each component.
- It asks the session for the components and for each flower's position along them, and builds the library's `components`
  record from them. The render package draws the record; the example only adds the legend of the three species.

## What to look for

- On the left each dot is a flower, placed by its first two components and coloured by species. The arrows show how each
  measurement points in that plane. Setosa (blue) is set apart from the other two, and the long arrow for petal length shows
  what separates it.
- Top right, the bar chart is the share of the variance that each component explains: 92.5%, 5.3%, 1.7% and 0.5%. The
  first component alone carries most of it. The line is the running total.
- Bottom right, the table of loadings says how much each measurement contributes to each component. The first component is
  mostly petal length (0.857) and the second mostly sepal width (0.73).

The tape of the run is `tape.json`. To run it again: `make readme-examples ARGS="pca"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
