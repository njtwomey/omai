# Complex systems (a methods chapter the encyclopedia has no home for; its pages are curiosities and foundations)

Emergence simulated: cellular automata (elementary rules, Life, lattice gases), agent-based models (boids, Schelling segregation, epidemics on networks, traffic), network models (random graphs, small worlds, preferential attachment, cascades and percolation), as sessions over `worlds`, `graph`, `stochastic` and `dynamics`.

## Capabilities

- Cellular automata: any rule stepped on a grid, density and entropy on the tape, the rule as a knob.
- Agent-based: boids with the three weights as knobs, Schelling with the tolerance, SIR on a contact network with the rate; the population's statistics on the tape and the world as a board or scene at the viewed step.
- Networks growing: preferential attachment's degree distribution forming, percolation's giant component appearing at the threshold, cascades spreading.
- Mean-field equations against the simulation (with `mathematics` ODEs).
- Fractals, displayed and zoomed, in their different forms: escape-time sets (Mandelbrot, Julia with its parameter as
  a knob, Burning Ship, Newton fractals of a polynomial), iterated function systems (Sierpinski, the Barnsley fern,
  random IFS with their maps as knobs), L-systems and curves (Koch, dragon, Hilbert, by generation), and strange
  attractors (Lorenz, Hénon, de Jong). Each is a procedure: the iteration count or generation is the step, the
  viewport is a knob (centre and scale), and the picture is a probe rendered at the viewport, so zooming is a change of
  knob that re-renders and the player replays the iterations.

## Examples to make

- Elementary rule 110 unfolding row by row, the rule number as the knob.
- Boids with the weights as knobs, the order parameter on the tape.
- SIR on a small-world network against its mean-field curve.
- Percolation on a grid: the giant component appearing as p crosses the threshold.
- The Mandelbrot set zoomed into a chosen point, the escape-time image sharpening as the iteration cap grows; a
  Julia set with the parameter dragged on the Mandelbrot set beside it.
- The Barnsley fern appearing point by point; a Koch curve or dragon curve generation by generation.

## Rendering

- Boards and scenes (`boardSteps`, `SceneView`), structure views with node state, curves against mean-field truth.
- Missing: a scene panel kind that follows the cursor (agents at the viewed step); a zoomable image panel (drag and
  wheel set the viewport knob, the probe re-renders at that viewport) for the escape-time fractals, and a point-cloud
  panel that accumulates for the iterated function systems.

## Shared

- `shared/`: the world slot (grids and networks from `worlds` and `graph`), the population-statistics record.

## Problems

- Keeps its name. The encyclopedia's part 9 (curiosities) or part 1 (mathematics) would link here; decide where its pages go.
