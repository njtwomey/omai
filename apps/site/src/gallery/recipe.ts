/**
 * A recipe: one file `recipes/<section>/<NN-slug>.tsx` answering one question about omai-render ("How do I take a
 * number on a log scale?"). It exports `recipe` (this metadata) and, as its default export, the live example; the
 * gallery's pages, its contents and the code shown beside the example (the file's own source) are generated from the
 * files. Toy data only: model demonstrations are the encyclopedia's.
 */
export interface Recipe {
  /** The card's and the contents' name, e.g. 'Number field'. */
  readonly title: string
  /** The question the page answers. */
  readonly question: string
  /** One to three sentences under the question; `code` in backticks is set as code. */
  readonly explain: string
  /**
   * What the page draws, for the coverage test (`coverage.test.ts`): the exposition record kinds it shows
   * (`'groupReport'`, `'featureAttributions'`), and the views it uses beyond the kind's standard one by component name
   * (`'WaterfallView'`, `'ForcePlotView'`). Every record kind the engine declares must be named by some recipe, so a new
   * kind (and its view) cannot go undisplayed; list what the page really draws, not what it could.
   */
  readonly shows?: readonly string[]
}

/** A section of the gallery: a folder of recipes. */
export interface Section {
  readonly id: string
  readonly title: string
  readonly blurb: string
}

/**
 * Titles and descriptions of the sections by folder name. A recipe's folder is its section, whether or not it is listed
 * here (a folder's `_section.ts` exporting `section: { title, blurb }` names it too); sections show in folder order.
 */
export const SECTIONS: readonly Section[] = [
  { id: 'charts', title: 'Charts and lines', blurb: 'Curves, scatter plots, bars, areas and bands from data.' },
  { id: 'axes', title: 'Axes and limits', blurb: 'Ranges and their policies, log and categorical axes, ticks.' },
  { id: 'layouts', title: 'Layouts', blurb: 'Grids of axes, shared axes, marginals, box and data aspects.' },
  {
    id: 'colour',
    title: 'Colour and themes',
    blurb: 'The tab10 slots, sequential and diverging ramps, light and dark.',
  },
  { id: 'statistical', title: 'Statistical', blurb: 'Histograms, densities, ECDFs, box plots and QQ plots.' },
  { id: 'fields', title: 'Fields', blurb: 'Smooth heatmaps, contours, decision regions and vector fields.' },
  { id: 'maths', title: 'Text and maths', blurb: 'TeX, the shared macros and equations bound to parameters.' },
  {
    id: 'controls',
    title: 'Controls',
    blurb: 'Number fields, cycling selects, toggles, seeds, panels, and controls generated from spaces and datasets.',
  },
  { id: 'interaction', title: 'Interaction', blurb: 'Dragging and adding points, constraints, probes and brushes.' },
  { id: 'animation', title: 'Animation', blurb: 'The player: precomputed timelines and live algorithms.' },
  {
    id: 'pages',
    title: 'Method pages',
    blurb:
      'Whole pages of a method as the encyclopedia composes them, from the controls, the session layer and the panels: the templates to copy.',
  },
  {
    id: 'diagrams',
    title: 'Diagrams and structures',
    blurb: 'Scenes, graph layouts, graphical models, computation graphs, architectures, trees and automata.',
  },
  {
    id: 'worlds',
    title: 'Environments and worlds',
    blurb: 'Grid worlds, mazes, ray casting and environment episodes.',
  },
  {
    id: 'panels',
    title: 'Panels of records',
    blurb:
      'The session panels that draw a record probed at the viewed step: vector fields, strips of tokens, tables with marks, grids of images, graphs, matrices with marks, scenes, accumulating points, the complex plane, the simplex and reachability plots, each on a toy method.',
  },
  {
    id: 'specialised',
    title: 'Specialised panels',
    blurb:
      'Panels for one kind of method: a zoomable picture with a pixel inspector, loss landscapes with the optimiser on them, regions of stability, equations against their truth, audio in sync, transcripts, samples over training, small multiples and a session inside a session.',
  },
  {
    id: 'infrastructure',
    title: 'Infrastructure',
    blurb:
      'What a method page runs on: sessions computed in a Web Worker while the page stays alive, the whole page exported as an image, a recording or a tape file that reproduces it, and one run told in several pages.',
  },
  {
    id: 'explanations',
    title: 'Explanations',
    blurb:
      'Records that explain a prediction: attributions of one row, of many rows and of images, text and series, dependence curves, importances, concepts, rules, prototypes, counterfactuals and the quality of an explanation, each on a made-up model.',
  },
  {
    id: 'trust',
    title: 'Trust and fairness',
    blurb:
      'Records that ask whether a model can be relied on: calibration, conformal sets, threshold sweeps, group and subgroup reports, bias in data, Pareto fronts of accuracy against parity and anomaly scores, on made-up scores.',
  },
  {
    id: 'selection',
    title: 'Model selection',
    blurb:
      'Records that compare models and settings: coefficient and likelihood tables, regularisation and pruning paths, learning and cross-validation curves, hyperparameter surfaces and search trials, plus the standard views of margins, neighbours, components, trees and decision surfaces.',
  },
  {
    id: 'steps',
    title: 'Steps and stages',
    blurb:
      'Records of a run that plays: clustering, mixture and embedding steps, populations, boards, agents, ensemble and pipeline stages, epicycles, hidden states, value grids and posteriors, each with the same scrubber as every player.',
  },
  { id: 'three-d', title: 'Three dimensions', blurb: 'Surfaces, curves and points in 3-D, orbited by dragging.' },
]
