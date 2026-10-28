/**
 * A recipe: one file `recipes/<section>/<NN-slug>.tsx` answering one question about mdml-render ("How do I take a
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
    id: 'diagrams',
    title: 'Diagrams and structures',
    blurb: 'Scenes, graph layouts, graphical models, computation graphs, architectures, trees and automata.',
  },
  {
    id: 'worlds',
    title: 'Environments and worlds',
    blurb: 'Grid worlds, mazes, ray casting and environment episodes.',
  },
  { id: 'three-d', title: 'Three dimensions', blurb: 'Surfaces, curves and points in 3-D, orbited by dragging.' },
]
