/**
 * The visual language of each kind of structure: how nodes of each role (latent and observed variables, factors,
 * operations, decisions, states), edges of each kind and groups of each kind are drawn for graphical models, trees,
 * automata, neural networks and a generic fallback, looked up with `styleFor`.
 */

import type { NodeRole, Paint, StructureEdge, StructureGroup, StructureNodeShape } from 'omai-engine/scene'

/** The outline of a node: the shapes a structure's nodes can ask for (`look.shape`), including `none` and `stack`. */
export type NodeShape = StructureNodeShape

/**
 * How one role of node is drawn. A node's own `look` (see `StructureNodeLook`) replaces fields of it, and a
 * `StructureEdgeLook` those of {@link EdgeLook}.
 */
export interface NodeLook {
  readonly shape: NodeShape
  readonly fill: Paint
  readonly stroke: Paint
  readonly text: Paint
  readonly strokeWidth?: number
  readonly dash?: readonly number[]
  /** A second outline just inside the first (deterministic nodes, accepting states). */
  readonly double?: boolean
}

/** How one kind of edge is drawn. */
export interface EdgeLook {
  readonly stroke: Paint
  readonly strokeWidth?: number
  readonly dash?: readonly number[]
  readonly arrow: boolean
}

/** How one kind of group is drawn. */
export interface GroupLook {
  readonly stroke: Paint
  readonly fill: Paint
  readonly dash?: readonly number[]
  readonly radius: number
}

/** The visual language of one kind of structure: node looks by role, edge looks by kind, group looks by kind. */
export interface StructureStyle {
  readonly node: (role: NodeRole) => NodeLook
  readonly edge: (edge: StructureEdge) => EdgeLook
  readonly group: (group: StructureGroup) => GroupLook
  /** The layout the kind reads best in. */
  readonly layout: 'layered' | 'tree' | 'circular' | 'force' | 'bipartite'
  readonly fontSize: number
}

const round = (fill: Paint, stroke: Paint): NodeLook => ({ shape: 'round', fill, stroke, text: 'fg' })

/** Nodes by role in the general language: data in blue, operations in green, outputs in orange, the rest neutral. */
function generalNode(role: NodeRole): NodeLook {
  switch (role) {
    case 'input':
    case 'constant':
      return round('surface', 'tab0')
    case 'output':
      return round('surface', 'tab1')
    case 'operation':
      return { shape: 'ellipse', fill: 'surface', stroke: 'tab2', text: 'fg' }
    case 'parameter':
      return round('surface', 'tab4')
    case 'tensor':
      return { shape: 'rect', fill: 'surface', stroke: 'muted', text: 'fg' }
    case 'block':
      return round('surface', 'accent')
    case 'latent':
      return { shape: 'circle', fill: 'bg', stroke: 'fg', text: 'fg', strokeWidth: 1.5 }
    case 'observed':
      return { shape: 'circle', fill: 'grid', stroke: 'fg', text: 'fg', strokeWidth: 1.5 }
    case 'deterministic':
      return { shape: 'circle', fill: 'bg', stroke: 'fg', text: 'fg', double: true }
    case 'factor':
      return { shape: 'square', fill: 'fg', stroke: 'fg', text: 'bg' }
    case 'decision':
      return { shape: 'rect', fill: 'surface', stroke: 'tab0', text: 'fg' }
    case 'utility':
      return { shape: 'diamond', fill: 'surface', stroke: 'tab1', text: 'fg' }
    case 'state':
      return { shape: 'circle', fill: 'surface', stroke: 'fg', text: 'fg' }
    case 'leaf':
      return round('surface', 'tab2')
    case 'internal':
      return round('surface', 'tab0')
    case 'other':
      return round('surface', 'muted')
  }
}

function generalEdge(edge: StructureEdge): EdgeLook {
  const arrow = edge.directed !== false
  switch (edge.kind) {
    case 'skip':
      return { stroke: 'muted', dash: [5, 4], arrow }
    case 'message':
      return { stroke: 'accent', strokeWidth: 1.5, arrow }
    case 'dependency':
      return { stroke: 'muted', dash: [2, 3], arrow }
    default:
      return { stroke: 'muted', strokeWidth: 1.2, arrow }
  }
}

function generalGroup(group: StructureGroup): GroupLook {
  switch (group.kind) {
    case 'plate':
      return { stroke: 'fg', fill: 'none', radius: 4 }
    case 'repeat':
      return { stroke: 'accent', fill: 'none', dash: [6, 4], radius: 8 }
    case 'block':
      return { stroke: 'grid', fill: 'surface', radius: 8 }
    case 'cluster':
      return { stroke: 'grid', fill: 'none', dash: [3, 3], radius: 8 }
  }
}

const general: StructureStyle = {
  node: generalNode,
  edge: generalEdge,
  group: generalGroup,
  layout: 'layered',
  fontSize: 12,
}

/**
 * Graphical models: circles for variables (observed shaded), squares for factors, small dots for fixed parameters and
 * hyperparameters (labelled beside them), plates around repeats.
 */
const graphicalModel: StructureStyle = {
  ...general,
  node: (role) => (role === 'parameter' ? { shape: 'dot', fill: 'fg', stroke: 'fg', text: 'fg' } : generalNode(role)),
  fontSize: 13,
}

/** Undirected graphs (Markov networks, factor graphs): no arrow heads. */
const undirected: StructureStyle = {
  ...general,
  edge: (e) => ({ ...generalEdge(e), arrow: e.directed === true }),
  layout: 'force',
}

/** Factor graphs: undirected, variables in one row and factors in another. */
const factorGraph: StructureStyle = { ...undirected, layout: 'bipartite' }

/** Trees (decision and search trees): leaves green, internal nodes blue, laid out as tidy trees. */
const tree: StructureStyle = { ...general, layout: 'tree' }

/** Automata: states as circles (accepting ones doubled), transitions labelled, in a circle. */
const automaton: StructureStyle = {
  ...general,
  node: (role) =>
    role === 'output'
      ? { shape: 'circle', fill: 'surface', stroke: 'fg', text: 'fg', double: true }
      : role === 'input'
        ? { shape: 'circle', fill: 'surface', stroke: 'accent', text: 'fg', strokeWidth: 2 }
        : { shape: 'circle', fill: 'surface', stroke: 'fg', text: 'fg' },
  layout: 'layered',
}

/** Networks (social and road graphs): small dots, force layout. */
const network: StructureStyle = {
  ...general,
  node: (role) => ({ ...generalNode(role), shape: 'dot', fill: 'tab0', stroke: 'bg' }),
  edge: (e) => ({ stroke: 'grid', arrow: e.directed === true }),
  layout: 'force',
  fontSize: 10,
}

const STYLES: Readonly<Record<string, StructureStyle>> = {
  'bayes-net': graphicalModel,
  'graphical-model': graphicalModel,
  'directed-model': graphicalModel,
  'markov-network': undirected,
  'factor-graph': factorGraph,
  undirected,
  tree,
  'decision-tree': tree,
  'search-tree': tree,
  automaton,
  'state-machine': automaton,
  network,
  graph: network,
}

/**
 * The visual language of a kind of structure (its `type`): graphical models (`bayes-net`, `factor-graph`,
 * `markov-network`), trees (`tree`, `decision-tree`, `search-tree`), automata, networks, and a general language for
 * computation graphs, architectures and pipelines.
 *
 * @param type - The structure's type.
 * @returns The style.
 * @example
 * ```ts
 * import { styleFor } from 'omai-render/structures'
 * styleFor('bayes-net').node('observed').shape // => 'circle'
 * styleFor('factor-graph').edge({ from: 'a', to: 'f' }).arrow // => false
 * ```
 */
export function styleFor(type: string): StructureStyle {
  return STYLES[type] ?? general
}
