/**
 * The role of a node in a structure; renderers pick shapes and colours by role.
 */
export type NodeRole =
  | 'input'
  | 'constant'
  | 'operation'
  | 'output'
  | 'parameter'
  | 'latent'
  | 'observed'
  | 'deterministic'
  | 'factor'
  | 'decision'
  | 'utility'
  | 'state'
  | 'block'
  | 'tensor'
  | 'leaf'
  | 'internal'
  | 'other'

/** A node of a {@link Structure}. */
export interface StructureNode {
  readonly id: string
  readonly label: string
  readonly role: NodeRole
  readonly group?: string
  readonly detail?: string
  readonly badges?: Readonly<Record<string, string | number>>
  readonly data?: Readonly<Record<string, unknown>>
}

/** An edge of a {@link Structure}. */
export interface StructureEdge {
  readonly from: string
  readonly to: string
  readonly label?: string
  readonly directed?: boolean
  readonly kind?: 'data' | 'dependency' | 'skip' | 'message' | 'other'
}

/** A group of nodes: a plate, a repeated block, a cluster. */
export interface StructureGroup {
  readonly id: string
  readonly label: string
  readonly kind: 'plate' | 'repeat' | 'cluster' | 'block'
  readonly parent?: string
  readonly count?: number | string
}

/**
 * A graph-shaped description as data: typed nodes, typed edges and groups. Graphical models, computation graphs,
 * network architectures, trees, automata and pipelines all describe themselves this way, and whitebox-render lays out
 * and draws any structure the same way.
 *
 * @concept structures
 */
export interface Structure {
  readonly kind: 'structure'
  readonly type: string
  readonly nodes: readonly StructureNode[]
  readonly edges: readonly StructureEdge[]
  readonly groups: readonly StructureGroup[]
  readonly title?: string
  readonly direction?: 'down' | 'right'
}

/**
 * Build a structure.
 *
 * @param type - What it describes (for example `computation-graph`, `bayes-net`, `architecture`).
 * @param nodes - The nodes.
 * @param edges - The edges.
 * @param options - Groups, a title and the preferred layout direction.
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { structure } from 'whitebox-engine/scene'
 * const s = structure('example', [{ id: 'a', label: 'a', role: 'input' }, { id: 'b', label: 'b', role: 'output' }], [{ from: 'a', to: 'b' }])
 * s.nodes.length // => 2
 * ```
 */
export function structure(
  type: string,
  nodes: readonly StructureNode[],
  edges: readonly StructureEdge[],
  options: {
    readonly groups?: readonly StructureGroup[]
    readonly title?: string
    readonly direction?: 'down' | 'right'
  } = {},
): Structure {
  const { groups = [], ...rest } = options
  return { kind: 'structure', type, nodes, edges, groups, ...rest }
}

/**
 * Whether a value is a {@link Structure}.
 *
 * @param value - Any value.
 * @returns `true` for structures.
 * @example
 * ```ts
 * import { isStructure, structure } from 'whitebox-engine/scene'
 * isStructure(structure('x', [], [])) // => true
 * ```
 */
export function isStructure(value: unknown): value is Structure {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'structure'
}

/**
 * The part of a structure reachable from (or leading to) some nodes.
 *
 * @param s - The structure.
 * @param ids - The starting nodes.
 * @param direction - Follow edges forwards (`descendants`) or backwards (`ancestors`).
 * @returns The sub-structure.
 * @example
 * ```ts
 * import { structure, subStructure } from 'whitebox-engine/scene'
 * const s = structure('x', [{ id: 'a', label: 'a', role: 'input' }, { id: 'b', label: 'b', role: 'output' }], [{ from: 'a', to: 'b' }])
 * subStructure(s, ['b'], 'ancestors').nodes.length // => 2
 * ```
 */
export function subStructure(s: Structure, ids: readonly string[], direction: 'ancestors' | 'descendants'): Structure {
  const keep = new Set(ids)
  let changed = true
  while (changed) {
    changed = false
    for (const e of s.edges) {
      const [from, to] = direction === 'descendants' ? [e.from, e.to] : [e.to, e.from]
      if (keep.has(from) && !keep.has(to)) {
        keep.add(to)
        changed = true
      }
    }
  }
  return {
    ...s,
    nodes: s.nodes.filter((n) => keep.has(n.id)),
    edges: s.edges.filter((e) => keep.has(e.from) && keep.has(e.to)),
  }
}
