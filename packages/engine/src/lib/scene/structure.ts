/**
 * Structures: graph-shaped descriptions as data, with typed nodes (roles, ports, badges), edges between ports, groups
 * such as plates and repeated blocks, and annotations. They describe trees, networks, factor graphs, model
 * architectures and search trees for omai-render to lay out; the functions here build, merge and slice them.
 */

import { OmaiError } from '../base/index.ts'

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

/** A named connection point on a node (a layer's inputs and outputs, a factor's variables, a block's slots). */
export interface StructurePort {
  readonly id: string
  readonly label?: string
  readonly direction: 'in' | 'out'
}

/**
 * A node of a {@link Structure}: an id, a label, a role, an optional enclosing group, a detail line, value badges
 * (shapes, parameter counts, current values), ports and free-form data.
 */
export interface StructureNode {
  readonly id: string
  readonly label: string
  readonly role: NodeRole
  readonly group?: string
  readonly detail?: string
  readonly badges?: Readonly<Record<string, string | number>>
  readonly ports?: readonly StructurePort[]
  readonly data?: Readonly<Record<string, unknown>>
}

/** An edge of a {@link Structure}, optionally between named ports of its end nodes. */
export interface StructureEdge {
  readonly from: string
  readonly to: string
  readonly fromPort?: string
  readonly toPort?: string
  readonly label?: string
  readonly directed?: boolean
  readonly kind?: 'data' | 'dependency' | 'skip' | 'message' | 'other'
}

/** A note attached to a node or group of a {@link Structure} (or to the whole structure when `target` is omitted). */
export interface StructureAnnotation {
  readonly target?: string
  readonly text: string
  readonly kind?: 'note' | 'warning' | 'highlight'
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
 * A graph-shaped description as data: typed nodes, typed edges, groups and annotations. Graphical models,
 * computation graphs, network architectures, trees, automata and pipelines all describe themselves this way, and
 * omai-render lays out and draws any structure the same way.
 *
 * @concept structures
 */
export interface Structure {
  readonly kind: 'structure'
  readonly type: string
  readonly nodes: readonly StructureNode[]
  readonly edges: readonly StructureEdge[]
  readonly groups: readonly StructureGroup[]
  readonly annotations?: readonly StructureAnnotation[]
  readonly title?: string
  readonly direction?: 'down' | 'right'
}

/**
 * Build a structure.
 *
 * @param type - What it describes (for example `computation-graph`, `bayes-net`, `architecture`).
 * @param nodes - The nodes.
 * @param edges - The edges.
 * @param options - Groups, annotations, a title and the preferred layout direction.
 * @returns The structure.
 * @concept structures
 * @throws OmaiError `value` when two nodes share an id. (An edge to an id that is not a node is allowed: a part
 *   of a structure may join nodes that another part defines; {@link mergeStructures} checks the whole.)
 * @example
 * ```ts
 * import { structure } from 'omai-engine/scene'
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
    readonly annotations?: readonly StructureAnnotation[]
    readonly title?: string
    readonly direction?: 'down' | 'right'
  } = {},
): Structure {
  const { groups = [], ...rest } = options
  const ids = new Set<string>()
  for (const n of nodes) {
    if (ids.has(n.id)) throw new OmaiError('value', `structure: the node id '${n.id}' is used twice`)
    ids.add(n.id)
  }
  return { kind: 'structure', type, nodes, edges, groups, ...rest }
}

/**
 * Combine structures into one. Nodes and groups with the same id are identified (later fields override earlier ones;
 * badges, data and ports are merged), and repeated edges are kept once. With `prefixes`, the ids of the `i`-th
 * structure are first prefixed with `prefixes[i] + '/'`, so unrelated structures sit side by side without clashing.
 *
 * @param parts - The structures.
 * @param options - Id prefixes per part, and the type, title and direction of the result (default: the first part's).
 * @returns The merged structure.
 * @throws OmaiError with code `value` when `prefixes` does not have one entry per part, or an edge of the result joins
 *   an id that is not a node.
 * @example
 * ```ts
 * import { mergeStructures, structure } from 'omai-engine/scene'
 * const a = structure('x', [{ id: 'a', label: 'a', role: 'input' }, { id: 'b', label: 'b', role: 'output' }], [{ from: 'a', to: 'b' }])
 * const b = structure('x', [{ id: 'b', label: 'b', role: 'output' }, { id: 'c', label: 'c', role: 'output' }], [{ from: 'b', to: 'c' }])
 * mergeStructures([a, b]).nodes.map((n) => n.id) // => ['a', 'b', 'c']
 * mergeStructures([a, b], { prefixes: ['l', 'r'] }).nodes.length // => 4
 * ```
 */
export function mergeStructures(
  parts: readonly Structure[],
  options: {
    readonly prefixes?: readonly string[]
    readonly type?: string
    readonly title?: string
    readonly direction?: 'down' | 'right'
  } = {},
): Structure {
  const { prefixes } = options
  if (prefixes !== undefined && prefixes.length !== parts.length) {
    throw new OmaiError('value', 'mergeStructures: give one prefix per structure')
  }
  const nodes = new Map<string, StructureNode>()
  const groups = new Map<string, StructureGroup>()
  const edges = new Map<string, StructureEdge>()
  const annotations: StructureAnnotation[] = []
  for (const [i, part] of parts.entries()) {
    const p = prefixes === undefined ? (id: string): string => id : (id: string): string => `${prefixes[i]}/${id}`
    for (const n of part.nodes) {
      const node: StructureNode = { ...n, id: p(n.id), ...(n.group === undefined ? {} : { group: p(n.group) }) }
      const old = nodes.get(node.id)
      if (old === undefined) {
        nodes.set(node.id, node)
        continue
      }
      const ports = new Map([...(old.ports ?? []), ...(node.ports ?? [])].map((q) => [q.id, q]))
      nodes.set(node.id, {
        ...old,
        ...node,
        ...((old.badges ?? node.badges) ? { badges: { ...old.badges, ...node.badges } } : {}),
        ...((old.data ?? node.data) ? { data: { ...old.data, ...node.data } } : {}),
        ...(ports.size > 0 ? { ports: [...ports.values()] } : {}),
      })
    }
    for (const g of part.groups) {
      const group: StructureGroup = { ...g, id: p(g.id), ...(g.parent === undefined ? {} : { parent: p(g.parent) }) }
      groups.set(group.id, { ...groups.get(group.id), ...group })
    }
    for (const e of part.edges) {
      const edge: StructureEdge = { ...e, from: p(e.from), to: p(e.to) }
      const key = JSON.stringify([edge.from, edge.to, edge.fromPort, edge.toPort, edge.label, edge.kind])
      if (!edges.has(key)) edges.set(key, edge)
    }
    for (const a of part.annotations ?? []) {
      annotations.push(a.target === undefined ? a : { ...a, target: p(a.target) })
    }
  }
  const first = parts[0]
  const title = options.title ?? first?.title
  const direction = options.direction ?? first?.direction
  const merged = structure(options.type ?? first?.type ?? 'merged', [...nodes.values()], [...edges.values()], {
    groups: [...groups.values()],
    ...(annotations.length > 0 ? { annotations } : {}),
    ...(title === undefined ? {} : { title }),
    ...(direction === undefined ? {} : { direction }),
  })
  for (const [i, e] of merged.edges.entries())
    for (const end of [e.from, e.to])
      if (!nodes.has(end))
        throw new OmaiError(
          'value',
          `mergeStructures: edge ${i} joins '${e.from}' to '${e.to}', and '${end}' is not a node`,
        )
  return merged
}

/**
 * Whether a value is a {@link Structure}.
 *
 * @param value - Any value.
 * @returns `true` for structures.
 * @example
 * ```ts
 * import { isStructure, structure } from 'omai-engine/scene'
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
 * import { structure, subStructure } from 'omai-engine/scene'
 * const s = structure('x', [{ id: 'a', label: 'a', role: 'input' }, { id: 'b', label: 'b', role: 'output' }], [{ from: 'a', to: 'b' }])
 * subStructure(s, ['b'], 'ancestors').nodes.length // => 2
 * ```
 */
export function subStructure(s: Structure, ids: readonly string[], direction: 'ancestors' | 'descendants'): Structure {
  // A queue over the adjacency lists: linear in the edges whatever their order (a sweep until nothing changes is
  // quadratic when the edges list the far end of a chain first).
  const next = new Map<string, string[]>()
  for (const e of s.edges) {
    const [from, to] = direction === 'descendants' ? [e.from, e.to] : [e.to, e.from]
    const out = next.get(from)
    if (out === undefined) next.set(from, [to])
    else out.push(to)
  }
  const keep = new Set(ids)
  const queue = [...keep]
  for (let head = 0; head < queue.length; head++) {
    for (const to of next.get(queue[head] as string) ?? []) {
      if (!keep.has(to)) {
        keep.add(to)
        queue.push(to)
      }
    }
  }
  const groups = new Set(s.groups.map((g) => g.id))
  return {
    ...s,
    nodes: s.nodes.filter((n) => keep.has(n.id)),
    edges: s.edges.filter((e) => keep.has(e.from) && keep.has(e.to)),
    ...(s.annotations === undefined
      ? {}
      : {
          annotations: s.annotations.filter(
            (a) => a.target === undefined || keep.has(a.target) || groups.has(a.target),
          ),
        }),
  }
}
