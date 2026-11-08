/**
 * Computation graphs as data: `computationGraph` records every primitive a function applies at an input, with shapes,
 * operations, scalar values and gradients, as a structure that can be drawn, and `graphStats` summarises it (node and
 * edge counts, depth, how often each primitive appears).
 */

import { OmaiError } from '../base/index.ts'
import { type Structure, type StructureEdge, type StructureNode, structure } from '../scene/index.ts'
import {
  formatShape,
  isTensor,
  ones,
  sizeOf,
  type Tensor,
  toFlat,
  treeFlatten,
  treeUnflatten,
  type Value,
} from '../tensor/index.ts'
import { withLevel } from './_level.ts'
import { backward, isReverseValueOf, type ReverseValue, reverseTrace, type TapeNode } from './reverse.ts'

/** Options for {@link computationGraph}. */
export interface GraphOptions {
  /** Names for the input leaves, in flattening order (default: their pytree paths). */
  readonly inputNames?: readonly string[]
  /** Include scalar values as badges (default true). */
  readonly values?: boolean
  /**
   * Badge every node with the derivative of the output with respect to it (`grad` for scalars, `|grad|` otherwise),
   * when the function returns a single scalar (default: the same as `values`).
   */
  readonly gradients?: boolean
}

function gradientBadge(g: Value): Record<string, number> {
  if (!isTensor(g)) return {}
  const flat = toFlat(g as Tensor)
  if (flat.length === 1) return { grad: Number((flat[0] ?? 0).toPrecision(6)) }
  let s = 0
  for (const v of flat) s += v * v
  return { '|grad|': Number(Math.sqrt(s).toPrecision(6)) }
}

function scalarOf(x: Value): number | undefined {
  return isTensor(x) && sizeOf(x.shape) === 1 ? (x.data[0] ?? undefined) : undefined
}

/**
 * Record the computation graph of a function at an input: every primitive applied to values derived from the input,
 * with shapes, operations, (for scalars) values and, for a scalar output, the gradient of the output with respect to
 * every node, as a {@link Structure} that omai-render draws.
 *
 * @param f - The function.
 * @param input - Its input (a value or a pytree).
 * @param options - Input names and whether to show values and gradients.
 * @returns The structure (type `computation-graph`).
 * @concept computation-graph
 * @example
 * ```ts
 * import { computationGraph } from 'omai-engine/autodiff'
 * import { add, mul, tensor, type Value } from 'omai-engine/tensor'
 * const g = computationGraph((x: Value) => add(mul(x, x), 1), tensor(2))
 * g.nodes.map((n) => n.label) // => ['x', 'mul', '1', 'add']
 * g.nodes[0].badges // => { value: 2, grad: 4 }
 * ```
 */
export function computationGraph<P>(f: (input: P) => unknown, input: P, options: GraphOptions = {}): Structure {
  const { leaves, def } = treeFlatten(input)
  return withLevel((level) => {
    const trace = reverseTrace(level)
    const traced = leaves.map((l) => trace.leafOf(l))
    const result = f(treeUnflatten(def, traced) as P)
    const outputs = treeFlatten(result).leaves.filter((o): o is ReverseValue => isReverseValueOf(o, trace))
    const names = options.inputNames ?? (leaves.length === 1 ? ['x'] : leaves.map((_, i) => `x${i}`))
    const showValues = options.values ?? true

    const nodes: StructureNode[] = []
    const edges: StructureEdge[] = []
    const ids = new Map<ReverseValue | TapeNode, string>()
    const valueOfId = new Map<string, ReverseValue>()
    const nextId = (): string => `n${nodes.length}`
    const badge = (v: Value): Record<string, string | number> => {
      const s = showValues ? scalarOf(v) : undefined
      return s === undefined ? { shape: formatShape(v.shape) } : { value: Number(s.toPrecision(6)) }
    }

    const visitValue = (x: ReverseValue): string => {
      const known = ids.get(x.node ?? x)
      if (known !== undefined) return known
      if (x.node !== null) return visitNode(x.node)
      const id = nextId()
      const leafIndex = traced.indexOf(x)
      const primal = x.primal
      const label = leafIndex >= 0 ? (names[leafIndex] ?? `x${leafIndex}`) : String(scalarOf(primal) ?? 'const')
      nodes.push({ id, label, role: leafIndex >= 0 ? 'input' : 'constant', badges: badge(primal) })
      ids.set(x, id)
      valueOfId.set(id, x)
      return id
    }

    const visitNode = (node: TapeNode): string => {
      const known = ids.get(node)
      if (known !== undefined) return known
      const inputIds = node.inputs.map(visitValue)
      const id = nextId()
      const detail = node.primitive.describe?.(node.params)
      nodes.push({
        id,
        label: node.primitive.name,
        role: 'operation',
        ...(detail === undefined ? {} : { detail }),
        badges: badge(node.outputs[0] as Value),
      })
      ids.set(node, id)
      const first = node.results[0]
      if (first !== undefined) valueOfId.set(id, first)
      for (const from of inputIds) edges.push({ from, to: id, directed: true, kind: 'data' })
      return id
    }

    for (const o of outputs) visitValue(o)
    const out = outputs[0]
    const wantGradients = options.gradients ?? showValues
    if (!wantGradients || outputs.length !== 1 || out === undefined || sizeOf(out.shape) !== 1) {
      return structure('computation-graph', nodes, edges, { direction: 'right' })
    }
    const marked = nodes.map((n) => (n.role === 'constant' ? null : (valueOfId.get(n.id) ?? null)))
    const tracked = marked.filter((v): v is ReverseValue => v !== null)
    let grads: Value[]
    try {
      grads = backward([out], [ones(out.shape, { dtype: out.dtype })], tracked)
    } catch (e) {
      // A graph through a primitive without derivative rules is still drawn, just without gradients.
      if (e instanceof OmaiError && e.code === 'not-differentiable') {
        return structure('computation-graph', nodes, edges, { direction: 'right' })
      }
      throw e
    }
    const gradOf = new Map(tracked.map((v, i) => [v, grads[i]]))
    const badged = nodes.map((n, i) => {
      const v = marked[i] ?? null
      const g = v === null ? undefined : gradOf.get(v)
      return g === undefined ? n : { ...n, badges: { ...n.badges, ...gradientBadge(g) } }
    })
    return structure('computation-graph', badged, edges, { direction: 'right' })
  })
}

/** Counts that summarise a computation graph. */
export interface GraphStats {
  readonly nodes: number
  readonly edges: number
  readonly inputs: number
  readonly constants: number
  readonly operations: number
  /** The longest chain of operations from an input to an output. */
  readonly depth: number
  /** How many times each primitive appears, by name. */
  readonly operationCounts: Readonly<Record<string, number>>
}

/**
 * Summarise a computation graph: numbers of nodes, edges, inputs, constants and operations, the depth (the longest
 * chain of operations) and how often each primitive appears.
 *
 * @param graph - A structure from {@link computationGraph}.
 * @returns The counts.
 * @concept computation-graph
 * @example
 * ```ts
 * import { computationGraph, graphStats } from 'omai-engine/autodiff'
 * import { add, mul, sin, tensor, type Value } from 'omai-engine/tensor'
 * const g = computationGraph((x: Value) => add(mul(x, x), sin(x)), tensor(2))
 * graphStats(g).operations // => 3
 * graphStats(g).depth // => 2
 * graphStats(g).operationCounts // => { mul: 1, sin: 1, add: 1 }
 * ```
 */
export function graphStats(graph: Structure): GraphStats {
  const roleOf = new Map(graph.nodes.map((n) => [n.id, n.role]))
  const incoming = new Map<string, string[]>()
  for (const e of graph.edges) incoming.set(e.to, [...(incoming.get(e.to) ?? []), e.from])
  const depthOf = new Map<string, number>()
  // Nodes are recorded after their inputs, so one pass in order finds every depth.
  let depth = 0
  for (const n of graph.nodes) {
    let d = 0
    for (const from of incoming.get(n.id) ?? []) d = Math.max(d, depthOf.get(from) ?? 0)
    const own = d + (roleOf.get(n.id) === 'operation' ? 1 : 0)
    depthOf.set(n.id, own)
    depth = Math.max(depth, own)
  }
  const operationCounts: Record<string, number> = {}
  let inputs = 0
  let constants = 0
  let operations = 0
  for (const n of graph.nodes) {
    if (n.role === 'operation') {
      operations += 1
      operationCounts[n.label] = (operationCounts[n.label] ?? 0) + 1
    } else if (n.role === 'input') inputs += 1
    else if (n.role === 'constant') constants += 1
  }
  return { nodes: graph.nodes.length, edges: graph.edges.length, inputs, constants, operations, depth, operationCounts }
}
