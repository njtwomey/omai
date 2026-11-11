/**
 * The schedule slot of the engine: in what order, and with what grouping into steps, the messages of a sweep are
 * computed. Flooding computes every message from the state at the start of the sweep, serial updates factor by factor
 * with the newest messages (Gauss-Seidel), tree goes leaves to root and back (one pass is exact on a tree), residual
 * always sends the message that would change most, and custom follows an order given by the caller. The granularity
 * says how much of a sweep one step does (the whole sweep, one factor, one message), which is what makes a run
 * steppable and cancellable.
 */

import { ParameterError } from '../core/errors.ts'
import type { CompiledGraph } from '../core/graph.ts'

/** The orders of a sweep. */
export type ScheduleKind = 'flooding' | 'serial' | 'tree' | 'residual' | 'custom'

/** How much of a sweep one step does. */
export type Granularity = 'sweep' | 'factor' | 'message'

/** The schedule of a custom order: factors by index, each with one port or all (`null`). */
export type CustomOrder = readonly (readonly [factor: number, port: number | null])[]

/** The edges of a sweep in order, grouped into steps. */
export interface Plan {
  /** The groups of edges one step each sends. */
  readonly groups: readonly (readonly number[])[]
  /** The number of edges a whole sweep sends. */
  readonly size: number
}

/** The factors of a graph in leaves-to-root-to-leaves order of a breadth-first spanning forest of the bipartite graph. */
function treeFactorOrder(g: CompiledGraph): number[] {
  const nv = g.variables.length
  const seen = new Set<number>()
  const discovered: number[] = []
  const total = nv + g.factors.length
  const neighbours = (u: number): number[] =>
    u < nv
      ? (g.variables[u]?.edges ?? []).map((e) => nv + (g.edges[e]?.factor ?? 0))
      : (g.factors[u - nv]?.ports ?? []).flatMap((p) => (p.edge >= 0 ? [g.edges[p.edge]?.variable ?? 0] : []))
  for (let root = 0; root < total; root++) {
    if (seen.has(root)) continue
    seen.add(root)
    const queue = [root]
    while (queue.length > 0) {
      const u = queue.shift() as number
      if (u >= nv) discovered.push(u - nv)
      for (const w of neighbours(u)) {
        if (!seen.has(w)) {
          seen.add(w)
          queue.push(w)
        }
      }
    }
  }
  // Leaves first: the reverse of discovery, then back out in discovery order.
  return [...[...discovered].reverse(), ...discovered]
}

const edgesOfFactor = (g: CompiledGraph, f: number): number[] =>
  (g.factors[f]?.ports ?? []).flatMap((p) => (p.edge >= 0 ? [p.edge] : []))

/**
 * The plan of a sweep: the edges in the order of the schedule, grouped as the granularity asks. A sweep sends every
 * edge once, except `tree`, which visits every factor twice (in and out), and `custom`, which sends what its order lists.
 *
 * @param g - The compiled graph.
 * @param kind - The schedule.
 * @param granularity - How many edges one step sends.
 * @param order - The order of a `custom` schedule.
 * @returns The plan.
 * @throws OmaiError with code `value` for a custom order that is missing, names a factor or port that does not exist, or is empty.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, planOf } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * planOf(g, 'serial', 'message').groups // => [[0]]
 * ```
 */
export function planOf(g: CompiledGraph, kind: ScheduleKind, granularity: Granularity, order?: CustomOrder): Plan {
  let steps: number[][]
  if (kind === 'flooding') {
    steps = g.edges.map((_, e) => [e])
  } else if (kind === 'serial') {
    steps = g.factors.flatMap((_, f) => edgesOfFactor(g, f).map((e) => [e, f]))
  } else if (kind === 'tree') {
    steps = treeFactorOrder(g).flatMap((f) => edgesOfFactor(g, f).map((e) => [e, f]))
  } else if (kind === 'residual') {
    steps = g.edges.map((_, e) => [e])
  } else {
    if (order === undefined || order.length === 0) throw new ParameterError('a custom schedule needs a non-empty order')
    steps = order.flatMap(([f, port]) => {
      const factor = g.factors[f]
      if (factor === undefined) throw new ParameterError(`custom schedule: there is no factor ${f}`)
      if (port === null) return edgesOfFactor(g, f).map((e) => [e, f])
      const p = factor.ports[port]
      if (p === undefined || p.edge < 0)
        throw new ParameterError(`custom schedule: factor ${f} has no message to send at port ${port}`)
      return [[p.edge, f]]
    })
  }
  // Each entry is [edge, factor]; grouping by granularity (flooding and residual have no factor grouping).
  const edges = steps.map((s) => s[0] as number)
  const size = edges.length
  if (granularity === 'sweep') return { groups: [edges], size }
  if (granularity === 'message' || kind === 'flooding' || kind === 'residual')
    return { groups: edges.map((e) => [e]), size }
  const groups: number[][] = []
  let last = -1
  for (const [e, f] of steps as [number, number][]) {
    if (f !== last) groups.push([])
    groups[groups.length - 1]?.push(e)
    last = f
  }
  return { groups, size }
}
