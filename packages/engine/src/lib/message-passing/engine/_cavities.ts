/**
 * The cavities an operator is shown, from the messages and the graph alone: a belief without one edge, the raw (unnormalised)
 * belief and cavity the evidence is taken of, the arguments an operator that needs a density is shown (a flat Gaussian
 * message widened), whether an edge's message is never stored, and the count of the edges a residual run could not
 * recompute. The leaves of the engine's modules; internal to message passing.
 */

import { FLAT_CAVITY_PRECISION } from '../core/args.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { Arg, Eta } from '../core/types.ts'
import type { Projection } from './projections.ts'
import type { MpState } from './state.ts'

/** The belief of a variable without the message of one edge. */
export function beliefWithout(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  variable: number,
  edge: number,
): Eta {
  const v = g.variables[variable] as CompiledGraph['variables'][number]
  let acc: Eta = v.family.flat
  for (const e of v.edges) if (e !== edge) acc = v.family.product(acc, messages[e] as Eta)
  return acc
}

/** Whether an edge's message is never stored: under the variational rule, a message to a defined variable from a factor that does not define it. */
export function isPulled(g: CompiledGraph, e: number, projection: Projection): boolean {
  if (projection.hook !== 'vmp') return false
  const definer = (
    g.variables[(g.edges[e] as CompiledGraph['edges'][number]).variable] as CompiledGraph['variables'][number]
  ).definer
  return definer !== undefined && definer !== e
}

/** The belief of a variable as the plain sum of its messages: no renormalisation, so the masses of the evidence keep their scale. */
export function rawBelief(g: CompiledGraph, messages: readonly (readonly number[])[], variable: number): number[] {
  const v = g.variables[variable] as CompiledGraph['variables'][number]
  const acc = new Array<number>(v.family.size).fill(0)
  for (const e of v.edges) {
    const m = messages[e] as Eta
    for (let i = 0; i < acc.length; i++) acc[i] = (acc[i] ?? 0) + (m[i] ?? 0)
  }
  return acc
}

/** The cavity as a raw difference of raw sums, without the renormalisation of a discrete family: what the evidence's masses are taken of. */
export function rawCavity(g: CompiledGraph, messages: readonly (readonly number[])[], factor: number): Arg[] {
  const f = g.factors[factor] as CompiledGraph['factors'][number]
  return f.ports.map((p): Arg => {
    if (p.edge < 0) return { family: p.family, eta: undefined, value: p.value }
    const b = rawBelief(g, messages, p.variable as number)
    const own = messages[p.edge] as Eta
    return { family: p.family, eta: b.map((v, i) => v - f.power * (own[i] ?? 0)), value: undefined }
  })
}

/**
 * The arguments an operator that `needsDensity` is shown: every flat Gaussian message replaced by a very wide one. Returns the
 * arguments unchanged for the others, and for a context in which nothing is flat.
 *
 * @param operator - The operator.
 * @param args - The cavities (or beliefs) and observed values.
 * @param to - The port the message is for.
 * @returns The arguments to show, and whether the target port was widened.
 */
export function readable(
  operator: CompiledGraph['factors'][number]['operator'],
  args: readonly Arg[],
  to: number,
): { readonly args: readonly Arg[]; readonly widenedTarget: boolean } {
  if (operator.needsDensity !== true) return { args, widenedTarget: false }
  const flat = (a: Arg): boolean =>
    a.eta !== undefined &&
    a.family.kind === 'gaussian' &&
    Math.abs(a.eta[0] ?? 0) < 1e-12 &&
    Math.abs(a.eta[1] ?? 0) < 1e-12
  if (!args.some(flat)) return { args, widenedTarget: false }
  return {
    args: args.map((a): Arg => (flat(a) ? { ...a, eta: [0, FLAT_CAVITY_PRECISION] } : a)),
    widenedTarget: flat(args[to] as Arg),
  }
}

/** The edges whose candidate message could not be computed (a message to a defined variable under the variational rule is never stored, so does not count). */
export function residualUnresolved(
  g: CompiledGraph,
  projection: Projection,
  candidates: MpState['candidates'],
): number {
  return candidates.filter((c, e) => c === null && !isPulled(g, e, projection)).length
}
