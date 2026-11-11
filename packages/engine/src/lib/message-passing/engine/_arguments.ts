/**
 * The arguments of a factor's operator from the messages: the cavity (or the belief) of each variable and the value of
 * each observed one, with the variational cavity of a defined variable pulled from the messages its other factors would
 * send, and the refresh of the deterministic factors whose inputs moved. Internal to message passing.
 */

import type { CompiledGraph } from '../core/graph.ts'
import type { Arg, Eta } from '../core/types.ts'
import { beliefOf } from './cavity.ts'
import { operatorMessage } from './dispatch.ts'
import type { Projection } from './projections.ts'

/**
 * What the messages of a variable's other factors add up to, when the variable is defined by a deterministic factor and
 * the rule is the variational one: the messages those factors would send now (`pull`), summed. A defined variable keeps
 * no message from them (its belief is its definer's pushforward alone); the definer reads their sum as the cavity of its
 * output and passes it to its inputs.
 */
function pulledCavity(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  variable: number,
  definer: number,
  projection: Projection,
): Eta {
  const v = g.variables[variable] as CompiledGraph['variables'][number]
  let acc: Eta = v.family.flat
  for (const e of v.edges) {
    if (e === definer) continue
    const edge = g.edges[e] as CompiledGraph['edges'][number]
    const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
    const p = operatorMessage(f, projection, edge.port, argsOf(g, messages, edge.factor, projection))
    if (p.kind === 'message') acc = v.family.product(acc, p.eta)
  }
  return acc
}

/**
 * Re-send the output message of every deterministic factor whose inputs moved (and, transitively, the factors reading
 * the outputs that changed), so that under the variational rule a defined variable's belief is at all times the
 * pushforward of the current beliefs of its inputs, whatever the schedule. The message to a defined output depends on the
 * inputs' beliefs alone, so the output port is shown no cavity.
 */
export function refreshDefined(
  g: CompiledGraph,
  projection: Projection,
  messages: (readonly number[])[],
  edges: readonly number[],
): void {
  if (projection.hook !== 'vmp' || !g.hasDefined) return
  const moved = new Set(edges.map((e) => (g.edges[e] as CompiledGraph['edges'][number]).variable))
  for (const f of g.factors) {
    const k = f.operator.defines
    const out = k === undefined ? undefined : f.ports[k]
    if (k === undefined || out === undefined || out.edge < 0) continue
    if (!f.ports.some((p, i) => i !== k && p.variable !== undefined && moved.has(p.variable))) continue
    const args = argsOf(g, messages, f.index, projection, true)
    const proposal = operatorMessage(f, projection, k, args)
    if (proposal.kind === 'message') messages[out.edge] = proposal.eta
    moved.add(out.variable as number)
  }
}

/** The arguments of a factor's operator: the cavity or the belief of each variable, the value of each observed one. */
export function argsOf(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  factor: number,
  projection: Projection,
  forward = false,
): Arg[] {
  const f = g.factors[factor] as CompiledGraph['factors'][number]
  return f.ports.map((p, k): Arg => {
    if (p.edge < 0) return { family: p.family, eta: undefined, value: p.value }
    if (projection.hook === 'vmp' && f.operator.defines === k)
      return {
        family: p.family,
        eta: forward ? p.family.flat : pulledCavity(g, messages, p.variable as number, p.edge, projection),
        value: undefined,
      }
    const b = beliefOf(g, messages, p.variable as number)
    if (projection.arguments === 'belief') return { family: p.family, eta: b, value: undefined }
    const own = messages[p.edge] as Eta
    const eta = f.power === 1 ? p.family.quotient(b, own).eta : p.family.product(b, p.family.power(own, -f.power))
    return { family: p.family, eta, value: undefined }
  })
}
