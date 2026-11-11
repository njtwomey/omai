/**
 * The update of one edge: the proposal of the operator (alone, or the fixed point of an alpha-divergence projection), and
 * its resolution into the new message, damped and repaired by the improper policy if it would make a proper belief
 * improper; with the change of the beliefs over a sweep. Internal to message passing.
 */

import { OmaiError } from '../../base/index.ts'
import type { CompiledGraph } from '../core/graph.ts'
import { type Arg, type Eta, takesPower } from '../core/types.ts'
import { argsOf } from './_arguments.ts'
import { beliefWithout, isPulled } from './_cavities.ts'
import { beliefOf } from './cavity.ts'
import { type FactorProposal, operatorMessage } from './dispatch.ts'
import type { EngineSettings } from './state.ts'

/** The largest number of fixed-point iterations of one alpha update. */
const ALPHA_ITERATIONS = 200

/**
 * The update of an alpha-divergence projection (Minka 2005) for one edge: the member of the family that minimises the
 * alpha-divergence D_alpha(f q_cavity || q) of the tilted distribution, found as the fixed point of "project the
 * distribution proportional to f^alpha cavity^alpha q^(1 - alpha) onto the family by moment matching" (the stationarity
 * condition of D_alpha for an exponential family). The first iteration at alpha = 1 is the EP update.
 */
function alphaProposal(
  g: CompiledGraph,
  s: EngineSettings,
  source: readonly (readonly number[])[],
  e: number,
  alpha: number,
): FactorProposal {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
  const to = edge.port
  const cavities = argsOf(g, source, edge.factor, s.projection)
  const target = cavities[to] as Arg
  const family = target.family
  const cavity = target.eta as Eta
  // The other ports see the cavity with the (1 - alpha) share of their own current message: cavity^alpha q^(1 - alpha).
  const others = f.ports.map((p, q): Arg => {
    const a = cavities[q] as Arg
    if (q === to || p.edge < 0 || a.eta === undefined) return a
    return { ...a, eta: a.family.product(a.eta, a.family.power(source[p.edge] as Eta, 1 - alpha)) }
  })
  let own: Eta = source[e] as Eta
  for (let k = 0; k < ALPHA_ITERATIONS; k++) {
    const base = family.product(cavity, family.power(own, 1 - alpha))
    const args = others.map((a, q) => (q === to ? { ...a, eta: base } : a))
    const p = operatorMessage(f, s.projection, to, args, alpha)
    if (p.kind !== 'message') return p
    // The operator returns (projection - base) / alpha: the projected member of the family is base + alpha * message.
    const projected = base.map((v, i) => v + alpha * (p.eta[i] ?? 0))
    const next = projected.map((v, i) => v - (cavity[i] ?? 0))
    const moved = family.change(own, next)
    own = next
    if (!(moved > 1e-12)) break
  }
  return { kind: 'message', eta: own }
}

export function propose(
  g: CompiledGraph,
  s: EngineSettings,
  source: readonly (readonly number[])[],
  e: number,
): FactorProposal {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
  if (isPulled(g, e, s.projection)) return { kind: 'blocked' }
  const alpha = s.projection.alpha
  if (alpha !== undefined && alpha !== 1 && takesPower(f.operator, f.params) && f.power === 1)
    return alphaProposal(g, s, source, e, alpha)
  return operatorMessage(f, s.projection, edge.port, argsOf(g, source, edge.factor, s.projection))
}

/** The new message of an edge from a proposal: damped, and repaired if it would make a proper belief improper. */
export function resolve(
  g: CompiledGraph,
  s: EngineSettings,
  current: readonly (readonly number[])[],
  e: number,
  proposal: FactorProposal,
  counts: { events: number; skipped: number; damped: number; reset: number },
): Eta {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const family = (g.variables[edge.variable] as CompiledGraph['variables'][number]).family
  const old = current[e] as Eta
  if (proposal.kind === 'blocked') return old
  if (proposal.kind === 'failed') {
    counts.events++
    counts.skipped++
    return old
  }
  const raw = proposal.kind === 'flat' ? family.flat : proposal.eta
  const d = s.damping
  const next = d === 0 ? raw : family.product(family.power(old, d), family.power(raw, 1 - d))
  const base = beliefWithout(g, current, edge.variable, e)
  if (!family.isProper(family.product(base, old)) || family.isProper(family.product(base, next))) return next
  counts.events++
  if (s.improper === 'error') {
    throw new OmaiError(
      'convergence',
      `message passing: an update of ${(g.factors[edge.factor] as CompiledGraph['factors'][number]).id} makes the belief of ${(g.variables[edge.variable] as CompiledGraph['variables'][number]).name} improper`,
    )
  }
  if (s.improper === 'reset') {
    counts.reset++
    return family.flat
  }
  if (s.improper === 'damp') {
    let step = 1
    for (let k = 0; k < 10; k++) {
      step /= 2
      const trial = old.map((v, i) => v + step * ((next[i] ?? 0) - v))
      if (family.isProper(family.product(base, trial))) {
        counts.damped++
        return trial
      }
    }
  }
  counts.skipped++
  return old
}

export function sweepChange(
  g: CompiledGraph,
  start: readonly (readonly number[])[],
  messages: readonly (readonly number[])[],
): {
  readonly change: number
  readonly beliefs: number[][]
} {
  let change = 0
  const beliefs: number[][] = []
  for (const [i, v] of g.variables.entries()) {
    const b = [...beliefOf(g, messages, i)]
    beliefs.push(b)
    if (v.edges.length > 0) change = Math.max(change, v.family.change(start[i] as Eta, b))
  }
  return { change, beliefs }
}
