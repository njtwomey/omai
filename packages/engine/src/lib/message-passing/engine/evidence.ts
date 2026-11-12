/**
 * The evidence of a state: the EP (or Bethe) free energy and the variational lower bound as functions of the messages, and
 * the factors that give no part of the EP evidence at a state, with why.
 */

import { evidenceOrNaN } from '../core/_messages.ts'
import { ImproperMessageError } from '../core/errors.ts'
import type { CompiledGraph } from '../core/graph.ts'
import { takesPower } from '../core/types.ts'
import { argsOf } from './_arguments.ts'
import { rawBelief, rawCavity, readable } from './_cavities.ts'
import { beliefOf } from './cavity.ts'
import type { EngineSettings, MpState } from './state.ts'

/**
 * The power a factor is taken to at a fixed point: its own, or, under an alpha-divergence projection, `alpha` for a factor that takes a
 * power and has none of its own (the update `alphaProposal` makes of it: the fixed points of D_alpha updates are those of power EP with
 * every such factor at the power `alpha`, Minka 2004, 2005), so the energy of an alpha run is the power EP energy at those powers.
 */
function powerOfFactor(
  f: CompiledGraph['factors'][number],
  projection: Pick<EngineSettings['projection'], 'alpha'>,
): number {
  const alpha = projection.alpha
  return alpha !== undefined && alpha !== 1 && f.power === 1 && takesPower(f.operator, f.params) ? alpha : f.power
}

/**
 * The EP evidence of one factor against its cavity. The one place an improper cavity (a message with no density, which makes the
 * integral of the factor diverge) is caught, for every operator, whether or not it wraps its own rule (`evidence.improperIsNaN`): the
 * factor then has no evidence, `NaN`, and the run goes on (`evidenceOf` is `NaN`, `evidenceDefects` names the factor).
 */
function factorEvidence(g: CompiledGraph, s: Pick<EngineSettings, 'projection'>, state: MpState, fi: number): number {
  const f = g.factors[fi] as CompiledGraph['factors'][number]
  const power = powerOfFactor(f, s.projection)
  const args = readable(f.operator, rawCavity(g, state.messages, fi, power), 0).args
  return evidenceOrNaN(
    () =>
      f.operator.logEvidence?.({
        mode: s.projection.hook,
        to: 0,
        args,
        params: f.params,
        power,
        memo: new Map(),
        ...(s.projection.numeric === undefined ? {} : { rule: s.projection.numeric }),
      }) ?? Number.NaN,
  )
}

/**
 * The evidence estimate or variational bound of a state: for moment matching and sum-product the EP free energy
 * (`sum_f log Z_f + sum_v (1 - degree_v) A(belief_v)`, which on a tree is the log evidence and for loopy BP the
 * Bethe approximation), for the expected-log rule the variational lower bound (`sum_f E_q[log f] + sum_v H(q_v)`).
 *
 * Factors with a power p contribute `log Z_a / p` and the variable terms use the sum of the inverse powers (power EP). An
 * alpha-divergence run has the energy of its own fixed points, which are those of power EP with every factor that takes a power at
 * the power alpha: so those factors count at alpha.
 *
 * @param g - The compiled graph.
 * @param s - What fixes the estimate: the engine settings of a run, or just `{ projection }` (the estimate depends on nothing else).
 * @param state - A state of a run.
 * @returns The estimate, or NaN when some factor has no way to give its part or a belief has no density yet (a state with flat
 * messages, under any projection).
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, evidenceOf, gaussianPrior, klqpProjection, vmpAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * Number.isNaN(evidenceOf(g, { projection: klqpProjection }, runAlgorithm(vmpAlgorithm(g), { maxSteps: 0 }).state)) // => true
 * ```
 */
export function evidenceOf(g: CompiledGraph, s: Pick<EngineSettings, 'projection'>, state: MpState): number {
  let total = 0
  if (s.projection.evidence === 'ep-energy') {
    // The power EP energy: sum_a (1 / p_a) log Z_a + sum_v (1 - sum_{a at v} 1 / p_a) A(belief_v), with Z_a the mass of
    // the factor to the power p_a against its cavity (the usual EP energy when every power is 1).
    const inversePowers = new Array<number>(g.variables.length).fill(0)
    for (const [fi, f] of g.factors.entries()) {
      if (f.operator.logEvidence === undefined) return Number.NaN
      const power = powerOfFactor(f, s.projection)
      total += factorEvidence(g, s, state, fi) / power
      for (const p of f.ports) {
        if (p.edge >= 0) inversePowers[p.variable as number] = (inversePowers[p.variable as number] ?? 0) + 1 / power
      }
    }
    for (const [i, v] of g.variables.entries()) {
      if (v.observed !== undefined) continue
      total += (1 - (inversePowers[i] ?? 0)) * v.family.logPartition(rawBelief(g, state.messages, i))
    }
    return total
  }
  for (const [fi, f] of g.factors.entries()) {
    if (f.operator.expectedLog === undefined) return Number.NaN
    const args = argsOf(g, state.messages, fi, s.projection)
    try {
      total += f.operator.expectedLog({ mode: 'vmp', to: 0, args, params: f.params, power: f.power })
    } catch (err) {
      // A belief with no density yet (flat messages at the start of a run) has no expectation: no bound, as for EP.
      if (err instanceof ImproperMessageError) return Number.NaN
      throw err
    }
  }
  for (const [i, v] of g.variables.entries()) {
    // A defined variable is a function of the others under the mean field: its entropy is not a term of the bound.
    if (v.observed !== undefined || v.definer !== undefined) continue
    const b = beliefOf(g, state.messages, i)
    if (!v.family.isProper(b)) return Number.NaN
    total += v.family.entropy(b)
  }
  return total
}

/**
 * The factors that give no part of the EP evidence at a state, with why: a factor whose cavity has no density (an `improper cavity`: the
 * integral of the factor against it diverges, which happens when another factor's site has a negative precision, as the site of a
 * scale mixture on a weight vector does, and the factor is one plate over many rows) or an operator with no evidence rule. The
 * evidence of the graph is then undefined. A rows-expanded graph (`plates: 'expand'`) has a proper cavity for each row.
 *
 * @param g - The compiled graph.
 * @param s - The projection (the estimate depends on nothing else).
 * @param state - A state of a run.
 * @returns The factors' ids and reasons, empty when every factor gives its part.
 */
export function evidenceDefects(
  g: CompiledGraph,
  s: Pick<EngineSettings, 'projection'>,
  state: MpState,
): readonly { readonly factor: string; readonly reason: 'improper cavity' | 'no evidence rule' }[] {
  const out: { factor: string; reason: 'improper cavity' | 'no evidence rule' }[] = []
  if (s.projection.evidence !== 'ep-energy') return out
  for (const [fi, f] of g.factors.entries()) {
    if (f.operator.logEvidence === undefined) {
      out.push({ factor: f.id, reason: 'no evidence rule' })
      continue
    }
    if (!Number.isFinite(factorEvidence(g, s, state, fi))) out.push({ factor: f.id, reason: 'improper cavity' })
  }
  return out
}
