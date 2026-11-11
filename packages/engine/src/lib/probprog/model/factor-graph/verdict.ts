/**
 * The verdict of a compilation for a message-passing method: `factorGraphVerdict` says whether a bound model has a factor
 * graph that `ep`, `vmp` or `bp` can run and exactly why not (the issues, the operators that lack the rule), with
 * `operatorsWithout` and `MessageRule`, and `compileFactorGraph` indexes the compiled graph for the engine.
 */

import { OmaiError } from '../../../base/index.ts'
import {
  type CompiledGraph,
  compileGraph,
  type FactorOperator,
  defaultOperators as factorOperators,
} from '../../../message-passing/index.ts'
import type { FactorGraphCompilation } from './types.ts'

/** The message rule a method asks of every operator of a graph. */
export type MessageRule = 'ep' | 'vmp' | 'bp'

/**
 * The factors of a compiled graph whose operator has no message rule for a method (an operator with only an EP rule, such as a hard comparison, cannot
 * run variational message passing; one with no `tilted` and no `ep` cannot run EP), with the operator's name: what a
 * method's capability check reports.
 *
 * @param g - The compiled graph.
 * @param rule - `ep`, `vmp` or `bp`.
 * @returns The names of the operators that lack the rule, without repeats.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators } from 'omai-engine/message-passing'
 * import { operatorsWithout } from 'omai-engine/probprog/model'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'b', family: { kind: 'bernoulli' }, observed: 1 }], factors: [{ operator: 'greater-than', args: ['b', 'x'], params: { threshold: 0 } }, { operator: 'prior', args: ['x'], params: { eta: [0, 1] } }] }, defaultOperators())
 * [operatorsWithout(g, 'ep'), operatorsWithout(g, 'vmp')] // => [[], ['greater-than']]
 * ```
 */
export function operatorsWithout(g: CompiledGraph, rule: MessageRule): string[] {
  const out = new Set<string>()
  for (const f of g.factors) {
    const op = f.operator
    const has = rule === 'ep' ? op.ep !== undefined || op.tilted !== undefined : op[rule] !== undefined
    if (!has || op.unsupported?.(rule, f.params) !== undefined) out.add(op.name)
  }
  return [...out]
}

/**
 * Whether a bound model has a factor graph a message-passing method can run, and exactly why not: the static check behind
 * `ep`, `vmp` and `bp` as inference methods. It reads the compilation (the issues, with node and reason) and, for a rule, the
 * operators of the compiled graph that lack it.
 *
 * @param c - The compilation.
 * @param g - The compiled graph, when the compilation has a graph.
 * @param rule - The method's rule.
 * @returns `ok` with a reason, or the reasons it does not apply and the missing features.
 * @example
 * ```ts
 * import { defineModel, factorGraphVerdict, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), b.latent('s', normal(1, 1))), b.data('y', { dims: ['row'] })) })
 * factorGraphVerdict(toFactorGraph(m.bind({ y: [1, 2] })), undefined, 'ep').ok // => false
 * ```
 */
export function factorGraphVerdict(
  c: FactorGraphCompilation,
  g: CompiledGraph | undefined,
  rule: MessageRule,
): { readonly ok: boolean; readonly reason: string; readonly missing: readonly string[] } {
  if (c.graph === undefined || g === undefined) {
    const reasons = c.issues.map((i) => `${i.node}: ${i.reason}`)
    const missing = [
      ...new Set(
        c.issues.map((i) =>
          i.kind === 'opaque' || i.kind === 'family'
            ? 'exponential-family factors'
            : `a factor graph without ${i.kind} issues`,
        ),
      ),
    ]
    return { ok: false, reason: reasons.join('; '), missing }
  }
  const lacking = operatorsWithout(g, rule)
  if (lacking.length > 0) {
    const why = lacking.flatMap((name) => {
      const factors = g.factors.filter((f) => f.operator.name === name)
      const note =
        factors[0]?.operator.noRule?.[rule] ??
        factors.map((f) => f.operator.unsupported?.(rule, f.params)).find((v) => v !== undefined)
      return note === undefined ? [] : [`${name}: ${note}`]
    })
    return {
      ok: false,
      reason: `the operator${lacking.length > 1 ? 's' : ''} ${lacking.join(', ')} ${lacking.length > 1 ? 'have' : 'has'} no ${rule === 'ep' ? 'expectation propagation' : rule === 'vmp' ? 'variational' : 'sum-product'} message rule${why.length > 0 ? ` (${why.join('; ')})` : ''}`,
      missing: [`${rule} message rules`],
    }
  }
  return {
    ok: true,
    reason: `a factor graph of ${g.variables.length} variables and ${g.factors.length} factors`,
    missing: [],
  }
}

/**
 * The compiled graph (indexed, ready for the engine) of a compilation, against the engine's operators or a registry of
 * your own.
 *
 * @param c - The compilation.
 * @param operators - The factor operators by name (default the engine's `defaultOperators()`).
 * @returns The compiled graph.
 * @throws OmaiError with code `value` when the compilation has issues (no graph), naming each node and reason.
 * @concept model-spec
 * @example
 * ```ts
 * import { compileFactorGraph, defineModel, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), 1), b.data('y', { dims: ['row'] })) })
 * const g = compileFactorGraph(toFactorGraph(m.bind({ y: [1, 2, 3] })))
 * [g.variables.length, g.factors.length, g.edges.length] // => [4, 4, 4]
 * ```
 */
export function compileFactorGraph(
  c: FactorGraphCompilation,
  operators: Readonly<Record<string, FactorOperator>> = factorOperators(),
): CompiledGraph {
  if (c.graph === undefined)
    throw new OmaiError(
      'value',
      `'${c.name}' does not compile to a factor graph: ${c.issues.map((i) => `${i.node}: ${i.reason}`).join('; ')}`,
    )
  return compileGraph(c.graph, operators)
}
