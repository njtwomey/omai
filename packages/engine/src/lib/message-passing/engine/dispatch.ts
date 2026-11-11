/**
 * The dispatch of one message: what a factor's operator tells one port under a projection, from the arguments it is shown
 * (the closed-form rule of the projection's hook, or the projected moments of a `tilted` function), and how an operator that
 * fails for an improper quantity is told apart from one that is blocked or has nothing to say yet.
 */

import { OmaiError } from '../../base/index.ts'
import { isFlat } from '../core/_messages.ts'
import { FLAT_CAVITY_PRECISION } from '../core/args.ts'
import { ImproperMessageError } from '../core/errors.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { Arg, Eta, OperatorContext, OperatorResult } from '../core/types.ts'
import { readable } from './_cavities.ts'
import type { Projection } from './projections.ts'

/** What an update proposes: a message, no information (a flat one), nothing (the old message stays), or a failure. */
export type FactorProposal =
  | { readonly kind: 'message'; readonly eta: Eta }
  | { readonly kind: 'flat' }
  | { readonly kind: 'blocked' }
  | { readonly kind: 'failed' }

/**
 * The message a factor's operator sends to one port under a projection, from the arguments it is shown: the closed-form
 * rule of the projection's hook when the operator has one, else (for expectation propagation) the moments of its
 * `tilted` function projected onto the family and divided by the factor's power. A port whose own incoming message has
 * no density is `blocked` for an operator that reads it, and an operator that fails for an improper quantity gives
 * `flat` (when some other message is still flat: nothing is known yet) or `failed`.
 *
 * @param f - The compiled factor.
 * @param projection - The projection (its hook is the rule asked for).
 * @param to - The port to send to.
 * @param args - The arguments of every port: the cavities (or the beliefs) and the observed values.
 * @param power - The exponent of the factor in this call (the factor's own power, or the exponent of an alpha update).
 * @returns The proposal.
 */
export function operatorMessage(
  f: CompiledGraph['factors'][number],
  projection: Projection,
  to: number,
  args: readonly Arg[],
  power: number = f.power,
): FactorProposal {
  const op = f.operator
  const target = args[to] as Arg
  const reads = typeof op.readsTarget === 'function' ? op.readsTarget(to) : op.readsTarget === true
  const shown = readable(op, args, to)
  // A flat target of an operator that needs a density is shown wide, so it is not blocked; an improper one still is.
  if (reads && !shown.widenedTarget && target.eta !== undefined && !target.family.isProper(target.eta))
    return { kind: 'blocked' }
  const ctx: OperatorContext = {
    mode: projection.hook,
    to,
    args: shown.args,
    params: f.params,
    power,
    ...(projection.numeric === undefined ? {} : { rule: projection.numeric }),
  }
  const hook = projection.hook === 'ep' ? op.ep : projection.hook === 'vmp' ? op.vmp : op.bp
  try {
    let result: OperatorResult
    if (hook !== undefined) {
      result = hook.call(op, ctx)
      // The message of a moment-matching rule is the projection divided by the cavity it was shown: against the flat cavity
      // that is the same message plus the wide one.
      if (shown.widenedTarget && projection.hook === 'ep')
        result = { ...result, message: result.message.map((v, i) => v + (i === 1 ? FLAT_CAVITY_PRECISION / power : 0)) }
    } else if (projection.hook === 'ep' && op.tilted !== undefined) {
      const t = op.tilted(ctx)
      const own = target.eta as Eta
      const projected = target.family.fromMoments(t.moments)
      result = { message: projected.map((v, i) => (v - (own[i] ?? 0)) / power) }
    } else {
      throw new OmaiError('not-implemented', `${op.name} has no ${projection.hook} message rule`)
    }
    return { kind: 'message', eta: result.message }
  } catch (err) {
    if (err instanceof ImproperMessageError) {
      const others = args.filter((a, q) => q !== to && a.eta !== undefined)
      return others.some((a) => isFlat(a.eta)) ? { kind: 'flat' } : { kind: 'failed' }
    }
    throw err
  }
}
