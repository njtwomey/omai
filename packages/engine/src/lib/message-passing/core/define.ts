/**
 * The constructor of a factor operator, `defineOperator`, and the two kinds of rule it takes. A `message(fn)` rule is a
 * closed-form message (a pushforward, an exact conjugate rule, an expected-log rule) that the operator computes itself. A
 * `site(tilt)` rule is what an operator that integrates numerically says: its tilted distribution at the target port (the
 * moments, or the mean and variance of a Gaussian port), and nothing else. The projection onto the port's family, the
 * quotient by the cavity and the division by the power of fractional EP are done here, once (`projectSite`), and the engine's
 * fallback for an operator with only `tilted` calls the same function, so no operator writes `siteMessage` or `perPower` by
 * hand. `defineOperator` also derives `support` from the kinds of the rules, wraps the evidence rule so that a message with
 * no density yet is a `NaN` and not an error, and refuses a contradictory declaration when the operator is built (a rule
 * declared twice, a `noRule` for a rule the operator has), which is what the registry test checked after the fact.
 */

import { evidenceOrNaN, perPower } from './_messages.ts'
import { etaOf, gaussianSiteMessage, siteMessage } from './args.ts'
import { ParameterError } from './errors.ts'
import type { ParamSpecOf } from './params.ts'
import type {
  Arg,
  Eta,
  FactorOperator,
  Family,
  Mode,
  OperatorContext,
  OperatorResult,
  Params,
  Port,
  Support,
  TiltedMoments,
} from './types.ts'

/**
 * What a `site(tilt)` rule returns: the tilted distribution at the target port, as the expected sufficient statistics of the
 * target's family (`moments`: a `TiltedMoments` is one), or, for a Gaussian port, its mean and variance (the projection of a
 * Gaussian from them is then exact in the form the operators always used, not through the second moment), or `{ natural }`, the
 * natural parameters of the tilted member when the operator makes the projection itself (a dense Gaussian vector from a Cholesky
 * factor), or `{ flat: true }` when the
 * factor says nothing to this port yet (the flat message of the family, whatever the cavity, and not divided by the power).
 */
export type Tilt =
  | { readonly moments: Eta }
  | { readonly mean: number; readonly variance: number }
  | { readonly natural: Eta }
  | { readonly flat: true }

/** A rule an operator computes itself: the message to the port `c.to`, with what kind of rule it is. */
export interface MessageRule {
  readonly kind: 'message'
  readonly message: (c: OperatorContext) => OperatorResult
  readonly support: Support
}

/** A rule given as the tilted distribution: the message is made from it here (`projectSite`). */
export interface SiteRule {
  readonly kind: 'site'
  readonly tilt: (c: OperatorContext) => Tilt
  readonly support: Support
}

/**
 * A rule given as the tilted moments and mass of the factor against the cavities: the engine projects them (`ruleMessage`, which
 * calls `projectSite`) when no `ep` hook exists, so the operator has a `tilted` member and no `ep` one. Always `numeric`.
 */
export interface TiltedRuleDeclaration {
  readonly kind: 'tilted'
  readonly tilted: (c: OperatorContext) => TiltedMoments
}

/**
 * A rule that is the factor's tilted moments and mass, projected by the engine.
 *
 * @param tilted - The moments of the tilted distribution at port `c.to` and its log mass.
 * @returns The rule.
 */
export function tiltedRule(tilted: (c: OperatorContext) => TiltedMoments): TiltedRuleDeclaration {
  return { kind: 'tilted', tilted }
}

/**
 * A closed-form rule: the operator computes the message to port `c.to` itself.
 *
 * @param message - The rule.
 * @param support - What it is: `exact` (default), `bound` (a variational lower bound) or `numeric` (a quadrature the operator runs).
 * @returns The rule.
 */
export function messageRule(message: (c: OperatorContext) => OperatorResult, support: Support = 'exact'): MessageRule {
  return { kind: 'message', message, support }
}

/**
 * A rule given as the tilted distribution at the target port. The operator returns the moments (or the mean and variance of a
 * Gaussian port) of its factor times the cavity; the message is the family's projection of them quotiented by the cavity, divided
 * by the power when the operator takes one (`traits.power`).
 *
 * @param tilt - The tilted distribution at port `c.to`, from the cavities in `c.args`.
 * @param support - `numeric` (default: quadrature, a grid, a sum over samples), or `exact` when the tilted moments are closed form
 * (a Gaussian mixture times a Gaussian): the projection is still the engine's.
 * @returns The rule.
 */
export function siteRule(tilt: (c: OperatorContext) => Tilt, support: Support = 'numeric'): SiteRule {
  return { kind: 'site', tilt, support }
}

/**
 * The message that turns the target's cavity into a tilted distribution: the projection of the tilt onto the family of the
 * target, divided by the cavity (their quotient in natural parameters) and by the power of the factor. The one place this is done;
 * the engine's fallback for an operator with only `tilted` and the `ep` rule `defineOperator` derives from a `site` rule both call it.
 *
 * @param target - The argument at the port the message is for (its `eta` is the cavity).
 * @param tilt - The tilted distribution there.
 * @param power - The exponent of the factor (1: the site as it is).
 * @returns The message.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, projectSite } from 'omai-engine/message-passing'
 * projectSite(messageArg(gaussian, gaussianEta(0, 1)), { mean: 1, variance: 0.5 }, 1) // => [2, 1]
 * ```
 */
export function projectSite(target: Arg, tilt: Tilt, power: number): Eta {
  if ('flat' in tilt) return [...target.family.flat]
  const cavity = etaOf(target)
  const site =
    'moments' in tilt
      ? siteMessage(target.family, cavity, tilt.moments)
      : 'natural' in tilt
        ? tilt.natural.map((v, i) => v - (cavity[i] ?? 0))
        : gaussianSiteMessage(cavity, tilt.mean, tilt.variance)
  return perPower(site, power)
}

/** The traits of an operator that are not rules: what the engine needs to know to call them. */
export interface OperatorTraits {
  /** True, or a function of the port, when the message to a port depends on that port's own incoming message (see `FactorOperator.readsTarget`). */
  readonly readsTarget?: boolean | ((to: number) => boolean)
  /** True when a flat Gaussian cavity must be shown wide to the operator (see `FactorOperator.needsDensity`). */
  readonly needsDensity?: true
  /**
   * True when the operator takes a power (see `FactorOperator.power`), or a function of the parameters. A `site` rule divides its
   * message by the power only then.
   */
  readonly power?: boolean | ((params: Params) => boolean)
  /** The port the factor defines as a deterministic function of the others (see `FactorOperator.defines`). */
  readonly defines?: number
}

/** The declaration `defineOperator` takes. */
export interface OperatorDeclaration {
  readonly name: string
  readonly description: string
  /** The arguments, in order; a function of the parameters for a factor of variable arity. */
  readonly ports: readonly Port[] | ((params: Params) => readonly Port[])
  /** The JSON parameters the operator reads (`param.number()`, ...); `{}` for none. */
  readonly params?: ParamSpecOf
  /** A reason the factor cannot be built from these parameters and families, or `undefined`. */
  readonly check?: (params: Params, families: readonly Family[]) => string | undefined
  /** The message rules by mode; `support` follows from their kinds. */
  readonly rules: {
    readonly ep?: SiteRule | MessageRule | TiltedRuleDeclaration
    readonly vmp?: MessageRule
    readonly bp?: MessageRule
  }
  /**
   * What the evidence estimate (`ep`: log of the integral of the factor against the cavities) and the variational bound (`vmp`: the
   * expected log factor) need. The engine catches an improper cavity for every operator when it reads the evidence of a state
   * (`evidenceOf`, `evidenceDefects`: the factor then has no evidence, `NaN`, never an error), so no operator needs to ask for it. With
   * `improperIsNaN` the `ep` rule is also wrapped here, for an operator whose `logEvidence` is called outside the engine (a direct
   * call then gives `NaN` where it would throw `ImproperMessageError`); no operator of the library sets it.
   */
  readonly evidence?: {
    readonly ep?: (c: OperatorContext) => number
    readonly vmp?: (c: OperatorContext) => number
    readonly improperIsNaN?: true
  }
  readonly traits?: OperatorTraits
  /** Why a rule has no message, in words (see `FactorOperator.noRule`). */
  readonly noRule?: Readonly<Partial<Record<Mode, string>>>
  /** A reason a factor has no message under a rule from its parameters (see `FactorOperator.unsupported`). */
  readonly unsupported?: (mode: Mode, params: Params) => string | undefined
}

/**
 * Build a factor operator from its declaration: the members of `FactorOperator` the engine reads, with `support` derived from the
 * kinds of the rules, the `ep` hook derived from a `site` rule (the tilted distribution projected, divided by the cavity and by the
 * power: `projectSite`), and the evidence rule wrapped, when asked, so that a message with no density yet gives `NaN`. Throws a `ParameterError`
 * when the declaration contradicts itself: no rule at all, a `noRule` for a mode that has a rule, an evidence rule for a mode
 * without one, a `defines` that is not a port.
 *
 * @param declaration - The operator.
 * @returns The operator, as `defaultOperators()` holds it.
 * @concept factor-operators
 * @index function Build a factor operator from its declaration, with support derived.
 * @example
 * ```ts
 * import { defineOperator, gaussian, gaussianEta, messageArg, siteRule } from 'omai-engine/message-passing'
 * const op = defineOperator({
 *   name: 'shifted',
 *   description: 'a made-up site',
 *   ports: [{ name: 'x', accepts: ['gaussian'] }],
 *   params: {},
 *   rules: { ep: siteRule(() => ({ mean: 1, variance: 0.5 })) },
 *   traits: { readsTarget: true },
 * })
 * op.support // => { ep: 'numeric' }
 * op.ep?.({ mode: 'ep', to: 0, args: [messageArg(gaussian, gaussianEta(0, 1))], params: {}, power: 1 }).message // => [2, 1]
 * ```
 */
export function defineOperator(declaration: OperatorDeclaration): FactorOperator {
  const { name, rules, evidence, traits, noRule } = declaration
  const fail = (why: string): never => {
    throw new ParameterError(`defineOperator ${name}: ${why}`)
  }
  if (name === '') fail('the name is empty')
  if (rules.ep === undefined && rules.vmp === undefined && rules.bp === undefined) fail('it has no rule')
  for (const mode of ['ep', 'vmp', 'bp'] as const)
    if (noRule?.[mode] !== undefined && rules[mode] !== undefined)
      fail(`noRule.${mode} is given and it has a ${mode} rule`)
  if (evidence?.ep !== undefined && rules.ep === undefined && rules.bp === undefined)
    fail('an ep evidence rule without an ep or bp rule')
  if (evidence?.vmp !== undefined && rules.vmp === undefined) fail('a vmp evidence rule without a vmp rule')
  if (traits?.defines !== undefined && !Number.isInteger(traits.defines)) fail('defines is not a port index')
  const ep = rules.ep
  const takesPower = traits?.power === true
  const epHook =
    ep === undefined || ep.kind === 'tilted'
      ? undefined
      : ep.kind === 'message'
        ? ep.message
        : (c: OperatorContext): OperatorResult => ({
            message: projectSite(c.args[c.to] as Arg, ep.tilt(c), takesPower ? c.power : 1),
          })
  const support: Partial<Record<Mode, Support>> = {
    ...(ep === undefined ? {} : { ep: ep.kind === 'tilted' ? ('numeric' as const) : ep.support }),
    ...(rules.vmp === undefined ? {} : { vmp: rules.vmp.support }),
    ...(rules.bp === undefined ? {} : { bp: rules.bp.support }),
  }
  const logEvidence = evidence?.ep
  return {
    name,
    description: declaration.description,
    ports: declaration.ports,
    ...(declaration.params === undefined ? {} : { params: declaration.params }),
    support,
    ...(traits?.readsTarget === undefined ? {} : { readsTarget: traits.readsTarget }),
    ...(traits?.needsDensity === true ? { needsDensity: true } : {}),
    ...(traits?.power === undefined ? {} : { power: traits.power }),
    ...(traits?.defines === undefined ? {} : { defines: traits.defines }),
    ...(noRule === undefined ? {} : { noRule }),
    ...(declaration.unsupported === undefined ? {} : { unsupported: declaration.unsupported }),
    ...(declaration.check === undefined ? {} : { check: declaration.check }),
    ...(epHook === undefined ? {} : { ep: epHook }),
    ...(ep?.kind === 'tilted' ? { tilted: ep.tilted } : {}),
    ...(rules.vmp === undefined ? {} : { vmp: rules.vmp.message }),
    ...(rules.bp === undefined ? {} : { bp: rules.bp.message }),
    ...(logEvidence === undefined
      ? {}
      : {
          logEvidence:
            evidence?.improperIsNaN === true
              ? (c: OperatorContext): number => evidenceOrNaN(() => logEvidence(c))
              : logEvidence,
        }),
    ...(evidence?.vmp === undefined ? {} : { expectedLog: evidence.vmp }),
  }
}
