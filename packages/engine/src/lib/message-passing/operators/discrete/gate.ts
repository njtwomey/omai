/**
 * The gate of Minka and Winn (2008, "Gates"): a discrete variable `z` selects which of several factors holds, each factor
 * (a branch) over its own variables, so a branch can have random variables of its own that the others do not touch (a mixture
 * with random components, a regime-switching model, zero inflation, a change point). `gateOperator` builds the operator
 * over a registry of the inner operators, because a branch is any operator of it. The message to `z` is the evidence of each
 * branch under the cavities of its variables, and the message to a variable is the moment-matched mixture of what its branches
 * (weighted by `z`'s cavity and their evidence) and the branches that do not involve it say.
 *
 * A gate takes a power (fractional EP, an alpha-divergence) when its branches are closed-form messages over one variable
 * (a conjugate likelihood such as a Poisson count on a gamma rate, a Gaussian observation): the factor to the power `p` is
 * the gate of its branches to the power `p`, each the same exponential family with its natural parameters times `p`. The
 * message to `z` is then the branch evidences to the power `1 / p`, and to a variable the moment match divided by `p`.
 * (A gate that selects among variables with a `select` factor, `y = c_z`, cannot: a point-mass constraint has no power.)
 */

import { type Json, logSumExpOf, OmaiError } from '../../../base/index.ts'
import { arg, errorText, evidenceOrNaN } from '../../core/_messages.ts'
import { etaOf, massOf, numberParam, scalarOf, valueArg } from '../../core/args.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gaussian } from '../../core/families.ts'
import { discreteEta, discreteProbs } from '../../core/families-vector.ts'
import { param, validateParams } from '../../core/params.ts'
import type {
  Arg,
  Eta,
  FactorOperator,
  Family,
  OperatorContext,
  OperatorResult,
  Params,
  Port,
} from '../../core/types.ts'

/** One branch as the parameters give it: an inner operator, its arguments (indices into the gate's variables, or constants) and its parameters. */
interface Branch {
  readonly operator: string | undefined
  readonly args: readonly (number | { readonly const: number })[]
  readonly params: Params
}

function branchesOf(params: Params): readonly Branch[] {
  const raw = params['branches']
  if (!Array.isArray(raw)) throw new ParameterError('gate: the parameter branches must be an array')
  return raw.map((b, k) => {
    const r = b as { readonly operator?: Json; readonly args?: Json; readonly params?: Json }
    const args = Array.isArray(r.args) ? r.args : []
    return {
      operator: typeof r.operator === 'string' ? r.operator : undefined,
      args: args.map((a) => {
        if (typeof a === 'number') return a
        if (a !== null && typeof a === 'object' && !Array.isArray(a) && typeof a['const'] === 'number')
          return { const: a['const'] }
        throw new ParameterError(`gate: branch ${k} has an argument that is neither a variable index nor a constant`)
      }),
      params: (r.params ?? {}) as Params,
    }
  })
}

/** The probabilities of the states of z (an observed one is a point mass). */
function stateProbabilities(a: Arg, k: number): number[] {
  if (a.value !== undefined) return Array.from({ length: k }, (_, i) => (i === Math.trunc(scalarOf(a)) ? 1 : 0))
  return a.family.kind === 'bernoulli' ? (a.family.logProbs?.(etaOf(a)) ?? []).map(Math.exp) : discreteProbs(etaOf(a))
}

/**
 * The gate over a registry of operators: `z` (discrete, or Bernoulli for two branches) and `arity` variables, whose
 * parameter `branches` lists one entry per state of `z`, each `{ operator, args, params }` with `args` indices into the
 * variables (or `{ const }` numbers) in the inner operator's own port order; a branch without an operator is the constant 1.
 *
 * @param inner - The inner operators by name.
 * @returns The `gate` operator.
 * @concept factor-operators
 * @index function The gate of Minka and Winn: a discrete variable selects among branch factors.
 * @example
 * ```ts
 * import { defaultOperators, discrete, discreteEta, gaussian, gaussianEta, messageArg, valueArg } from 'omai-engine/message-passing'
 * const ops = defaultOperators()
 * const branch = (m: number) => ({ operator: 'gaussian', args: [0, { const: m }], params: { precision: 1 } })
 * const args = [messageArg(discrete(2), discreteEta([0.5, 0.5])), valueArg(gaussian, 1)]
 * const logs = ops['gate']?.ep?.({ mode: 'ep', to: 0, args, params: { arity: 1, branches: [branch(0), branch(2)] }, power: 1 }).message
 * logs?.map((v) => v.toFixed(4)) // => ['-0.6931', '-0.6931']
 * ```
 */
export function gateOperator(inner: Readonly<Record<string, FactorOperator>>): FactorOperator {
  const innerOf = (b: Branch): FactorOperator | undefined => {
    if (b.operator === undefined) return undefined
    const op = inner[b.operator]
    if (op === undefined) throw new ParameterError(`gate: no operator '${b.operator}' in the registry`)
    return op
  }
  /** The arguments a branch is shown: the gate's cavities (or beliefs) of its variables and its constants. */
  const argsOf = (c: OperatorContext, b: Branch): Arg[] =>
    b.args.map((a) => (typeof a === 'number' ? arg(c, a + 1) : valueArg(gaussian, a.const)))
  const context = (c: OperatorContext, b: Branch, to: number, mode: 'ep' | 'vmp'): OperatorContext => ({
    mode,
    to,
    args: argsOf(c, b),
    params: b.params,
    power: 1,
    ...(c.rule === undefined ? {} : { rule: c.rule }),
  })
  /**
   * The one variable a branch can be raised to a power over: its other arguments are constants and its inner message is
   * closed form and does not depend on the cavity (a conjugate likelihood, a Gaussian observation), so the branch is
   * `exp(c + m . T(x))` in the family of its variable and its power `p` is `exp(p c + p m . T(x))`.
   */
  const powerable = (b: Branch): { readonly op: FactorOperator; readonly variable: number } | undefined => {
    const op = innerOf(b)
    const vars = b.args.filter((a): a is number => typeof a === 'number')
    const [variable] = vars
    if (op === undefined || variable === undefined || vars.length !== 1 || op.ep === undefined) return undefined
    if (op.support.ep !== 'exact' || (typeof op.readsTarget === 'function' ? true : op.readsTarget === true))
      return undefined
    return { op, variable }
  }
  const unpowerable = (b: Branch): never => {
    throw new OmaiError(
      'not-implemented',
      `gate: a power other than 1 needs branches over one variable with a closed-form message, and ${b.operator ?? 'a constant'} is not one`,
    )
  }
  /** The log evidence of a branch under the cavities of its variables, as a normalised expectation (the masses of the cavities removed). */
  const logBranch = (c: OperatorContext, b: Branch): number => {
    if (c.power !== 1 && innerOf(b) !== undefined) {
      const q = powerable(b) ?? unpowerable(b)
      const own = etaOf(arg(c, q.variable + 1))
      const family = arg(c, q.variable + 1).family
      const message = q.op.ep?.(context(c, b, b.args.indexOf(q.variable), 'ep')).message ?? []
      const once = logBranchAt(c, b)
      const base = family.logPartition(own)
      const constant = once - (family.logPartition(own.map((v, i) => v + (message[i] ?? 0))) - base)
      return c.power * constant + family.logPartition(own.map((v, i) => v + c.power * (message[i] ?? 0))) - base
    }
    return logBranchAt(c, b)
  }
  const logBranchAt = (c: OperatorContext, b: Branch): number => {
    const op = innerOf(b)
    if (op === undefined) return 0
    if (op.logEvidence === undefined) throw new OmaiError('not-implemented', `gate: ${op.name} has no logEvidence`)
    const args = argsOf(c, b)
    for (const a of args)
      if (a.eta !== undefined && !a.family.isProper(a.eta))
        throw new ImproperMessageError('gate: a cavity of a branch has no density yet')
    let total = op.logEvidence(context(c, b, 0, 'ep'))
    for (const a of args) total -= massOf(a)
    if (Number.isNaN(total)) throw new ImproperMessageError('gate: a branch has no evidence yet')
    return total
  }
  /** What a branch sends to one of the gate's variables, as the tilted natural parameters (the cavity plus the message): undefined when the branch does not involve it. */
  const sent = (c: OperatorContext, b: Branch, variable: number): Eta | undefined => {
    const op = innerOf(b)
    const local = b.args.indexOf(variable)
    if (op === undefined || local < 0) return undefined
    const ctx = context(c, b, local, 'ep')
    const own = etaOf(arg(c, variable + 1))
    let message: Eta
    if (c.power !== 1) {
      // The tilted natural parameters of the branch to the power p: the cavity plus p times the message of the closed-form branch.
      if (powerable(b) === undefined) unpowerable(b)
      return own.map((v, i) => v + c.power * ((op.ep?.(ctx).message ?? [])[i] ?? 0))
    }
    if (op.ep !== undefined) message = op.ep(ctx).message
    else if (op.tilted !== undefined) {
      const target = arg(c, variable + 1)
      message = target.family.fromMoments(op.tilted(ctx).moments).map((v, i) => v - (own[i] ?? 0))
    } else throw new OmaiError('not-implemented', `gate: ${op.name} has no expectation propagation rule`)
    return own.map((v, i) => v + (message[i] ?? 0))
  }
  return {
    name: 'gate',
    description: 'a discrete variable selects which factor holds, each branch over its own variables',
    needsDensity: true,
    ports: (params: Params): readonly Port[] => [
      { name: 'z', accepts: ['discrete', 'bernoulli'], constant: true },
      ...Array.from(
        { length: Math.trunc(numberParam(params, 'arity')) },
        (_, i): Port => ({
          name: `x${i}`,
          accepts: [
            'gaussian',
            'gaussian-mv',
            'gamma',
            'beta',
            'bernoulli',
            'discrete',
            'dirichlet',
            'normal-gamma',
            'wishart',
            'normal-wishart',
          ],
          constant: true,
        }),
      ),
    ],
    params: { arity: param.integer({ min: 1 }), branches: param.any({ example: [] }) },
    support: { ep: 'numeric', vmp: 'bound' },
    // A power (fractional EP, an alpha-divergence) is taken of the branches: each is the factor to the power p, and the gate's indicators are disjoint so the factor to the power p is the gate of the branches to the power p.
    power: (params) => branchesOf(params).every((b) => b.operator === undefined || powerable(b) !== undefined),
    check: (params, families) => {
      const states = families[0]?.kind === 'bernoulli' ? 2 : families[0]?.dim
      const branches = branchesOf(params)
      if (branches.length !== states) return `gate: z has ${states} states and there are ${branches.length} branches`
      const arity = Math.trunc(numberParam(params, 'arity'))
      for (const [k, b] of branches.entries()) {
        const op = b.operator === undefined ? undefined : inner[b.operator]
        if (b.operator !== undefined && op === undefined) return `gate: branch ${k}: no operator '${b.operator}'`
        for (const a of b.args)
          if (typeof a === 'number' && !(a >= 0 && a < arity)) return `gate: branch ${k}: variable ${a} is out of range`
        if (op !== undefined) {
          if (op.params !== undefined) {
            try {
              validateParams(`gate: branch ${k} (${op.name})`, op.params, b.params)
            } catch (e) {
              return errorText(e)
            }
          }
          const ports = typeof op.ports === 'function' ? op.ports(b.params) : op.ports
          if (ports.length !== b.args.length)
            return `gate: branch ${k}: ${op.name} has ${ports.length} ports, ${b.args.length} given`
          const fams = b.args.map((a): Family => (typeof a === 'number' ? (families[a + 1] as Family) : gaussian))
          const why = op.check?.(b.params, fams)
          if (why !== undefined) return `gate: branch ${k}: ${why}`
        }
      }
      return undefined
    },
    ep: (c): OperatorResult => {
      const branches = branchesOf(c.params)
      const k = branches.length
      const z = arg(c, 0)
      const evidence = branches.map((b) => logBranch(c, b))
      if (c.to === 0) {
        const top = Math.max(...evidence)
        if (!(top > Number.NEGATIVE_INFINITY)) throw new ImproperMessageError('gate: no branch has evidence')
        // The message that, to the power p, is the branch evidences: their logs divided by p.
        const logs = evidence.map((v) => (v - top) / c.power)
        return {
          message:
            z.family.kind === 'bernoulli'
              ? (z.family.fromLogProbs?.(logs) as Eta)
              : discreteEta(logs.map((v) => Math.exp(v))),
        }
      }
      const pz = stateProbabilities(z, k)
      const lw = evidence.map((v, i) => Math.log(pz[i] ?? 0) + v)
      const top = Math.max(...lw)
      if (!(top > Number.NEGATIVE_INFINITY)) throw new ImproperMessageError('gate: no branch has evidence')
      const w = lw.map((v) => Math.exp(v - top))
      const norm = w.reduce((a, b) => a + b, 0)
      const variable = c.to - 1
      const target = arg(c, c.to)
      const family = target.family
      const own = etaOf(target)
      const mixed = new Array<number>(family.size).fill(0)
      for (const [i, b] of branches.entries()) {
        if ((w[i] ?? 0) === 0) continue
        const tilted = sent(c, b, variable) ?? own
        const mo = family.moments(tilted)
        for (let j = 0; j < mo.length; j++) mixed[j] = (mixed[j] ?? 0) + ((w[i] ?? 0) / norm) * (mo[j] ?? 0)
      }
      return { message: family.fromMoments(mixed).map((v, i) => (v - (own[i] ?? 0)) / c.power) }
    },
    vmp: (c): OperatorResult => {
      const branches = branchesOf(c.params)
      const z = arg(c, 0)
      if (c.to === 0) {
        const logs = branches.map((b) => {
          const op = innerOf(b)
          return op?.expectedLog === undefined ? 0 : op.expectedLog(context(c, b, 0, 'vmp'))
        })
        return {
          message: z.family.kind === 'bernoulli' ? (z.family.fromLogProbs?.(logs) as Eta) : logs,
        }
      }
      const r = stateProbabilities(z, branches.length)
      const variable = c.to - 1
      const family = arg(c, c.to).family
      const out = new Array<number>(family.size).fill(0)
      for (const [i, b] of branches.entries()) {
        const op = innerOf(b)
        const local = b.args.indexOf(variable)
        if (op?.vmp === undefined || local < 0 || (r[i] ?? 0) === 0) continue
        const m = op.vmp(context(c, b, local, 'vmp')).message
        for (let j = 0; j < out.length; j++) out[j] = (out[j] ?? 0) + (r[i] ?? 0) * (m[j] ?? 0)
      }
      return { message: out }
    },
    logEvidence: (c) => {
      return evidenceOrNaN(() => {
        const branches = branchesOf(c.params)
        const pz = stateProbabilities(arg(c, 0), branches.length)
        const terms = branches.map((b, i) => Math.log(pz[i] ?? 0) + logBranch(c, b))
        let total = massOf(arg(c, 0))
        for (let i = 1; i < c.args.length; i++) total += massOf(arg(c, i))
        return total + logSumExpOf(terms)
      })
    },
    expectedLog: (c) => {
      const branches = branchesOf(c.params)
      const r = stateProbabilities(arg(c, 0), branches.length)
      let total = 0
      for (const [i, b] of branches.entries()) {
        const op = innerOf(b)
        if (op?.expectedLog === undefined || (r[i] ?? 0) === 0) continue
        total += (r[i] ?? 0) * op.expectedLog(context(c, b, 0, 'vmp'))
      }
      return total
    },
  }
}
