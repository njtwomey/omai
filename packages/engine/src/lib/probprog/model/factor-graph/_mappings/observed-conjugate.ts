/**
 * Observations whose parameter must be a latent of a conjugate family: a Bernoulli with a beta probability, a gamma or exponential
 * with a gamma rate, a mixture weight, failure counts (geometric, negative binomial) and the conjugate Poisson, binomial and
 * categorical likelihoods. The parameter is checked, the observed variable is declared and one factor is added.
 */

import type { Json } from '../../../../base/index.ts'
import type { FamilyKind } from '../../../../message-passing/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { factorName } from '../_builder.ts'
import { affOf, type Cell, isConstCell, scaledOf } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { randomTransitionRow, unwrap } from '../_idioms.ts'
import { cellsOf, constAt, paramCell } from '../_params.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import { observedRowGate } from './observed-gates.ts'

export function observedBernoulli(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
  index: number | undefined,
): string | null {
  const probs = node.dist.params['probs']
  const logits = node.dist.params['logits']
  const link = (op: string, arg: ExprSpec, params: Record<string, Json> = {}): string | null => {
    const factor = factorName(cx.modelOps, op)
    cx.g.need(factor)
    const c = cellsOf(cx, node, `link:${op}`, () => evalSym(cx, arg))[i] as Cell
    if (isConstCell(c)) return null
    const x = cx.g.materialise(affOf(c), node.name, index) as string
    cx.g.addVariable({ name, family: { kind: 'bernoulli' }, observed: y })
    cx.g.addFactor({ operator: factor, args: [name, x], params }, name)
    return name
  }
  if (logits !== undefined) return link('sigmoid', logits)
  if (probs === undefined) throw new Refusal('structure', `'${node.name}': give probs or logits`)
  const e = unwrap(cx.nodes, probs)
  if (e.kind === 'op' && e.op === 'probit') return link('probit', e.args[0] as ExprSpec)
  if (e.kind === 'op' && e.op === 'sigmoid') return link('sigmoid', e.args[0] as ExprSpec)
  if (e.kind === 'op' && e.op === 'gt') {
    const threshold = cx.evalNamed(e.args[1] as ExprSpec)
    const cell = cx.over(cx.fromNamed(threshold), node.dims)[i] as Cell & { kind: 'aff' }
    return link('gt', e.args[0] as ExprSpec, { threshold: cell.aff.offset })
  }
  const p = paramCell(cx, node, 'probs', i)
  if (isConstCell(p)) return null
  if (p.kind === 'rv' && p.family === 'beta') {
    cx.g.addVariable({ name, family: { kind: 'bernoulli' }, observed: y })
    cx.g.addFactor({ operator: 'bernoulli', args: [name, p.variable] }, name)
    return name
  }
  throw new Refusal(
    'operator',
    `the probability of '${node.name}' is not a link of an affine value (probit, sigmoid, a comparison) or a beta latent: no message rule`,
  )
}

/**
 * An observed exponential or gamma value (a known shape) whose rate is a gamma latent or a known multiple of one: the conjugate
 * `gamma-observed` factor, a gamma message to the rate (a waiting time with a gamma prior on the rate).
 */
export function observedGammaRate(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  fam: string,
  name: string,
  y: number,
  i: number,
): string | null {
  if (!(y > 0)) throw new Refusal('data', `'${node.name}' is ${fam}: the value ${y} must be positive`)
  const shape = fam === 'gamma' ? constAt(cx, node, node.dist.params['concentration'] as ExprSpec, i) : 1
  const target = paramCell(cx, node, 'rate', i)
  if (isConstCell(target)) return null
  const tau = scaledOf(target)
  if (tau === undefined || tau.offset !== 0 || !(tau.scale > 0))
    throw new Refusal(
      'operator',
      `the rate of '${node.name}' must be a gamma latent or a known multiple of one for the conjugate '${fam}' operator, it is not`,
    )
  cx.g.need('gamma-observed')
  cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
  cx.g.addFactor(
    {
      operator: 'gamma-observed',
      args: [name, tau.variable],
      params: { shape, ...(tau.scale === 1 ? {} : { scale: tau.scale }) },
    },
    name,
  )
  return name
}

/**
 * An observed two-component mixture with known means and scale and a beta weight (`normalPair`, the switch summed out): the
 * `mixture-weight` factor on the weight, with the likelihood of the value under each component.
 */
export function observedMixtureWeight(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
): string | null {
  const target = paramCell(cx, node, 'weight', i)
  if (isConstCell(target)) return null
  if (target.kind !== 'rv' || target.family !== 'beta')
    throw new Refusal(
      'operator',
      `the weight of '${node.name}' must be a beta latent for the 'mixture-weight' operator`,
    )
  const [low, high, scale] = ['low', 'high', 'scale'].map((k) => constAt(cx, node, node.dist.params[k] as ExprSpec, i))
  const density = (m: number): number =>
    Math.exp(-0.5 * ((y - m) / (scale as number)) ** 2) / ((scale as number) * Math.sqrt(2 * Math.PI))
  cx.g.need('mixture-weight')
  cx.g.addFactor(
    {
      operator: 'mixture-weight',
      args: [target.variable],
      params: { l1: density(high as number), l0: density(low as number) },
    },
    name,
  )
  return null
}

/**
 * An observed geometric (trials up to the first success) or negative binomial (failures before the `n`-th success, `n` known)
 * count whose success probability is a beta latent: the conjugate `negative-binomial-beta` factor.
 */
export function observedFailures(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  fam: string,
  name: string,
  y: number,
  i: number,
): string | null {
  const geometric = fam === 'geometric'
  const failures = geometric ? y - 1 : y
  if (!(failures >= 0 && Number.isInteger(failures)))
    throw new Refusal('data', `'${node.name}' is ${fam}: the count ${y} is not on its support`)
  const successes = geometric ? 1 : constAt(cx, node, node.dist.params['n'] as ExprSpec, i)
  const target = paramCell(cx, node, 'probs', i)
  if (isConstCell(target)) return null
  if (target.kind !== 'rv' || target.family !== 'beta')
    throw new Refusal(
      'operator',
      `the probs of '${node.name}' must be a beta latent for the conjugate '${fam}' operator, it is not`,
    )
  cx.g.need('negative-binomial-beta')
  cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: failures })
  cx.g.addFactor({ operator: 'negative-binomial-beta', args: [name, target.variable], params: { successes } }, name)
  return name
}

export function observedConjugate(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  fam: string,
  name: string,
  y: number,
  i: number,
): string | null {
  const param = fam === 'poisson' ? 'rate' : 'probs'
  const expr = node.dist.params[param]
  if (expr === undefined) throw new Refusal('structure', `'${node.name}': the ${param} is missing`)
  // A word read through a discrete index of Dirichlet rows (`phi.at(z)`, a topic model): a gate over the index.
  if (fam === 'categorical' && randomTransitionRow(cx.idiom, expr) !== undefined)
    return observedRowGate(cx, node, name, y, i, expr)
  let exposure = 1
  let target: Cell
  const e = unwrap(cx.nodes, expr)
  if (fam === 'poisson' && e.kind === 'op' && e.op === 'mul') {
    const [a, b] = e.args as [ExprSpec, ExprSpec]
    const [k, r] = cx.isConst(a) ? [a, b] : [b, a]
    if (!cx.isConst(k)) throw new Refusal('operator', `the rate of '${node.name}' is a product of two random values`)
    exposure = (cx.over(cx.fromNamed(cx.evalNamed(k)), node.dims)[i] as Cell & { kind: 'aff' }).aff.offset
    target = cx.over(evalSym(cx, r), node.dims)[i] as Cell
  } else target = paramCell(cx, node, param, i)
  if (isConstCell(target)) return null
  const want: FamilyKind = fam === 'poisson' ? 'gamma' : fam === 'binomial' ? 'beta' : 'dirichlet'
  if (target.kind !== 'rv' || target.family !== want)
    throw new Refusal(
      'operator',
      `the ${param} of '${node.name}' must be a ${want} latent for the conjugate '${fam}' operator, it is not`,
    )
  if (fam === 'poisson') {
    cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    cx.g.addFactor({ operator: 'poisson', args: [name, target.variable], params: { exposure } }, name)
  } else if (fam === 'binomial') {
    const trials = (
      cx.over(cx.fromNamed(cx.evalNamed(node.dist.params['n'] as ExprSpec)), node.dims)[i] as Cell & {
        kind: 'aff'
      }
    ).aff.offset
    cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    cx.g.addFactor({ operator: 'binomial', args: [name, target.variable], params: { n: trials } }, name)
  } else {
    const size = cx.g.sizeOf(target.variable)
    cx.g.addVariable({ name, family: { kind: 'discrete', size }, observed: y })
    cx.g.addFactor({ operator: 'discrete', args: [name, target.variable] }, name)
  }
  return name
}
