/**
 * Observed Gaussians and their relatives: a known scale (`observedGaussian`), a scale that is `exp` of a Gaussian (the stochastic
 * volatility observation), a random scale, a gamma precision, the heavy-tailed densities and their scale mixtures
 * (`observedHeavy`), the outlier switch, the normal-gamma rows of a pair the plan found, and the rest of the Gaussians whose
 * parameters are random.
 */

import type { Json } from '../../../../base/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { precisionFactor } from '../_builder.ts'
import { type Atom, affOf, type Cell, isConstCell, lookup, scaledOf } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { expArgument, logVariableOfCell, precisionExpr, stripClip, unwrap } from '../_idioms.ts'
import { NOT_MINE, type NotMine, probe } from '../_mapping.ts'
import { cellsOf, paramCell, paramIsConst, randomPrecisions } from '../_params.ts'
import type { NormalGammaPlan } from '../_plan.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import { hiddenSwitch, observedSwitch } from './observed-gates.ts'
import { emitSite, observedPair } from './observed-numeric.ts'

/** The observed Gaussian rows of a normal-gamma pair: a `gaussian-normal-gamma` factor on the shared variable per row. */
export function observedNormalGamma(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  ng: { readonly plan: NormalGammaPlan; readonly multiple: number },
  values: readonly number[],
): void {
  const vars: (string | null)[] = []
  values.forEach((y, i) => {
    if (Number.isNaN(y)) {
      vars.push(null)
      return
    }
    const name = cx.elementName(node.name, node.dims, i)
    cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    cx.g.addFactor(
      {
        operator: 'gaussian-normal-gamma',
        args: [name, ng.plan.variable],
        ...(ng.multiple === 1 ? {} : { params: { precision: ng.multiple } }),
      },
      name,
    )
    vars.push(name)
  })
  cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
}

/** An observed Gaussian with a known scale around an affine mean, named by the node's parameters (`loc`/`scale`, or a log-normal's `mu`/`sigma`). */
export function observedGaussian(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
  index: number | undefined,
  locName: string,
  scaleName: string,
): string | null {
  const scale = paramCell(cx, node, scaleName, i)
  if (!isConstCell(scale)) throw new Refusal('operator', `the scale of '${node.name}' is not known`)
  if (!(scale.aff.offset > 0))
    throw new Refusal('data', `the scale of '${node.name}' is ${scale.aff.offset}: it must be positive`)
  const precision = 1 / scale.aff.offset ** 2
  const loc = paramCell(cx, node, locName, i)
  if (isConstCell(loc)) return null
  const aff = affOf(loc)
  const terms = [...aff.terms].filter(([, c]) => c !== 0)
  const comps = terms.map(([k]) => cx.g.atoms.get(k) as Atom)
  const oneVector =
    aff.offset === 0 &&
    comps.length > 0 &&
    comps.every((a) => a.kind === 'comp' && a.vector === (comps[0] as Atom & { kind: 'comp' }).vector)
  cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
  if (oneVector) {
    const first = comps[0] as Atom & { kind: 'comp' }
    const a = new Array<number>(first.size).fill(0)
    for (const [k, c] of terms) a[(cx.g.atoms.get(k) as Atom & { kind: 'comp' }).index] = c
    cx.g.addFactor({ operator: 'linear-regression-row', args: [name, first.vector], params: { a, precision } }, name)
  } else {
    const m = cx.g.materialise(aff, node.name, index)
    cx.g.addFactor({ operator: 'gaussian', args: [name, m as string], params: { precision } }, name)
  }
  return name
}

/**
 * A heavy-tailed likelihood of an affine predictor with a known scale: `studentT` (the degrees of freedom known),
 * `laplace`, or `normalOutliers` (a mixture of two normal noise levels). One row factor on the packed weights when the
 * predictor is a single dot product with one vector, else a site factor on an auxiliary predictor variable. The outcome
 * is a parameter of the factor, so the observed value has no variable of its own.
 */
export function observedHeavy(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  fam: string,
  y: number,
  i: number,
  index: number | undefined,
): string | null {
  const known = (p: string): number => {
    const c = paramCell(cx, node, p, i)
    if (!isConstCell(c))
      throw new Refusal(
        'operator',
        `the ${p} of '${node.name}' is random: the ${fam} likelihood has a message rule for a known ${p} only`,
      )
    return c.aff.offset
  }
  if (fam === 'laplace' && !isConstCell(paramCell(cx, node, 'scale', i)))
    return observedPair(cx, node, 'laplace-scale', y, i, index)
  if (fam === 'studentT' && !isConstCell(paramCell(cx, node, 'df', i)))
    return observedPair(cx, node, 'student-t-df', y, i, index)
  if (fam === 'normalOutliers' && ['scale', 'outlier'].some((p) => !paramIsConst(cx, node, p)))
    return observedOutlierSwitch(cx, node, y, i, index)
  const scale = known('scale')
  if (!(scale > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${scale}: it must be positive`)
  let params: Record<string, Json>
  let row: string
  let site: string
  if (fam === 'studentT') {
    params = { y, scale, df: known('df') }
    row = 'student-t-row'
    site = 'student-t-site'
  } else if (fam === 'laplace') {
    params = { y, scale }
    row = 'laplace-row'
    site = 'laplace-site'
  } else {
    const outlier = known('outlier')
    params = { weights: [1 - outlier, outlier], means: [y, y], sds: [scale, scale * known('width')] }
    row = 'mixture-row'
    site = 'mixture-site'
  }
  return emitSite(cx, node, paramCell(cx, node, 'loc', i), i, index, row, site, params)
}

/**
 * An observation `normalOutliers(loc, scale, share, width)` with a random scale (the precision idiom of a gamma latent) or a
 * random share (a beta latent), the switch summed out: the gate of an inlier and an outlier branch over a hidden Bernoulli
 * switch, read through the same route as an observation whose precision is chosen by a discrete variable (`observedSwitch`):
 * the inlier has the precision `tau`, the outlier `tau / width^2`, both `gaussian-gamma-precision` branches over the centre and
 * the gamma precision.
 */
export function observedOutlierSwitch(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  y: number,
  i: number,
  index: number | undefined,
): string | null {
  const widthCell = paramCell(cx, node, 'width', i)
  if (!isConstCell(widthCell))
    throw new Refusal(
      'operator',
      `the width of '${node.name}' is random: the outlier mixture has a message rule for a known width`,
    )
  const width = widthCell.aff.offset
  if (!(width > 0)) throw new Refusal('data', `the width of '${node.name}' is ${width}: it must be positive`)
  const z = hiddenSwitch(cx, node, paramCell(cx, node, 'outlier', i), index)
  const loc = paramCell(cx, node, 'loc', i)
  let scale: Cell
  if (paramIsConst(cx, node, 'scale')) {
    const sd = (paramCell(cx, node, 'scale', i) as Cell & { kind: 'aff' }).aff.offset
    scale = lookup(z, [sd, sd * width])
  } else {
    const scaleExpr = node.dist.params['scale'] as ExprSpec
    const precision = precisionExpr(cx.idiom, scaleExpr)
    const tau =
      precision === undefined
        ? undefined
        : scaledOf(cellsOf(cx, node, 'precisionProbe', () => evalSym(cx, precision))[i] as Cell)
    if (tau === undefined)
      throw new Refusal(
        'operator',
        `the scale of '${node.name}' must be the precision idiom 1 / sqrt(tau) of a gamma or exponential latent for the outlier mixture`,
      )
    const w2 = width * width
    scale = lookup(z, [0, 0], { precisions: [tau, { ...tau, scale: tau.scale / w2, offset: tau.offset / w2 }] })
  }
  return observedSwitch(cx, node, cx.elementName(node.name, node.dims, i), y, index, loc, scale)
}

/**
 * An observed Gaussian whose scale is `exp(s)` with `s` an affine form of Gaussian variables (a `gaussian-log-scale`
 * factor: the observation of a stochastic volatility model); the mean must be known. `NOT_MINE` when the mean is random or the scale's
 * argument is not such a form (the precision idiom is tried next).
 */
export function observedLogScale(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
  index: number | undefined,
): string | NotMine {
  const inner = unwrap(cx.nodes, node.dist.params['scale'] as ExprSpec) as ExprSpec & { readonly kind: 'op' }
  const loc = paramCell(cx, node, 'loc', i)
  if (!isConstCell(loc)) return NOT_MINE
  // Not an affine form of Gaussian variables (a product of random values, a gamma latent): not this mapping.
  const cell = probe(cx, () => cellsOf(cx, node, 'logScale', () => evalSym(cx, inner.args[0] as ExprSpec))[i] as Cell)
  if (cell === NOT_MINE || isConstCell(cell) || cell.kind !== 'aff') return NOT_MINE
  const s = cx.g.materialise(affOf(cell), node.name, index) as string
  cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
  cx.g.addFactor({ operator: 'gaussian-log-scale', args: [name, s], params: { loc: loc.aff.offset } }, name)
  return name
}

/**
 * An observed Gaussian whose scale is a positive latent (the `exp` of a Gaussian log-scale variable: a half-normal or
 * half-Cauchy sd projected by `latentProjected`, or `exp(affine)`) and whose mean may be random too: the `gaussian-scale`
 * factor, a one-dimensional quadrature over the log scale. Anything else is refused with the precision idiom named.
 */
export function observedRandomScale(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
  index: number | undefined,
  scaleExpr: ExprSpec,
): string {
  const stripped = stripClip(cx.idiom, scaleExpr)
  // The log variable behind the scale, or none (the scale is not an exp of a Gaussian form, nor a value with a log variable): then the
  // scale must be the precision idiom, and `randomPrecisions` refuses with it named.
  const found = probe(cx, () => {
    const logArgument = expArgument(stripped)
    if (logArgument !== undefined) {
      const cell = cellsOf(cx, node, 'logScale', () => evalSym(cx, logArgument))[i] as Cell
      return cell.kind === 'aff' && !isConstCell(cell)
        ? (cx.g.materialise(cell.aff, node.name, index) as string)
        : undefined
    }
    const cell = cellsOf(cx, node, 'scaleValue', () => evalSym(cx, stripped))[i] as Cell
    return logVariableOfCell(cell, cx.g.atoms, cx.logScale)
  })
  const logVar = found === NOT_MINE ? undefined : found
  if (logVar === undefined) randomPrecisions(cx, node)
  const loc = paramCell(cx, node, 'loc', i)
  const mean = cx.g.portOf(loc, node.name, index)
  cx.g.need('gaussian-scale')
  cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
  cx.g.addFactor({ operator: 'gaussian-scale', args: [name, mean, logVar as string] }, name)
  return name
}

/** An observed Gaussian with a gamma precision (`gaussian-gamma-precision`): the mean is a variable or a known value. */
export function observedRandomPrecision(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
  index: number | undefined,
): string {
  const tau = cellsOf(cx, node, 'precision', () => ({
    dims: node.dims,
    cells: randomPrecisions(cx, node).map((p): Cell => ({ kind: 'scaled', ...p })),
  }))[i] as Cell & { kind: 'scaled' }
  const loc = paramCell(cx, node, 'loc', i)
  const mean = cx.g.portOf(loc, node.name, index)
  cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
  cx.g.addFactor(precisionFactor(name, mean, tau), name)
  return name
}
