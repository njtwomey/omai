/**
 * Observations with a numeric likelihood of the registries (`likelihoods()`, `pairLikelihoods()`): a name, a predictor in the
 * coordinate the rule reads, and one factor (`observedNumeric`, `emitSite`), the pair of two random scalars (`observedPair`,
 * `positivePort`), the negative binomial with a random dispersion, multiclass sites, and the hurdle Poisson.
 */

import type { Json } from '../../../../base/index.ts'
import { likelihoods, pairLikelihoods } from '../../../../message-passing/index.ts'
import { toFlat } from '../../../../tensor/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { type Atom, affOf, type Cell, isConstCell, prod } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { expArgument, logVariableOfCell, scaledExp, stripClip, unwrap } from '../_idioms.ts'
import { NOT_MINE, type NotMine } from '../_mapping.ts'
import { cellsOf, constAt, paramCell } from '../_params.ts'
import {
  classDimensionMissing,
  classOutsideRange,
  cutPointsRandom,
  dispersionNotPositiveLatent,
  hurdleNotCount,
  hurdleRateNotExposureExp,
  keyNotPositive,
  keyNotPositiveLatent,
  knownParameterRandom,
  knownParameterRandomBecause,
  likelihoodRejects,
  negativeBinomialOutcome,
  quantileLevelRandom,
  shapeAndScaleBothRandom,
  weibullScaleNotLog,
  zeroProbabilityNotKnownOrBeta,
} from '../_reasons.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import { observedPoissonGate } from './observed-gates.ts'

/**
 * The negative binomial in the NB2 idiom with a random dispersion `a`: `n = 1 / a`, `probs = 1 / (1 + a * mean)` with `mean`
 * `exp(L)` (times a known exposure) and `a` a positive latent (carried on its log by `latentProjected`): the
 * `negative-binomial-dispersion` factor on the predictor `L` and the log of `a`. `NOT_MINE` when the parameters are not that
 * idiom or the dispersion is known (the registry's `negative-binomial-log` takes that).
 */
export function negBinomialRandomDispersion(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  y: number,
  i: number,
  index: number | undefined,
): string | null | NotMine {
  const dist = node.dist.params
  const nExpr = dist['n'] === undefined ? undefined : unwrap(cx.nodes, dist['n'])
  const pExpr = dist['probs'] === undefined ? undefined : unwrap(cx.nodes, dist['probs'])
  if (nExpr?.kind !== 'op' || nExpr.op !== 'div' || pExpr?.kind !== 'op' || pExpr.op !== 'div') return NOT_MINE
  const isOne = (e: ExprSpec): boolean => cx.isConst(e) && constAt(cx, node, e, i) === 1
  const [one, a] = nExpr.args as [ExprSpec, ExprSpec]
  if (!isOne(one) || cx.isConst(a)) return NOT_MINE
  const [num, den] = pExpr.args as [ExprSpec, ExprSpec]
  const d = unwrap(cx.nodes, den)
  if (!isOne(num) || d.kind !== 'op' || d.op !== 'add') return NOT_MINE
  const [p, q] = d.args as [ExprSpec, ExprSpec]
  const [unit, product] = cx.isConst(p) ? [p, unwrap(cx.nodes, q)] : [q, unwrap(cx.nodes, p)]
  if (!isOne(unit) || product.kind !== 'op' || product.op !== 'mul') return NOT_MINE
  const same = (x: ExprSpec, z: ExprSpec): boolean =>
    JSON.stringify(unwrap(cx.nodes, x)) === JSON.stringify(unwrap(cx.nodes, z))
  const [m1, m2] = product.args as [ExprSpec, ExprSpec]
  const mean = same(m1, a) ? m2 : same(m2, a) ? m1 : undefined
  if (mean === undefined) return NOT_MINE
  const m = scaledExp(cx.idiom, mean)
  if (m === undefined) return NOT_MINE
  const aCell = cellsOf(cx, node, 'dispersion', () => evalSym(cx, a))[i] as Cell
  const logDispersion = logVariableOfCell(aCell, cx.g.atoms, cx.logScale)
  if (logDispersion === undefined) throw dispersionNotPositiveLatent(node.name)
  const params: Record<string, Json> = { y, exposure: m.scale === undefined ? 1 : constAt(cx, node, m.scale, i) }
  if (!(Number.isInteger(y) && y >= 0)) throw negativeBinomialOutcome(node.name)
  const cell = cellsOf(cx, node, 'predictor', () => evalSym(cx, m.inner))[i] as Cell
  if (isConstCell(cell)) return null
  const u = cx.g.materialise(affOf(cell), node.name, index) as string
  cx.g.addFactor(
    { operator: 'negative-binomial-dispersion', args: [u, logDispersion], params },
    cx.elementName(node.name, node.dims, i),
  )
  return null
}

/**
 * The likelihoods of the registry of `message-passing/operators/likelihoods.ts` for `observe y ~ D(g(affine))`: Poisson with
 * `rate = exposure * exp(affine)`, binomial with `probs = sigmoid(affine)` or `probit(affine)`, negative binomial with a log
 * link and a known dispersion (`n = r`, `probs = r / (r + mean)`, the mean `exposure * exp(affine)`), and a Weibull with
 * `scale = exp(affine)` and a known shape. Returns `NOT_MINE` when the parameters are not that shape (the caller tries the
 * conjugate rule or refuses), else the factor's outcome (always `null`: the outcome is a parameter of the factor).
 */
export function observedNumeric(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  fam: string,
  y: number,
  i: number,
  index: number | undefined,
): string | null | NotMine {
  const dist = node.dist.params
  const known = (p: string, what: string): number => {
    let c: Cell
    try {
      c = paramCell(cx, node, p, i)
    } catch (e) {
      // A parameter computed from a random value (1 / alpha with alpha a latent) is random.
      if (e instanceof Refusal && e.kind === 'operator')
        throw knownParameterRandomBecause(what, node.name, e.message, fam)
      throw e
    }
    if (!isConstCell(c)) throw knownParameterRandom(what, node.name, fam)
    return c.aff.offset
  }
  const emit = (likelihood: string, expr: ExprSpec, extra: Record<string, Json>): string | null => {
    const params: Record<string, Json> = { likelihood, y, ...extra }
    const why = likelihoods()[likelihood]?.check?.(y, params)
    if (why !== undefined) throw likelihoodRejects(node.name, likelihood, why)
    const cell = cellsOf(cx, node, 'predictor', () => evalSym(cx, expr))[i] as Cell
    return emitSite(cx, node, cell, i, index, 'likelihood-row', 'likelihood-site', params)
  }
  if (fam === 'zeroTruncatedPoisson' || fam === 'hurdlePoisson') {
    const m = dist['rate'] === undefined ? undefined : scaledExp(cx.idiom, dist['rate'])
    if (m === undefined) return NOT_MINE
    return emit('poisson-log-truncated', m.inner, {
      exposure: m.scale === undefined ? 1 : constAt(cx, node, m.scale, i),
    })
  }
  if (fam === 'poisson') {
    const m = dist['rate'] === undefined ? undefined : scaledExp(cx.idiom, dist['rate'], true)
    if (m === undefined) return NOT_MINE
    if (m.scale !== undefined && !cx.isConst(m.scale))
      return observedPoissonGate(cx, node, y, i, index, m.inner, m.scale)
    return emit('poisson-log', m.inner, { exposure: m.scale === undefined ? 1 : constAt(cx, node, m.scale, i) })
  }
  if (fam === 'binomial') {
    const probs = dist['probs'] === undefined ? undefined : unwrap(cx.nodes, dist['probs'])
    if (probs?.kind !== 'op' || (probs.op !== 'sigmoid' && probs.op !== 'probit')) return NOT_MINE
    return emit(probs.op === 'sigmoid' ? 'binomial-logit' : 'binomial-probit', probs.args[0] as ExprSpec, {
      n: known('n', 'number of trials'),
    })
  }
  if (fam === 'orderedProbit' || fam === 'orderedLogistic') {
    const cutsExpr = dist['cutpoints'] as ExprSpec
    if (!cx.isConst(cutsExpr)) throw cutPointsRandom(node.name)
    const cuts = Array.from(toFlat(cx.evalNamed(cutsExpr).value))
    return emit(fam === 'orderedProbit' ? 'ordinal-probit' : 'ordinal-logit', dist['predictor'] as ExprSpec, { cuts })
  }
  if (fam === 'asymmetricLaplace') {
    if (!isConstCell(paramCell(cx, node, 'scale', i)))
      return observedPair(cx, node, 'asymmetric-laplace-scale', y, i, index)
    return emit('asymmetric-laplace', dist['loc'] as ExprSpec, {
      scale: known('scale', 'scale'),
      tau: known('tau', 'quantile level'),
    })
  }
  if (fam === 'weibull') {
    const m = dist['scale'] === undefined ? undefined : scaledExp(cx.idiom, dist['scale'])
    if (m === undefined || m.scale !== undefined) return NOT_MINE
    if (!isConstCell(paramCell(cx, node, 'concentration', i)))
      return observedPair(cx, node, 'weibull-shape', y, i, index)
    return emit('weibull-log', m.inner, { shape: known('concentration', 'shape') })
  }
  if (fam === 'negativeBinomial') {
    const random = negBinomialRandomDispersion(cx, node, y, i, index)
    if (random !== NOT_MINE) return random
    // The negative binomial with a log link: n = r, probs = r / (r + mean) = A / (A + K exp(L)) with mean = n K exp(L) / A.
    const r = known('n', 'dispersion (the number of successes n = 1 / dispersion)')
    const probs = dist['probs'] === undefined ? undefined : unwrap(cx.nodes, dist['probs'])
    if (probs?.kind !== 'op' || probs.op !== 'div') return NOT_MINE
    const [num, den] = probs.args as [ExprSpec, ExprSpec]
    const d = unwrap(cx.nodes, den)
    if (!cx.isConst(num) || d.kind !== 'op' || d.op !== 'add') return NOT_MINE
    const [p, q] = d.args as [ExprSpec, ExprSpec]
    const [one, rest] = cx.isConst(p) ? [p, q] : [q, p]
    if (!cx.isConst(one)) return NOT_MINE
    const A = constAt(cx, node, num, i)
    if (Math.abs(constAt(cx, node, one, i) - A) > 1e-12 * Math.abs(A)) return NOT_MINE
    const m = scaledExp(cx.idiom, rest)
    if (m === undefined) return NOT_MINE
    const K = m.scale === undefined ? 1 : constAt(cx, node, m.scale, i)
    return emit('negative-binomial-log', m.inner, { dispersion: 1 / r, exposure: (r * K) / A })
  }
  return NOT_MINE
}

/**
 * A categorical outcome whose class scores are affine forms of Gaussian variables over one extra dimension (the classes): a
 * `logits` expression `X W` with `W` over (feature, class) for the softmax, or `scores` for the multinomial probit. One
 * `multiclass-site` factor per observation on the `K` scores (each a Gaussian variable, or a known value). The outcome is a
 * parameter of the factor.
 */
export function observedMulticlass(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  y: number,
  i: number,
  index: number | undefined,
  param: string,
  link: 'logit' | 'probit',
): string | null {
  const key = `${node.name}|${param}`
  let cells = cx.cells.get(key)
  if (cells === undefined) {
    const sym = evalSym(cx, node.dist.params[param] as ExprSpec)
    const extra = sym.dims.filter((dim) => (node.dist.eventDims ?? []).includes(dim))
    if (extra.length !== 1) throw classDimensionMissing(param, node.name, sym.dims)
    cells = cx.over(sym, [...node.dims, extra[0] as string])
    cx.cells.set(key, cells)
  }
  const classes = cells.length / Math.max(1, prod(cx.shapeOf(node.dims)))
  if (!(Number.isInteger(y) && y >= 0 && y < classes)) throw classOutsideRange(node.name, y, classes - 1)
  const args = Array.from({ length: classes }, (_, k) => {
    return cx.g.portOf(cells?.[i * classes + k] as Cell, node.name, index)
  })
  if (args.every((a) => typeof a !== 'string')) return null
  cx.g.addFactor(
    { operator: 'multiclass-site', args, params: { y, classes, link } },
    cx.elementName(node.name, node.dims, i),
  )
  return null
}

/**
 * A positive random parameter of an observed node as one port of a `likelihood-pair` factor, in the coordinate the rule reads
 * (the log of the value): a gamma variable (its own variable), a positive latent carried on its log (a log-normal, half-normal or
 * half-Cauchy prior: the log variable), or `exp(affine)` (a Gaussian variable made of the affine form). A known value is the
 * constant coordinate `log value`.
 */
export function positivePort(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  key: string,
  i: number,
  index: number | undefined,
): string | { readonly const: number } {
  const expr = node.dist.params[key]
  const stripped = expr === undefined ? undefined : stripClip(cx.idiom, expr)
  const logArgument = stripped === undefined ? undefined : expArgument(stripped)
  if (logArgument !== undefined) {
    const cell = cellsOf(cx, node, `${key}:log`, () => evalSym(cx, logArgument))[i] as Cell
    if (cell.kind === 'aff') {
      const m = cx.g.materialise(cell.aff, node.name, index)
      return typeof m === 'string' ? m : { const: m }
    }
  }
  const c = paramCell(cx, node, key, i)
  if (isConstCell(c)) {
    if (!(c.aff.offset > 0)) throw keyNotPositive(key, node.name, c.aff.offset)
    return { const: Math.log(c.aff.offset) }
  }
  if (c.kind === 'rv' && c.family === 'gamma') return c.variable
  const logVar = logVariableOfCell(c, cx.g.atoms, cx.logScale)
  if (logVar !== undefined) return logVar
  throw keyNotPositiveLatent(key, node.name)
}

/**
 * A numeric likelihood over two random scalars (`likelihood-pair`): a Laplace with a random scale, a Student-t with random
 * degrees of freedom (known scale), a Weibull with a random shape (`scale = exp(affine)`) and a beta outcome with two random
 * shapes. The outcome is a parameter of the factor. A parameter that is known is a constant coordinate; nothing random at
 * all says nothing.
 */
export function observedPair(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  likelihood: string,
  y: number,
  i: number,
  index: number | undefined,
): string | null {
  const extra: Record<string, Json> = {}
  const predictor = (cell: Cell): string | { readonly const: number } => cx.g.portOf(cell, node.name, index)
  let args: [string | { readonly const: number }, string | { readonly const: number }]
  if (likelihood === 'laplace-scale' || likelihood === 'asymmetric-laplace-scale') {
    if (likelihood === 'asymmetric-laplace-scale') {
      const tau = paramCell(cx, node, 'tau', i)
      if (!isConstCell(tau)) throw quantileLevelRandom(node.name)
      extra['tau'] = tau.aff.offset
    }
    args = [predictor(paramCell(cx, node, 'loc', i)), positivePort(cx, node, 'scale', i, index)]
  } else if (likelihood === 'student-t-df') {
    const scale = paramCell(cx, node, 'scale', i)
    if (!isConstCell(scale)) throw shapeAndScaleBothRandom(node.name)
    extra['scale'] = scale.aff.offset
    args = [predictor(paramCell(cx, node, 'loc', i)), positivePort(cx, node, 'df', i, index)]
  } else if (likelihood === 'weibull-shape') {
    const m = node.dist.params['scale'] === undefined ? undefined : scaledExp(cx.idiom, node.dist.params['scale'])
    if (m === undefined || m.scale !== undefined) throw weibullScaleNotLog(node.name)
    args = [
      predictor(cellsOf(cx, node, 'predictor', () => evalSym(cx, m.inner))[i] as Cell),
      positivePort(cx, node, 'concentration', i, index),
    ]
  } else {
    args = [positivePort(cx, node, 'a', i, index), positivePort(cx, node, 'b', i, index)]
  }
  const params: Record<string, Json> = { likelihood, y, ...extra }
  const why = pairLikelihoods()[likelihood]?.check?.(y, params)
  if (why !== undefined) throw likelihoodRejects(node.name, likelihood, why)
  if (typeof args[0] !== 'string' && typeof args[1] !== 'string') return null
  cx.g.addFactor({ operator: 'likelihood-pair', args, params }, cx.elementName(node.name, node.dims, i))
  return null
}

/**
 * The factor of a numeric likelihood of an affine predictor: one row factor on the packed weights when the predictor is a
 * single dot product with one vector, else a site factor on an auxiliary predictor variable. The outcome is a parameter of
 * the factor, so the observed value has no variable of its own. A predictor with no random part says nothing (no factor).
 */
export function emitSite(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  predictor: Cell,
  i: number,
  index: number | undefined,
  row: string,
  site: string,
  params: Record<string, Json>,
): string | null {
  if (isConstCell(predictor)) return null
  const aff = affOf(predictor)
  const terms = [...aff.terms].filter(([, c]) => c !== 0)
  const comps = terms.map(([k]) => cx.g.atoms.get(k) as Atom)
  const oneVector =
    aff.offset === 0 &&
    comps.length > 0 &&
    comps.every((a) => a.kind === 'comp' && a.vector === (comps[0] as Atom & { kind: 'comp' }).vector)
  const id = cx.elementName(node.name, node.dims, i)
  if (oneVector) {
    const first = comps[0] as Atom & { kind: 'comp' }
    const a = new Array<number>(first.size).fill(0)
    for (const [k, c] of terms) a[(cx.g.atoms.get(k) as Atom & { kind: 'comp' }).index] = c
    cx.g.addFactor({ operator: row, args: [first.vector], params: { a, ...params } }, id)
  } else {
    const m = cx.g.materialise(aff, node.name, index)
    if (typeof m !== 'string') return null
    cx.g.addFactor({ operator: site, args: [m], params }, id)
  }
  return null
}

/**
 * An observed hurdle count: zero with probability `zero`, else a zero-truncated Poisson of `exposure * exp(affine)`. The state
 * of the hurdle (a structural zero, or a positive count) is determined by the count and summed out exactly: a zero is an
 * observation of the Bernoulli `zero`, with a beta probability the conjugate `bernoulli` factor, and a positive count is the same
 * Bernoulli observed false and the zero-truncated Poisson site on the log rate (the gate of {point mass at zero, truncated
 * Poisson}, with its switch observed). The positive part and the zero part share no variable, so each is exact.
 */
export function observedHurdle(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  y: number,
  i: number,
  index: number | undefined,
): string | null {
  if (!(Number.isInteger(y) && y >= 0)) throw hurdleNotCount(node.name, y)
  const zero = paramCell(cx, node, 'zero', i)
  if (!isConstCell(zero)) {
    if (zero.kind !== 'rv' || zero.family !== 'beta') throw zeroProbabilityNotKnownOrBeta(node.name)
    const crossed = cx.g.auxName(node.name, index)
    cx.g.addVariable({ name: crossed, family: { kind: 'bernoulli' }, observed: y === 0 ? 1 : 0 })
    cx.g.addFactor({ operator: 'bernoulli', args: [crossed, zero.variable] }, crossed)
  }
  if (y === 0) return null
  const numeric = observedNumeric(cx, node, 'hurdlePoisson', y, i, index)
  if (numeric === NOT_MINE) throw hurdleRateNotExposureExp(node.name)
  return numeric
}
