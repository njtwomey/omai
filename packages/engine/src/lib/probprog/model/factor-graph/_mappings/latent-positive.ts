import {
  betaEta,
  type FamilyKind,
  gammaPrior,
  normalGammaPrior,
  priorFactor,
} from '../../../../message-passing/index.ts'
import { ones, toFlat } from '../../../../tensor/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { type Cell, isConstCell, prod, scaledOf } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { logVariableOf, unwrap } from '../_idioms.ts'
import { NOT_MINE, type NotMine } from '../_mapping.ts'
import { constParam } from '../_params.ts'
import {
  concentrationNotMultipleOfLatent,
  concentrationNotMultiplesOfOne,
  concentrationNotRandomTimesKnown,
  gammaRateNotMultiple,
} from '../_reasons.ts'
import { evalSym } from '../_symbolic/_affine.ts'

/** One component of a random concentration, `weight * alpha + offset`, with the variable of `alpha` (the variable of its log when `log`). */
export interface ConcentrationPart {
  readonly variable?: string
  readonly log?: boolean
  readonly weight: number
  readonly offset: number
}

/**
 * The normal-gamma pair of a gamma (or inverse-gamma) latent `T` (a precision or a variance), when the plan found one (a Gaussian
 * latent `M` with a known mean and the variance `1 / (kappa lambda)` and observed Gaussians `y ~ N(M, 1 / (c lambda))`, with
 * nothing else reading either of them; `_plan.ts` has the detection). The pair is one `normal-gamma` variable with a prior of both
 * and a `gaussian-normal-gamma` factor per observation, so the joint posterior is exactly the conjugate normal-gamma (the classic
 * normal-inverse-gamma prior on a mean and a variance). Returns `NOT_MINE` (and changes nothing) when the model is not that shape: the
 * mean and precision then compile to a Gaussian with a gamma precision each.
 */
export function latentNormalGamma(cx: Context, node: ModelNode & { readonly kind: 'latent' }): true | NotMine {
  const detection = cx.plan.normalGamma.get(node.name)
  if (detection === undefined) return NOT_MINE
  for (const name of detection.customs) cx.handledCustoms.add(name)
  if (detection.failure !== undefined) throw detection.failure
  if (detection.pair === undefined) return NOT_MINE
  const { plan, multiples } = detection.pair
  const inverse = plan.inverse
  const rateName = inverse ? 'scale' : 'rate'
  const mean = cx.nodes.get(plan.mean) as ModelNode & { readonly kind: 'latent' }
  cx.g.need('gaussian-normal-gamma')
  const a = constParam(cx, node, 'concentration', 'the shape')[0] as number
  const b = constParam(cx, node, rateName, 'the rate')[0] as number
  const m = constParam(cx, mean, 'loc', 'the mean')[0] as number
  cx.normalGammaByMean.set(mean.name, plan)
  for (const name of plan.observed) cx.normalGammaObserved.set(name, { plan, multiple: multiples.get(name) as number })
  cx.g.addVariable({ name: plan.variable, family: { kind: 'normal-gamma' } })
  cx.g.addLibraryFactor(normalGammaPrior(plan.variable, m, plan.kappa, a, b), plan.variable)
  cx.layout(node, 'latent', 'normal-gamma', 'scalar', [plan.variable], true, {
    part: 'precision',
    ...(inverse ? { reciprocal: true } : {}),
  })
  return true
}

/**
 * A beta, gamma, inverse-gamma or exponential latent whose parameters are all known: one variable per element of the message family
 * `kind` and its prior factor (`beta-eta`, or `gamma-prior`). An inverse gamma `x` with scale `b` is the precision `1 / x` ~ Gamma(a,
 * rate b): the variable is that gamma and the site its reciprocal.
 */
export function latentKnownPrior(cx: Context, node: ModelNode & { readonly kind: 'latent' }, kind: FamilyKind): void {
  const fam = node.dist.family
  const inverse = fam === 'inverseGamma'
  const rateName = inverse ? 'scale' : 'rate'
  const shape = cx.shapeOf(node.dims)
  const n = prod(shape)
  const params =
    fam === 'gamma' || inverse
      ? [constParam(cx, node, 'concentration', 'the shape'), constParam(cx, node, rateName, 'the rate')]
      : fam === 'exponential'
        ? [toFlat(ones([n])), constParam(cx, node, 'rate', 'the rate')]
        : [constParam(cx, node, 'a', 'the shape'), constParam(cx, node, 'b', 'the shape')]
  const cells: Cell[] = []
  const vars: string[] = []
  for (let i = 0; i < n; i++) {
    const name = cx.g.addVariable({ name: cx.elementName(node.name, node.dims, i), family: { kind } })
    const p = [params[0]?.[i] as number, params[1]?.[i] as number]
    const f =
      fam === 'beta'
        ? priorFactor(name, betaEta(p[0] as number, p[1] as number))
        : gammaPrior(name, p[0] as number, p[1] as number)
    cx.g.addLibraryFactor(f, name)
    cells.push(inverse ? { kind: 'recip', variable: name, scale: 1 } : { kind: 'rv', variable: name, family: kind })
    vars.push(name)
  }
  cx.finishLatent(node, kind, cells, vars, inverse ? { reciprocal: true } : {})
}

/**
 * A gamma latent whose rate is a known multiple of another gamma latent (`gamma(0.5, b)` with `b` gamma: the
 * Makalic-Schmidt half-Cauchy, or a Student-t's mixing precision): a `gamma-rate` factor per element, whose shape is known.
 */
export function latentGammaRate(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const inverse = node.dist.family === 'inverseGamma'
  const shapes = constParam(cx, node, 'concentration', 'the shape')
  const rate = node.dist.params[inverse ? 'scale' : 'rate'] as ExprSpec
  const cells = cx.over(evalSym(cx, rate), node.dims)
  const out: Cell[] = []
  const vars: string[] = []
  for (const [i, c] of cells.entries()) {
    const r = scaledOf(c)
    if (r === undefined || r.offset !== 0 || !(r.scale > 0)) throw gammaRateNotMultiple(node.name)
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gamma' },
    })
    cx.g.need('gamma-rate')
    cx.g.addFactor(
      {
        operator: 'gamma-rate',
        args: [name, r.variable],
        params: { shape: shapes[i] as number, ...(r.scale === 1 ? {} : { scale: r.scale }) },
      },
      name,
    )
    out.push(inverse ? { kind: 'recip', variable: name, scale: 1 } : { kind: 'rv', variable: name, family: 'gamma' })
    vars.push(name)
  }
  cx.finishLatent(node, 'gamma', out, vars, inverse ? { reciprocal: true } : {})
}

/**
 * A beta or Dirichlet latent whose concentrations are known multiples of one random positive value plus known offsets
 * (`beta(alpha, alpha)`, `beta(alpha, 2)`, `dirichlet(alpha * ones)`: the strength of the prior learned from the data) with `alpha`
 * a gamma latent or a log-normal one (carried on its log): a `dirichlet-concentration` factor per element, which is the
 * prior of the variable (a numeric rule: a grid over `log alpha`).
 */
export function latentRandomConcentration(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const isBeta = node.dist.family === 'beta'
  const shape = cx.shapeOf(node.dims)
  const n = prod(shape)
  // The concentration of each element: one cell per component (a beta has two parameters, a Dirichlet an event axis).
  let perElement: ConcentrationPart[][]
  if (isBeta) {
    const a = cx.over(evalSym(cx, node.dist.params['a'] as ExprSpec), node.dims)
    const b = cx.over(evalSym(cx, node.dist.params['b'] as ExprSpec), node.dims)
    perElement = Array.from({ length: n }, (_, i) => [
      concentrationOf(cx, node, a[i] as Cell),
      concentrationOf(cx, node, b[i] as Cell),
    ])
  } else {
    // `concentration = S * V + O` with `S` random (a gamma or log-normal latent, a known multiple of one plus a known value) and `V`, `O`
    // known vectors over the event axis: the cell of each component is `V_k S + O_k`.
    const parsed = concentrationForm(cx, node.dist.params['concentration'] as ExprSpec)
    if (parsed === undefined) throw concentrationNotRandomTimesKnown(node.name)
    const scalar = cx.over(evalSym(cx, parsed.random), node.dims)
    perElement = scalar.map((s) => {
      const base = concentrationOf(cx, node, s)
      return parsed.scale.map(
        (v, k): ConcentrationPart => ({
          ...base,
          weight: v * base.weight,
          offset: v * base.offset + (parsed.offset[k] as number),
        }),
      )
    })
  }
  const cells: Cell[] = []
  const vars: string[] = []
  cx.g.need('dirichlet-concentration')
  for (const [i, parts] of perElement.entries()) {
    const comps = parts
    const random = parts.filter((p) => p.variable !== undefined)
    const variable = random[0]?.variable
    const log = random[0]?.log === true
    // Two separate random shapes, each exactly a gamma or log-normal latent: the two-dimensional rule.
    const [pa, pb] = comps
    if (
      isBeta &&
      pa?.variable !== undefined &&
      pb?.variable !== undefined &&
      pa.variable !== pb.variable &&
      [pa, pb].every((q) => q.weight === 1 && q.offset === 0)
    ) {
      cx.g.need('beta-two-shapes')
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'beta' },
      })
      cx.g.addFactor({ operator: 'beta-two-shapes', args: [name, pa.variable, pb.variable] }, name)
      cells.push({ kind: 'rv', variable: name, family: 'beta' })
      vars.push(name)
      continue
    }
    if (variable === undefined || random.some((p) => p.variable !== variable || (p.log === true) !== log))
      throw concentrationNotMultiplesOfOne(node.name)
    const family: FamilyKind = isBeta ? 'beta' : 'dirichlet'
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: isBeta ? { kind: 'beta' } : { kind: 'dirichlet', size: comps.length },
    })
    cx.g.addFactor(
      {
        operator: 'dirichlet-concentration',
        args: [name, variable],
        params: {
          weights: parts.map((p) => p.weight),
          offsets: parts.map((p) => p.offset),
          ...(log ? { log: true } : {}),
        },
      },
      name,
    )
    cells.push({ kind: 'rv', variable: name, family })
    vars.push(name)
  }
  cx.finishLatent(node, isBeta ? 'beta' : 'dirichlet', cells, vars)
}

/** `S * V + O` for known vectors `V` and `O` over an event axis and a random scalar expression `S`: the form of a Dirichlet's random concentration. */
export function concentrationForm(
  cx: Context,
  e: ExprSpec,
): { readonly random: ExprSpec; readonly scale: readonly number[]; readonly offset: readonly number[] } | undefined {
  const u = unwrap(cx.nodes, e)
  const vector = (x: ExprSpec): number[] | undefined => {
    if (!cx.isConst(x)) return undefined
    const nv = cx.evalNamed(x)
    return nv.value.shape.length > nv.dims.length ? toFlat(nv.value) : undefined
  }
  if (u.kind !== 'op') return undefined
  const [a, b] = u.args as [ExprSpec, ExprSpec]
  if (u.op === 'mul') {
    const va = vector(a)
    const vb = vector(b)
    if (va !== undefined && vb === undefined) return { random: b, scale: va, offset: va.map(() => 0) }
    if (vb !== undefined && va === undefined) return { random: a, scale: vb, offset: vb.map(() => 0) }
    return undefined
  }
  if (u.op === 'add') {
    const va = vector(a)
    const vb = vector(b)
    const [inner, o] = va !== undefined ? [b, va] : vb !== undefined ? [a, vb] : [undefined, undefined]
    if (inner === undefined || o === undefined) return undefined
    const base = concentrationForm(cx, inner)
    return base === undefined ? undefined : { ...base, offset: base.offset.map((v, k) => v + (o[k] as number)) }
  }
  return undefined
}

/**
 * One component of a concentration as `weight * alpha + offset`: a known value, a gamma variable (or a known multiple of one
 * plus an offset), or a positive latent carried on its log (`alpha = exp(l)`, the variable returned is `l`, with `log`).
 */
export function concentrationOf(cx: Context, node: ModelNode, c: Cell): ConcentrationPart {
  if (isConstCell(c)) return { weight: 0, offset: c.aff.offset }
  const g = scaledOf(c)
  if (g !== undefined) return { variable: g.variable, weight: g.scale, offset: g.offset }
  if (c.kind === 'aff' && c.aff.terms.size === 1) {
    const [key, coefficient] = [...c.aff.terms][0] as [string, number]
    const l = logVariableOf(cx.g.atoms, cx.logScale, key)
    if (l !== undefined) return { variable: l, log: true, weight: coefficient, offset: c.aff.offset }
  }
  throw concentrationNotMultipleOfLatent(node.name)
}
