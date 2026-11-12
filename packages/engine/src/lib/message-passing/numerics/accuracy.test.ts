/*
 * The error of the numeric rules against a high-order reference, operator by operator: the table of
 * `docs/probprog/review-rules-numerics.md` is this file's output (`OMAI_ACCURACY_PRINT=1` prints it), and its assertions are the bounds
 * the rules must stay under. Each scenario integrates the exact tilted density of one operator with a composite Gauss-Legendre rule
 * of high order on a window found by scanning (no code of the operator in it: the integrand is written out here), turns the
 * tilted moments of the target port into the mean and sd of the member of its family, and compares the operator's message the
 * same way: `z` is the error of the mean in standard deviations of the reference, `sd` the relative error of the sd.
 */
import { describe, expect, it } from 'vitest'
import { gaussLegendre } from '../../numerics/quadrature/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { messageArg, valueArg } from '../core/args.ts'
import { bernoulli, bernoulliEta, beta, betaEta, gamma, gammaEta, gaussian, gaussianEta } from '../core/families.ts'
import { gaussianMv } from '../core/families-vector.ts'
import type { Arg, Eta, FactorOperator, Family, Params } from '../core/types.ts'
import { gaussianGammaPrecision, product } from '../operators/gaussian/gaussian.ts'
import { gaussianScale } from '../operators/gaussian/log-scale.ts'
import { apply } from '../operators/likelihoods/functions.ts'
import { likelihoodSiteOperator, negativeBinomialDispersion } from '../operators/likelihoods/likelihoods.ts'
import { likelihoodPairOperator } from '../operators/likelihoods/pairs.ts'
import { priorSite } from '../operators/likelihoods/prior-sites.ts'
import { mixtureSite } from '../operators/likelihoods/robust.ts'
import { likelihoodTripleOperator, tripleLikelihoods } from '../operators/likelihoods/triples.ts'
import { greaterThan } from '../operators/links/truncation.ts'
import { betaTwoShapes, dirichletConcentration } from '../operators/scales/concentration.ts'
import { gaussianMvFunctionScale } from '../operators/scales/function-scale.ts'
import { gammaProduct, gammaRate } from '../operators/scales/gamma-scale.ts'
import { gaussianMvPrecisionScale } from '../operators/scales/scale.ts'
import { digammaOf, lgammaOf, Phi } from './_scalar.ts'

// ----- the high-order reference -----

const rules = new Map<number, { x: number[]; w: number[] }>()
function rule(order: number): { x: number[]; w: number[] } {
  let r = rules.get(order)
  if (r === undefined) {
    const g = gaussLegendre(order)
    r = { x: toFlat(g.nodes), w: toFlat(g.weights) }
    rules.set(order, r)
  }
  return r
}

/** Composite Gauss-Legendre nodes on [a, b]. */
function nodes(a: number, b: number, panels: number, order: number): { x: number[]; w: number[] } {
  const { x: rx, w: rw } = rule(order)
  const x: number[] = []
  const w: number[] = []
  const h = (b - a) / panels
  for (let p = 0; p < panels; p++)
    for (let i = 0; i < order; i++) {
      x.push(a + p * h + ((rx[i] as number) + 1) * 0.5 * h)
      w.push((rw[i] as number) * 0.5 * h)
    }
  return { x, w }
}

/** Nodes on [lo, hi] split at the kinks inside it. */
function splitNodes(lo: number, hi: number, kinks: readonly number[], panels: number, order: number) {
  const cuts = [lo, ...kinks.filter((k) => k > lo && k < hi).sort((p, q) => p - q), hi]
  const x: number[] = []
  const w: number[] = []
  for (let s = 0; s + 1 < cuts.length; s++) {
    const n = Math.max(8, Math.round((panels * ((cuts[s + 1] as number) - (cuts[s] as number))) / (hi - lo)))
    const g = nodes(cuts[s] as number, cuts[s + 1] as number, n, order)
    x.push(...g.x)
    w.push(...g.w)
  }
  return { x, w }
}

/** The window of [lo, hi] where `logw` is within 60 nats of its maximum, found by scanning and narrowing, padded. */
function massWindow1(logw: (t: number) => number, lo: number, hi: number): [number, number] {
  let a = lo
  let b = hi
  for (let round = 0; round < 8; round++) {
    const n = 20001
    const h = (b - a) / (n - 1)
    const lp = Array.from({ length: n }, (_, i) => logw(a + i * h))
    const top = Math.max(...lp)
    let first = 0
    while (first < n - 1 && (lp[first] as number) < top - 60) first++
    let last = n - 1
    while (last > 0 && (lp[last] as number) < top - 60) last--
    const na = Math.max(lo, a + (first - 3) * h)
    const nb = Math.min(hi, a + (last + 3) * h)
    if (nb - na > 0.7 * (b - a)) return [na, nb]
    a = na
    b = nb
  }
  return [a, b]
}

/** An expectation under the density proportional to `exp(logw)` on [lo, hi], with the log mass, by a high-order rule. */
function expect1(logw: (t: number) => number, lo: number, hi: number, kinks: readonly number[] = []) {
  const [a, b] = massWindow1(logw, lo, hi)
  const g = splitNodes(a, b, kinks, 600, 12)
  const lw = g.x.map((t, i) => Math.log(g.w[i] as number) + logw(t))
  const top = Math.max(...lw)
  const w = lw.map((v) => Math.exp(v - top))
  const z = w.reduce((s, v) => s + v, 0)
  return {
    logZ: top + Math.log(z),
    of: (f: (t: number) => number): number => w.reduce((s, v, i) => s + v * f(g.x[i] as number), 0) / z,
    /** The mean and the centred variance of `f` under the density. */
    gauss: (f: (t: number) => number): { mean: number; variance: number } => {
      const mean = w.reduce((s, v, i) => s + v * f(g.x[i] as number), 0) / z
      return { mean, variance: w.reduce((s, v, i) => s + v * (f(g.x[i] as number) - mean) ** 2, 0) / z }
    },
  }
}

/** The same in two coordinates: a product rule on a window found by zooming onto the marginals of a first pass. */
function expect2(
  logw: (t0: number, t1: number) => number,
  box: readonly [readonly [number, number], readonly [number, number]],
  kinks: readonly [readonly number[], readonly number[]] = [[], []],
) {
  const win: [number, number][] = [
    [box[0][0], box[0][1]],
    [box[1][0], box[1][1]],
  ]
  const pass = (panels: number, order: number) => {
    const g = [0, 1].map((d) => splitNodes(win[d]?.[0] as number, win[d]?.[1] as number, kinks[d] ?? [], panels, order))
    const g0 = g[0] as { x: number[]; w: number[] }
    const g1 = g[1] as { x: number[]; w: number[] }
    const lw = new Float64Array(g0.x.length * g1.x.length)
    let top = Number.NEGATIVE_INFINITY
    for (let i = 0; i < g0.x.length; i++)
      for (let j = 0; j < g1.x.length; j++) {
        const v = Math.log((g0.w[i] as number) * (g1.w[j] as number)) + logw(g0.x[i] as number, g1.x[j] as number)
        lw[i * g1.x.length + j] = v
        if (v > top) top = v
      }
    return { g0, g1, lw, top }
  }
  for (let round = 0; round < 5; round++) {
    const { g0, g1, lw, top } = pass(24, 8)
    const m0 = new Array<number>(g0.x.length).fill(0)
    const m1 = new Array<number>(g1.x.length).fill(0)
    for (let i = 0; i < g0.x.length; i++)
      for (let j = 0; j < g1.x.length; j++) {
        const e = Math.exp((lw[i * g1.x.length + j] as number) - top)
        m0[i] = (m0[i] as number) + e
        m1[j] = (m1[j] as number) + e
      }
    let shrunk = false
    for (const [d, m, g] of [
      [0, m0, g0],
      [1, m1, g1],
    ] as const) {
      const peak = Math.max(...m)
      const live = m.flatMap((v, k) => (v > 1e-14 * peak ? [k] : []))
      const [a, b] = win[d] as [number, number]
      const margin = (b - a) / 24
      const na = Math.max(a, (g.x[Math.min(...live)] as number) - margin)
      const nb = Math.min(b, (g.x[Math.max(...live)] as number) + margin)
      if (nb - na < 0.7 * (b - a)) {
        win[d] = [na, nb]
        shrunk = true
      }
    }
    if (!shrunk) break
  }
  const { g0, g1, lw, top } = pass(40, 10)
  const total = Array.from(lw).reduce((s, v) => s + Math.exp(v - top), 0)
  return {
    logZ: top + Math.log(total),
    of: (f: (t0: number, t1: number) => number): number => {
      let s = 0
      for (let i = 0; i < g0.x.length; i++)
        for (let j = 0; j < g1.x.length; j++)
          s += Math.exp((lw[i * g1.x.length + j] as number) - top) * f(g0.x[i] as number, g1.x[j] as number)
      return s / total
    },
    /** The mean and the centred variance of `f` under the density. */
    gauss: (f: (t0: number, t1: number) => number): { mean: number; variance: number } => {
      let m = 0
      for (let i = 0; i < g0.x.length; i++)
        for (let j = 0; j < g1.x.length; j++)
          m += Math.exp((lw[i * g1.x.length + j] as number) - top) * f(g0.x[i] as number, g1.x[j] as number)
      const mean = m / total
      let v = 0
      for (let i = 0; i < g0.x.length; i++)
        for (let j = 0; j < g1.x.length; j++)
          v +=
            Math.exp((lw[i * g1.x.length + j] as number) - top) * (f(g0.x[i] as number, g1.x[j] as number) - mean) ** 2
      return { mean, variance: v / total }
    },
  }
}

/** The same in three coordinates: a product rule (panels of 8 points) on a window found by zooming onto the marginals of coarser passes. */
function expect3(
  logw: (t0: number, t1: number, t2: number) => number,
  box: readonly [readonly [number, number], readonly [number, number], readonly [number, number]],
) {
  const win: [number, number][] = box.map(([a, b]) => [a, b])
  const pass = (panels: number) => {
    const g = [0, 1, 2].map((d) => nodes(win[d]?.[0] as number, win[d]?.[1] as number, panels, 8))
    const [g0, g1, g2] = g as [{ x: number[]; w: number[] }, { x: number[]; w: number[] }, { x: number[]; w: number[] }]
    const lw = new Float64Array(g0.x.length * g1.x.length * g2.x.length)
    let top = Number.NEGATIVE_INFINITY
    let at = 0
    for (let i = 0; i < g0.x.length; i++)
      for (let j = 0; j < g1.x.length; j++)
        for (let k = 0; k < g2.x.length; k++) {
          const v =
            Math.log((g0.w[i] as number) * (g1.w[j] as number) * (g2.w[k] as number)) +
            logw(g0.x[i] as number, g1.x[j] as number, g2.x[k] as number)
          lw[at++] = v
          if (v > top) top = v
        }
    return { g0, g1, g2, lw, top }
  }
  for (let round = 0; round < 5; round++) {
    const { g0, g1, g2, lw, top } = pass(10)
    const m = [g0, g1, g2].map((g) => new Array<number>(g.x.length).fill(0))
    let at = 0
    for (let i = 0; i < g0.x.length; i++)
      for (let j = 0; j < g1.x.length; j++)
        for (let k = 0; k < g2.x.length; k++) {
          const e = Math.exp((lw[at++] as number) - top)
          ;(m[0] as number[])[i] = ((m[0] as number[])[i] as number) + e
          ;(m[1] as number[])[j] = ((m[1] as number[])[j] as number) + e
          ;(m[2] as number[])[k] = ((m[2] as number[])[k] as number) + e
        }
    let shrunk = false
    for (const [d, g] of [g0, g1, g2].entries()) {
      const marginal = m[d] as number[]
      const peak = Math.max(...marginal)
      const live = marginal.flatMap((v, q) => (v > 1e-13 * peak ? [q] : []))
      const [a, b] = win[d] as [number, number]
      const margin = (b - a) / 10
      const na = Math.max(a, (g.x[Math.min(...live)] as number) - margin)
      const nb = Math.min(b, (g.x[Math.max(...live)] as number) + margin)
      if (nb - na < 0.7 * (b - a)) {
        win[d] = [na, nb]
        shrunk = true
      }
    }
    if (!shrunk) break
  }
  const { g0, g1, g2, lw, top } = pass(16)
  let total = 0
  for (const v of lw) total += Math.exp(v - top)
  const moments = (f: (t0: number, t1: number, t2: number) => number): { mean: number; variance: number } => {
    let m1 = 0
    let at = 0
    for (let i = 0; i < g0.x.length; i++)
      for (let j = 0; j < g1.x.length; j++)
        for (let k = 0; k < g2.x.length; k++)
          m1 += Math.exp((lw[at++] as number) - top) * f(g0.x[i] as number, g1.x[j] as number, g2.x[k] as number)
    const mean = m1 / total
    let v = 0
    at = 0
    for (let i = 0; i < g0.x.length; i++)
      for (let j = 0; j < g1.x.length; j++)
        for (let k = 0; k < g2.x.length; k++)
          v +=
            Math.exp((lw[at++] as number) - top) *
            (f(g0.x[i] as number, g1.x[j] as number, g2.x[k] as number) - mean) ** 2
    return { mean, variance: v / total }
  }
  return { logZ: top + Math.log(total), gauss: moments }
}

// ----- comparing a message with the reference -----

/** The mean and sd of the member of a family with these natural parameters. */
function memberMeanSd(family: Family, eta: readonly number[]): { mean: number; sd: number } {
  if (family.kind === 'gaussian')
    return { mean: (eta[0] as number) / (eta[1] as number), sd: 1 / Math.sqrt(eta[1] as number) }
  if (family.kind === 'gamma') {
    const shape = (eta[0] as number) + 1
    const rate = -(eta[1] as number)
    return { mean: shape / rate, sd: Math.sqrt(shape) / rate }
  }
  const a = (eta[0] as number) + 1
  const b = (eta[1] as number) + 1
  return { mean: a / (a + b), sd: Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1))) }
}

/** What a reference gives: the moments of the target's tilted distribution, or for a Gaussian its mean and (centred) variance. */
type Reference = readonly number[] | { readonly mean: number; readonly variance: number }

function referenceMeanSd(family: Family, reference: Reference): { mean: number; sd: number } {
  if ('variance' in reference) return { mean: reference.mean, sd: Math.sqrt(reference.variance) }
  if (family.kind === 'gaussian')
    return { mean: reference[0] as number, sd: Math.sqrt((reference[1] as number) - (reference[0] as number) ** 2) }
  return memberMeanSd(family, family.fromMoments(reference))
}

/** The mean and sd of the target's tilted distribution that the operator's message implies (its tilted moments for an operator that has only those). */
function implied(op: FactorOperator, args: readonly Arg[], to: number, params: Params): { mean: number; sd: number } {
  const c = { mode: 'ep' as const, to, args: [...args], params, power: 1 }
  const target = args[to] as Arg
  if (op.tilted !== undefined) return referenceMeanSd(target.family, [...op.tilted(c).moments])
  const message = (op.ep as NonNullable<FactorOperator['ep']>)(c).message
  const cavity = target.eta as Eta
  return memberMeanSd(
    target.family,
    cavity.map((v, i) => v + (message[i] ?? 0)),
  )
}

interface Row {
  z: number
  sd: number
  n: number
}
const table = new Map<string, Row>()
const detail: string[] = []

/** Compare the operator's tilted moments at port `to` with the reference's; the errors go into the operator's row. */
function compare(
  name: string,
  label: string,
  op: FactorOperator,
  args: readonly Arg[],
  to: number,
  params: Params,
  reference: Reference,
): { z: number; sd: number } {
  const family = (args[to] as Arg).family
  const got = implied(op, args, to, params)
  const want = referenceMeanSd(family, reference)
  const z = Math.abs(got.mean - want.mean) / want.sd
  const sd = Math.abs(got.sd / want.sd - 1)
  const row = table.get(name) ?? { z: 0, sd: 0, n: 0 }
  table.set(name, { z: Math.max(row.z, z), sd: Math.max(row.sd, sd), n: row.n + 1 })
  detail.push(
    `${name} ${label} to=${to}: z ${z.toExponential(2)} sd ${sd.toExponential(2)}${Number.isFinite(z + sd) ? '' : ` (reference ${JSON.stringify(want)} operator ${JSON.stringify(got)})`}`,
  )
  return { z, sd }
}

const normalLog = (x: number, m: number, v: number): number =>
  -0.5 * Math.log(2 * Math.PI * v) - (0.5 * (x - m) ** 2) / v
const gammaLogDensity = (t: number, shape: number, rate: number): number => shape * t - rate * Math.exp(t)

/** A window of `t = log x` for a gamma cavity that reaches far into its exponential left tail (the factors may slow it). */
const gammaBox = (shape: number, rate: number): [number, number] => {
  const centre = digammaOf(shape) - Math.log(rate)
  return [centre - 60 / Math.min(shape, 1) - 10 / Math.sqrt(shape), centre + 12 / Math.sqrt(shape) + 4]
}

const gaussianArg = (m: number, v: number): Arg => messageArg(gaussian, gaussianEta(m, v))
const gammaArg = (shape: number, rate: number): Arg => messageArg(gamma, gammaEta(shape, rate))

// ----- the scenarios -----

describe('the error of the numeric rules against a high-order reference', () => {
  it('gaussian-gamma-precision', () => {
    const cases: [number, number, number, number, number, number, number | undefined][] = [
      // x ~ N(mx, vx), u ~ N(mu, vu) (or observed), lambda ~ Gamma(shape, rate), scale, offset
      [0.7, 2, 0, 0, 3, 1.5, undefined],
      [0, 1, 3, 0.02, 0.6, 0.3, undefined],
      [1, 0.5, 0, 0, 40, 20, undefined],
      [2, 25, 0, 0, 1.2, 0.01, undefined],
      [500, 0.04, 499, 0.01, 2, 1, undefined],
    ]
    for (const [mx, vx, mu, vu, shape, rate] of cases) {
      const u: Arg = vu === 0 ? valueArg(gaussian, mu) : gaussianArg(mu, vu)
      const args = [gaussianArg(mx, vx), u, gammaArg(shape, rate)]
      const gap = mx - mu
      const joint = vx + vu
      const logw = (t: number): number => gammaLogDensity(t, shape, rate) + normalLog(0, gap, joint + Math.exp(-t))
      const r = expect1(logw, -60, 60)
      const cond = (t: number) => {
        const s = joint + Math.exp(-t)
        return { m: mx - (vx * gap) / s, v: vx - (vx * vx) / s, mu: mu + (vu * gap) / s, vu: vu - (vu * vu) / s }
      }
      const params = { scale: 1, offset: 0 }
      const label = `x~N(${mx},${vx}) u~N(${mu},${vu}) lambda~G(${shape},${rate})`
      // The mixture of the conditional updates: the mean of the variances plus the variance of the means, centred.
      const mix = (m: (t: number) => number, v: (t: number) => number) => {
        const mean = r.of(m)
        return { mean, variance: r.of(v) + r.of((t) => (m(t) - mean) ** 2) }
      }
      compare(
        'gaussian-gamma-precision',
        label,
        gaussianGammaPrecision,
        args,
        0,
        params,
        mix(
          (t) => cond(t).m,
          (t) => cond(t).v,
        ),
      )
      if (vu > 0) {
        compare(
          'gaussian-gamma-precision',
          label,
          gaussianGammaPrecision,
          args,
          1,
          params,
          mix(
            (t) => cond(t).mu,
            (t) => cond(t).vu,
          ),
        )
      }
      compare('gaussian-gamma-precision', label, gaussianGammaPrecision, args, 2, params, [
        r.of((t) => t),
        r.of((t) => Math.exp(t)),
      ])
    }
  })

  it('gamma-rate and gamma-product', () => {
    const cavities: [readonly [number, number], readonly [number, number]][] = [
      [
        [2.5, 1.2],
        [3.1, 2.4],
      ],
      [
        [0.5, 0.2],
        [0.8, 0.1],
      ],
      [
        [30, 12],
        [45, 20],
      ],
    ]
    for (const [qx, qr] of cavities) {
      const a = 0.7
      const c = 1.6
      const args = [gammaArg(...qx), gammaArg(...qr)]
      const logw = (tx: number, tr: number): number =>
        gammaLogDensity(tx, ...qx) +
        gammaLogDensity(tr, ...qr) +
        a * (Math.log(c) + tr) +
        (a - 1) * tx -
        c * Math.exp(tr) * Math.exp(tx)
      const centre = (q: readonly [number, number]): number => digammaOf(q[0]) - Math.log(q[1])
      const box = (q: readonly [number, number]): [number, number] => [
        centre(q) - 60 / Math.min(q[0], 1) - 10 / Math.sqrt(q[0]),
        centre(q) + 12 / Math.sqrt(q[0]) + 4,
      ]
      const r = expect2(logw, [box(qx), box(qr)])
      const label = `x~G(${qx}) r~G(${qr})`
      const params = { shape: a, scale: c }
      compare('gamma-rate', label, gammaRate, args, 0, params, [r.of((tx) => tx), r.of((tx) => Math.exp(tx))])
      compare('gamma-rate', label, gammaRate, args, 1, params, [r.of((_x, tr) => tr), r.of((_x, tr) => Math.exp(tr))])
      // gamma-product: p = u tau with a cavity of p
      const qp: readonly [number, number] = [2.8, 1.9]
      const pargs = [gammaArg(...qp), gammaArg(...qx), gammaArg(...qr)]
      const plogw = (tu: number, tt: number): number =>
        gammaLogDensity(tu, ...qx) + gammaLogDensity(tt, ...qr) + (qp[0] - 1) * (tu + tt) - qp[1] * Math.exp(tu + tt)
      const rp = expect2(plogw, [box(qx), box(qr)])
      for (const [to, pick] of [
        [1, (tu: number) => tu],
        [2, (_u: number, tt: number) => tt],
        [0, (tu: number, tt: number) => tu + tt],
      ] as const) {
        const f = pick as (a: number, b: number) => number
        const exp = (a: number, b: number): number => Math.exp(f(a, b))
        compare('gamma-product', label, gammaProduct, pargs, to, {}, [rp.of(f), rp.of(exp)])
      }
    }
  })

  it('dirichlet-concentration', () => {
    const cases: [number, number, number, number][] = [
      [3, 2, 2, 1],
      [8, 2, 0.5, 0.2],
      [30, 25, 15, 5],
      [1.5, 1.4, 4, 0.5],
      [300, 200, 1, 0.01],
    ]
    for (const [e1, e2, shape, rate] of cases) {
      const w = [1, 1]
      const args = [messageArg(beta, betaEta(e1, e2)), gammaArg(shape, rate)]
      const eta = [e1 - 1, e2 - 1]
      const logw = (t: number): number => {
        const a = Math.exp(t)
        let total = 0
        let totalPost = 0
        let sumLog = 0
        for (let k = 0; k < 2; k++) {
          const kappa = (w[k] as number) * a
          const post = kappa + (eta[k] as number)
          total += kappa
          totalPost += post
          sumLog += lgammaOf(post) - lgammaOf(kappa)
        }
        return gammaLogDensity(t, shape, rate) + lgammaOf(total) + sumLog - lgammaOf(totalPost)
      }
      const r = expect1(logw, -60, 60)
      const label = `p~Beta(${e1},${e2}) alpha~G(${shape},${rate})`
      const params = { weights: w }
      compare('dirichlet-concentration', label, dirichletConcentration, args, 1, params, [
        r.of((t) => t),
        r.of((t) => Math.exp(t)),
      ])
      const logMean =
        (k: number) =>
        (t: number): number => {
          const a = Math.exp(t)
          return digammaOf(a + (eta[k] as number)) - digammaOf(2 * a + (eta[0] as number) + (eta[1] as number))
        }
      compare('dirichlet-concentration', label, dirichletConcentration, args, 0, params, [
        r.of(logMean(0)),
        r.of(logMean(1)),
      ])
    }
  })

  it('gaussian-mv-precision-scale and gaussian-mv-function-scale', () => {
    const v = gaussianMv(2)
    const cases: [number[], number[], number, number][] = [
      [[0.4, -0.3], [2, 0.3, 0.3, 1.5], 3, 1.5],
      [[1.2, 0.5], [0.2, 0, 0, 0.1], 0.8, 0.2],
      [[3, 3], [40, 5, 5, 25], 20, 10],
    ]
    const scale = [1, 2]
    for (const [h, j, shape, rate] of cases) {
      const eta = [...h, ...j]
      const args = [messageArg(v, eta), gammaArg(shape, rate)]
      const logw = (t: number): number => {
        const p = scale.map((s) => s * Math.exp(t))
        const m = [
          (j[0] as number) + (p[0] as number),
          j[1] as number,
          j[2] as number,
          (j[3] as number) + (p[1] as number),
        ]
        const det = (m[0] as number) * (m[3] as number) - (m[1] as number) * (m[2] as number)
        const mu = [
          ((m[3] as number) * (h[0] as number) - (m[1] as number) * (h[1] as number)) / det,
          ((m[0] as number) * (h[1] as number) - (m[2] as number) * (h[0] as number)) / det,
        ]
        return (
          gammaLogDensity(t, shape, rate) +
          0.5 * Math.log((p[0] as number) * (p[1] as number)) -
          0.5 * Math.log(det) +
          0.5 * ((h[0] as number) * (mu[0] as number) + (h[1] as number) * (mu[1] as number))
        )
      }
      const r = expect1(logw, -60, 60)
      const label = `h=${h} J=${j} lambda~G(${shape},${rate})`
      const params = { scale }
      compare('gaussian-mv-precision-scale', label, gaussianMvPrecisionScale, args, 1, params, [
        r.of((t) => t),
        r.of((t) => Math.exp(t)),
      ])
      // The vector: the mean of its first coordinate and its sd, from the mixture of conditionals.
      const cond = (t: number) => {
        const p = scale.map((s) => s * Math.exp(t))
        const m = [
          (j[0] as number) + (p[0] as number),
          j[1] as number,
          j[2] as number,
          (j[3] as number) + (p[1] as number),
        ]
        const det = (m[0] as number) * (m[3] as number) - (m[1] as number) * (m[2] as number)
        return {
          mean: ((m[3] as number) * (h[0] as number) - (m[1] as number) * (h[1] as number)) / det,
          variance: (m[3] as number) / det,
        }
      }
      const mean0 = r.of((t) => cond(t).mean)
      const variance0 = r.of((t) => cond(t).variance) + r.of((t) => (cond(t).mean - mean0) ** 2)
      const vec = (op: FactorOperator, a: readonly Arg[], par: Params): { mean: number; sd: number } => {
        const c = { mode: 'ep' as const, to: 0, args: [...a], params: par, power: 1 }
        const m = (op.ep as NonNullable<FactorOperator['ep']>)(c).message.map((x, i) => x + (eta[i] ?? 0))
        const jn = [m[2], m[3], m[4], m[5]] as number[]
        const det = (jn[0] as number) * (jn[3] as number) - (jn[1] as number) * (jn[2] as number)
        const hn = m.slice(0, 2) as number[]
        return {
          mean: ((jn[3] as number) * (hn[0] as number) - (jn[1] as number) * (hn[1] as number)) / det,
          sd: Math.sqrt((jn[3] as number) / det),
        }
      }
      const got = vec(gaussianMvPrecisionScale, args, params)
      const want = { mean: mean0, sd: Math.sqrt(variance0) }
      const row = table.get('gaussian-mv-precision-scale') as Row
      const z = Math.abs(got.mean - want.mean) / want.sd
      const sd = Math.abs(got.sd / want.sd - 1)
      table.set('gaussian-mv-precision-scale', { z: Math.max(row.z, z), sd: Math.max(row.sd, sd), n: row.n + 1 })
      detail.push(
        `gaussian-mv-precision-scale ${label} to=0 (vector): z ${z.toExponential(2)} sd ${sd.toExponential(2)}`,
      )
    }
    // The function scale: beta_j = z_j exp(-x / 2) (one chain for each coefficient) with x Gaussian, or exp(x) a gamma.
    const steps = [
      [['mul', -0.5], ['exp']],
      [['mul', -0.5], ['exp']],
    ]
    for (const [mx, vx] of [
      [0, 0.25],
      [1.5, 4],
      [-2, 0.01],
    ] as const) {
      const [h, j] = [
        [0.4, -0.3],
        [2, 0.3, 0.3, 1.5],
      ] as const
      const eta = [...h, ...j]
      const args = [messageArg(v, eta), gaussianArg(mx, vx)]
      const logw = (x: number): number => {
        const p = Math.exp(x) // the precision of each coefficient is 1 / u^2 = exp(x)
        const m = [(j[0] as number) + p, j[1] as number, j[2] as number, (j[3] as number) + p]
        const det = (m[0] as number) * (m[3] as number) - (m[1] as number) * (m[2] as number)
        const mu = [
          ((m[3] as number) * (h[0] as number) - (m[1] as number) * (h[1] as number)) / det,
          ((m[0] as number) * (h[1] as number) - (m[2] as number) * (h[0] as number)) / det,
        ]
        return (
          normalLog(x, mx, vx) +
          Math.log(p) -
          0.5 * Math.log(det) +
          0.5 * ((h[0] as number) * (mu[0] as number) + (h[1] as number) * (mu[1] as number))
        )
      }
      const r = expect1(logw, mx - 14 * Math.sqrt(vx), mx + 14 * Math.sqrt(vx))
      compare(
        'gaussian-mv-function-scale',
        `x~N(${mx},${vx})`,
        gaussianMvFunctionScale,
        args,
        1,
        { steps },
        r.gauss((x) => x),
      )
    }
  })

  it('likelihood-pair and beta-two-shapes', () => {
    // laplace-scale: y | u, s ~ Laplace(u, exp(s)), the location Gaussian and the log scale Gaussian or the scale a gamma variable.
    const cases: [number, number, number, number, number][] = [
      [1, 0, 4, 0, 1],
      [2, 0.5, 0.2, -1, 2],
      [-1.5, 1, 9, 0.5, 0.5],
    ]
    for (const [y, mu, vu, ms, vs] of cases) {
      const args = [gaussianArg(mu, vu), gaussianArg(ms, vs)]
      const logw = (u: number, s: number): number =>
        normalLog(u, mu, vu) + normalLog(s, ms, vs) - Math.log(2) - s - Math.abs(y - u) * Math.exp(-s)
      const r = expect2(
        logw,
        [
          [mu - 10 * Math.sqrt(vu), mu + 10 * Math.sqrt(vu)],
          [ms - 10 * Math.sqrt(vs), ms + 10 * Math.sqrt(vs)],
        ],
        [[y], []],
      )
      const params = { likelihood: 'laplace-scale', y }
      const label = `y=${y} u~N(${mu},${vu}) s~N(${ms},${vs})`
      compare(
        'likelihood-pair',
        label,
        likelihoodPairOperator,
        args,
        0,
        params,
        r.gauss((u) => u),
      )
      compare(
        'likelihood-pair',
        label,
        likelihoodPairOperator,
        args,
        1,
        params,
        r.gauss((_u, s) => s),
      )
    }
    // student-t-df with a gamma degrees of freedom: y | u, df ~ t(df, u, 1).
    for (const [y, mu, vu, shape, rate] of [
      [1, 0, 2, 3, 0.5],
      [0.2, 0.1, 0.05, 1.5, 0.2],
    ] as const) {
      const args = [gaussianArg(mu, vu), gammaArg(shape, rate)]
      const logw = (u: number, t: number): number => {
        const df = Math.exp(t)
        return (
          normalLog(u, mu, vu) +
          gammaLogDensity(t, shape, rate) +
          lgammaOf((df + 1) / 2) -
          lgammaOf(df / 2) -
          0.5 * Math.log(df * Math.PI) -
          ((df + 1) / 2) * Math.log1p((y - u) ** 2 / df)
        )
      }
      const r = expect2(logw, [[mu - 10 * Math.sqrt(vu), mu + 10 * Math.sqrt(vu)], gammaBox(shape, rate)])
      const params = { likelihood: 'student-t-df', y, scale: 1 }
      const label = `y=${y} u~N(${mu},${vu}) df~G(${shape},${rate})`
      compare(
        'likelihood-pair',
        label,
        likelihoodPairOperator,
        args,
        0,
        params,
        r.gauss((u) => u),
      )
      compare('likelihood-pair', label, likelihoodPairOperator, args, 1, params, [
        r.of((_u, t) => t),
        r.of((_u, t) => Math.exp(t)),
      ])
    }
    // beta-two-shapes: p ~ Beta(a, b) with both shapes gamma variables.
    for (const [e1, e2, sa, ra, sb, rb] of [
      [3, 2, 2, 1, 3, 1.5],
      [1.5, 6, 1, 0.5, 4, 1],
      [150, 100, 2, 0.01, 2, 0.01],
    ] as const) {
      const args = [messageArg(beta, betaEta(e1, e2)), gammaArg(sa, ra), gammaArg(sb, rb)]
      const logw = (ta: number, tb: number): number => {
        const a = Math.exp(ta)
        const b = Math.exp(tb)
        return (
          gammaLogDensity(ta, sa, ra) +
          gammaLogDensity(tb, sb, rb) +
          lgammaOf(a + e1 - 1) -
          lgammaOf(a) +
          lgammaOf(b + e2 - 1) -
          lgammaOf(b) -
          lgammaOf(a + b + e1 + e2 - 2) +
          lgammaOf(a + b)
        )
      }
      const r = expect2(logw, [gammaBox(sa, ra), gammaBox(sb, rb)])
      const label = `p~Beta(${e1},${e2}) a~G(${sa},${ra}) b~G(${sb},${rb})`
      compare('beta-two-shapes', label, betaTwoShapes, args, 1, {}, [r.of((ta) => ta), r.of((ta) => Math.exp(ta))])
      compare('beta-two-shapes', label, betaTwoShapes, args, 2, {}, [
        r.of((_a, tb) => tb),
        r.of((_a, tb) => Math.exp(tb)),
      ])
      const psi = (a: number, b: number): number => digammaOf(a + b + e1 + e2 - 2)
      compare('beta-two-shapes', label, betaTwoShapes, args, 0, {}, [
        r.of((ta, tb) => digammaOf(Math.exp(ta) + e1 - 1) - psi(Math.exp(ta), Math.exp(tb))),
        r.of((ta, tb) => digammaOf(Math.exp(tb) + e2 - 1) - psi(Math.exp(ta), Math.exp(tb))),
      ])
    }
  })

  it('likelihood-triple: a Student-t with a random location, scale and degrees of freedom, an ordinal outcome with random cut points', () => {
    const cases: [string, number, Record<string, number>, [number, number][]][] = [
      [
        'student-t-full',
        2.4,
        {},
        [
          [0.3, 1],
          [-0.2, 0.4],
          [1.2, 0.5],
        ],
      ],
      [
        'student-t-full',
        -0.5,
        {},
        [
          [0.5, 0.3],
          [0, 0.2],
          [1.8, 0.3],
        ],
      ],
      [
        'ordinal-probit-cuts',
        1,
        { categories: 4 },
        [
          [0.4, 1],
          [-0.8, 0.5],
          [0.7, 0.6],
        ],
      ],
      [
        'ordinal-logit-cuts',
        2,
        { categories: 4 },
        [
          [-0.5, 1.5],
          [0.2, 0.8],
          [1.1, 0.4],
        ],
      ],
      [
        'ordinal-probit-cuts',
        1,
        { categories: 3 },
        [
          [0, 4],
          [-1, 3],
          [1, 3],
        ],
      ],
    ]
    for (const [likelihood, y, extra, cavities] of cases) {
      const params = { likelihood, y, ...extra }
      const density = tripleLikelihoods()[likelihood]?.density(y, params) as (a: number, b: number, c: number) => number
      const args = cavities.map(([m, v]) => gaussianArg(m, v))
      const logw = (a: number, b: number, c: number): number =>
        normalLog(a, cavities[0]?.[0] as number, cavities[0]?.[1] as number) +
        normalLog(b, cavities[1]?.[0] as number, cavities[1]?.[1] as number) +
        normalLog(c, cavities[2]?.[0] as number, cavities[2]?.[1] as number) +
        density(a, b, c)
      const r = expect3(
        logw,
        cavities.map(([m, v]) => [m - 9 * Math.sqrt(v), m + 9 * Math.sqrt(v)]) as [
          [number, number],
          [number, number],
          [number, number],
        ],
      )
      const label = `${likelihood} y=${y} ${cavities.map(([m, v]) => `N(${m},${v})`).join(' ')}`
      for (const to of [0, 1, 2])
        compare(
          'likelihood-triple',
          label,
          likelihoodTripleOperator,
          args,
          to,
          params,
          r.gauss((a, b, c) => [a, b, c][to] as number),
        )
    }
  })

  it('negative-binomial-dispersion', () => {
    for (const [y, mu, vu, ms, vs] of [
      [3, 1, 0.5, -1, 0.5],
      [0, 0, 1, 0, 1],
      [25, 2.5, 0.1, -2, 2],
      [7, 1, 4, 0.5, 4],
    ] as const) {
      const args = [gaussianArg(mu, vu), gaussianArg(ms, vs)]
      const logw = (u: number, s: number): number => {
        const r = Math.exp(-s)
        let rising = 0
        for (let k = 0; k < y; k++) rising += Math.log(r + k)
        let logFact = 0
        for (let k = 2; k <= y; k++) logFact += Math.log(k)
        const logMu = u
        const den = Math.max(-s, logMu) + Math.log1p(Math.exp(-Math.abs(-s - logMu)))
        return normalLog(u, mu, vu) + normalLog(s, ms, vs) + rising - logFact + r * -s - (r + y) * den + y * logMu
      }
      const r = expect2(logw, [
        [mu - 10 * Math.sqrt(vu), mu + 10 * Math.sqrt(vu)],
        [ms - 10 * Math.sqrt(vs), ms + 10 * Math.sqrt(vs)],
      ])
      const params = { y, exposure: 1 }
      const label = `y=${y} u~N(${mu},${vu}) s~N(${ms},${vs})`
      compare(
        'negative-binomial-dispersion',
        label,
        negativeBinomialDispersion,
        args,
        0,
        params,
        r.gauss((u) => u),
      )
      compare(
        'negative-binomial-dispersion',
        label,
        negativeBinomialDispersion,
        args,
        1,
        params,
        r.gauss((_u, s) => s),
      )
    }
  })

  it('gaussian-scale (x observed, Gaussian mean, Gaussian log scale)', () => {
    for (const [x, mm, vm, ms, vs] of [
      [2, 0, 1, 0, 1],
      [1000.3, 1000, 1e-4, -1, 0.5],
      [0.1, 50, 4, -3, 2],
    ] as const) {
      const args = [valueArg(gaussian, x), gaussianArg(mm, vm), gaussianArg(ms, vs)]
      const logw = (s: number): number => normalLog(s, ms, vs) + normalLog(x, mm, vm + Math.exp(2 * s))
      const r = expect1(logw, ms - 14 * Math.sqrt(vs), ms + 14 * Math.sqrt(vs))
      const post = (s: number) => {
        const noise = Math.exp(-2 * s)
        const variance = 1 / (1 / vm + noise)
        return { mean: variance * (mm / vm + x * noise), variance }
      }
      const label = `x=${x} m~N(${mm},${vm}) s~N(${ms},${vs})`
      compare(
        'gaussian-scale',
        label,
        gaussianScale,
        args,
        2,
        {},
        r.gauss((s) => s),
      )
      // The mean: a mixture of conditional Gaussians; its second moment as the centred pair (mean, variance).
      const mean = r.of((s) => post(s).mean)
      const variance = r.of((s) => post(s).variance) + r.of((s) => (post(s).mean - mean) ** 2)
      compare('gaussian-scale', label, gaussianScale, args, 1, {}, { mean, variance })
    }
  })

  it('product (the output weighted by its own message)', () => {
    for (const [ma, va, mb, vb, my, vy] of [
      [1, 0.5, 2, 0.4, 3, 1],
      [300, 0.04, 0.5, 0.01, 150, 0.0004],
      [-2, 1, 4, 2, 1, 50],
    ] as const) {
      const args = [gaussianArg(my, vy), gaussianArg(ma, va), gaussianArg(mb, vb)]
      // y = a b, with the message of y: the tilted density of y integrates a against N(y; a mb, a^2 vb) N(y; my, vy).
      const logw = (a: number): number => normalLog(a, ma, va) + normalLog(my, a * mb, vy + a * a * vb)
      const r = expect1(logw, ma - 10 * Math.sqrt(va), ma + 10 * Math.sqrt(va))
      const cond = (a: number) => {
        const s2 = a * a * vb
        const centre = a * mb
        const post = s2 > 0 ? 1 / (1 / s2 + 1 / vy) : 0
        const mp = s2 > 0 ? post * (centre / s2 + my / vy) : centre
        return { post, mp }
      }
      const mean = r.of((a) => cond(a).mp)
      const variance = r.of((a) => cond(a).post) + r.of((a) => (cond(a).mp - mean) ** 2)
      compare('product', `a~N(${ma},${va}) b~N(${mb},${vb}) y~N(${my},${vy})`, product, args, 0, {}, { mean, variance })
    }
  })

  it('likelihood-site, prior-site and apply (windows of a one-dimensional tilt)', () => {
    const site = (likelihood: string, y: number, extra: Params, density: (u: number) => number, kinks: number[]) => {
      for (const [m, v] of [
        [0, 100],
        [0.5, 1],
        [4, 0.04],
      ] as const) {
        const args = [gaussianArg(m, v)]
        const logw = (u: number): number => normalLog(u, m, v) + density(u)
        const r = expect1(logw, m - 12 * Math.sqrt(v), m + 12 * Math.sqrt(v), kinks)
        compare(
          `likelihood-site ${likelihood}`,
          `y=${y} u~N(${m},${v})`,
          likelihoodSiteOperator,
          args,
          0,
          { likelihood, y, ...extra },
          r.gauss((u) => u),
        )
      }
    }
    const lt = (y: number, s: number, df: number) => (u: number) =>
      lgammaOf((df + 1) / 2) -
      lgammaOf(df / 2) -
      0.5 * Math.log(df * Math.PI) -
      Math.log(s) -
      ((df + 1) / 2) * Math.log1p(((y - u) / s) ** 2 / df)
    site('student-t', 2, { scale: 0.1, df: 3 }, lt(2, 0.1, 3), [])
    site('laplace', 1, { scale: 0.05 }, (u) => -Math.log(0.1) - Math.abs(1 - u) / 0.05, [1])
    const logDiffPhi = (a: number, b: number): number => Math.log(Phi(b) - Phi(a))
    site('ordinal-probit', 1, { cuts: [-1, 0, 1] }, (u) => logDiffPhi(-1 - u, 0 - u), [])
    // The half-normal prior of a log scale (log: 1), the uniform on [0, 2], a truncated normal.
    for (const [m, v] of [
      [0, 4],
      [-2, 9],
      [1, 0.25],
    ] as const) {
      const s = 1
      const logw = (l: number): number =>
        normalLog(l, m, v) + Math.log(2) - 0.5 * Math.log(2 * Math.PI) - Math.log(s) - Math.exp(2 * l) / (2 * s * s) + l
      const r = expect1(logw, m - 14 * Math.sqrt(v), m + 14 * Math.sqrt(v))
      compare(
        'prior-site half-normal (log)',
        `l~N(${m},${v})`,
        priorSite,
        [gaussianArg(m, v)],
        0,
        { y: 0, base: 'half-normal', log: 1, scale: s },
        r.gauss((l) => l),
      )
    }
    for (const [m, v] of [
      [1, 0.25],
      [0.2, 4],
    ] as const) {
      const logw = (x: number): number =>
        x >= 0 && x <= 2 ? normalLog(x, m, v) - Math.log(2) : Number.NEGATIVE_INFINITY
      const r = expect1(logw, 0, 2, [])
      compare(
        'prior-site uniform',
        `x~N(${m},${v})`,
        priorSite,
        [gaussianArg(m, v)],
        0,
        { y: 0, base: 'uniform', log: 0, low: 0, high: 2 },
        r.gauss((x) => x),
      )
    }
    // apply: y = |x| (a kink at the origin), the message to x against the output message exp(h y - J y^2 / 2).
    for (const [m, v, h, J] of [
      [0.3, 1, 0.5, 2],
      [-1, 0.2, 2, 4],
    ] as const) {
      const args = [messageArg(gaussian, [h, J]), gaussianArg(m, v)]
      const logw = (x: number): number => normalLog(x, m, v) + h * Math.abs(x) - 0.5 * J * x * x
      const r = expect1(logw, m - 12 * Math.sqrt(v), m + 12 * Math.sqrt(v), [0])
      compare(
        'apply abs',
        `x~N(${m},${v}) h=${h} J=${J}`,
        apply,
        args,
        1,
        { fn: 'abs' },
        r.gauss((x) => x),
      )
    }
  })

  it('mixture-site, greater-than and apply relu (closed forms centred or floored)', () => {
    for (const [m, v, pi, s1, s2] of [
      [1, 1, 0.5, 3, 0.1],
      [1000, 1e-4, 0.5, 0.01, 0.001],
    ] as const) {
      const params = { weights: [pi, 1 - pi], means: [m - 0.05, m + 0.02], sds: [s1, s2] }
      const logw = (x: number): number =>
        normalLog(x, m, v) +
        Math.log(pi * Math.exp(normalLog(x, m - 0.05, s1 * s1)) + (1 - pi) * Math.exp(normalLog(x, m + 0.02, s2 * s2)))
      const r = expect1(logw, m - 12 * Math.sqrt(v), m + 12 * Math.sqrt(v))
      compare(
        'mixture-site',
        `x~N(${m},${v})`,
        mixtureSite,
        [gaussianArg(m, v)],
        0,
        params,
        r.gauss((x) => x),
      )
    }
    // The step factor [x > 0] with a Bernoulli outcome message: the truncated normal.
    for (const [m, v] of [
      [0.5, 1],
      [-4, 1],
      [3000, 1e-2],
    ] as const) {
      const logw = (x: number): number => normalLog(x, m, v) + Math.log(x > 0 ? 0.7 : 0.3)
      const r = expect1(logw, m - 12 * Math.sqrt(v), m + 12 * Math.sqrt(v), [0])
      const bern = messageArg(bernoulli, bernoulliEta(0.7))
      compare(
        'greater-than',
        `x~N(${m},${v})`,
        greaterThan,
        [bern, gaussianArg(m, v)],
        1,
        { threshold: 0 },
        r.gauss((x) => x),
      )
    }
  })

  it('stays under the bound of each operator and prints the table', () => {
    const lines = ['operator'.padEnd(38) + 'z (max)'.padStart(12) + 'sd (max)'.padStart(12) + 'cases'.padStart(7)]
    for (const [name, r] of [...table].sort((a, b) => a[0].localeCompare(b[0])))
      lines.push(
        name.padEnd(38) +
          r.z.toExponential(2).padStart(12) +
          r.sd.toExponential(2).padStart(12) +
          String(r.n).padStart(7),
      )
    if (process.env['OMAI_ACCURACY_PRINT'] === '1') process.stdout.write(`${lines.join('\n')}\n${detail.join('\n')}\n`)
    expect(table.size).toBe(Object.keys(bounds).length)
    // Each bound is about ten times the error the rule has now (the table of docs/probprog/review-rules-numerics.md): a rule that
    // gets worse than that has changed what it computes, and the change is a decision, not an accident.
    for (const [name, r] of table) {
      const bound = bounds[name] as number
      expect(r.z, `${name}: error of the mean in sd`).toBeLessThan(bound)
      expect(r.sd, `${name}: relative error of the sd`).toBeLessThan(bound)
    }
  })
})

/** The most the error of the mean (in standard deviations) and of the sd (relative) of each operator may be, over its scenarios. */
const bounds: Readonly<Record<string, number>> = {
  'apply abs': 1e-12,
  'beta-two-shapes': 1e-2,
  'dirichlet-concentration': 1e-9,
  'gamma-product': 1e-9,
  'gamma-rate': 1e-7,
  'gaussian-gamma-precision': 1e-8,
  'gaussian-mv-function-scale': 1e-9,
  'gaussian-mv-precision-scale': 1e-5,
  'gaussian-scale': 1e-5,
  'greater-than': 1e-6,
  'likelihood-pair': 1e-3,
  'likelihood-site laplace': 1e-4,
  'likelihood-site ordinal-probit': 1e-9,
  'likelihood-site student-t': 1e-2,
  'likelihood-triple': 1e-2,
  'mixture-site': 1e-8,
  'negative-binomial-dispersion': 1e-3,
  'prior-site half-normal (log)': 1e-9,
  'prior-site uniform': 1e-9,
  product: 1e-8,
}
