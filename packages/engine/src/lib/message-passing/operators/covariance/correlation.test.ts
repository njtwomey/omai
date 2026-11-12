import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { beta, betaEta, gamma, gammaEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Eta, OperatorContext } from '../../core/types.ts'
import { choleskyInPlace, choleskyLogDet, choleskySolveInPlace } from '../../numerics/_bartlett.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { mvnSdCorr } from './correlation.ts'

const G = gaussianMv(2)
const ROWS = [
  [1.1, 0.6],
  [-0.4, -0.2],
  [0.9, 1.3],
  [0.2, 0.1],
  [-1.2, -0.8],
  [0.5, 0.9],
]
const n = ROWS.length
const ybar = [0, 1].map((k) => ROWS.reduce((s, y) => s + (y[k] as number), 0) / n)
const scatter = [0, 0, 0, 0]
for (const y of ROWS)
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++)
      scatter[i * 2 + j] =
        (scatter[i * 2 + j] as number) +
        ((y[i] as number) - (ybar[i] as number)) * ((y[j] as number) - (ybar[j] as number))

const ctx = (to: number, args: readonly Arg[], match?: 'moments' | 'log-moments'): OperatorContext => ({
  mode: 'ep',
  to,
  args,
  params: { count: n, scatter, ...(match === undefined ? {} : { match }) },
  power: 1,
})

/** log N(vec(Y); 1 (x) m, I (x) Sigma + 1 1' (x) V): the rows of a shared random mean with cavity N(m, V), all integrated out. */
function logJoint(sigma: readonly number[], m: readonly number[], v: readonly number[]): number {
  const dim = 2 * n
  const a = new Float64Array(dim * dim)
  const r = new Float64Array(dim)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let p = 0; p < 2; p++)
        for (let q = 0; q < 2; q++)
          a[(2 * i + p) * dim + 2 * j + q] = (i === j ? (sigma[p * 2 + q] as number) : 0) + (v[p * 2 + q] as number)
  for (let i = 0; i < n; i++)
    for (let p = 0; p < 2; p++) r[2 * i + p] = ((ROWS[i] as number[])[p] as number) - (m[p] as number)
  if (!choleskyInPlace(a, dim)) return Number.NEGATIVE_INFINITY
  const w = Float64Array.from(r)
  choleskySolveInPlace(a, dim, w)
  let quad = 0
  for (let i = 0; i < dim; i++) quad += (r[i] as number) * (w[i] as number)
  return -0.5 * (dim * Math.log(2 * Math.PI) + choleskyLogDet(a, dim) + quad)
}

/** The brute-force integral over (t1, t2, u) on a midpoint grid against Gamma, Gamma and Beta cavities. */
function brute(
  cav: { a1: number; b1: number; a2: number; b2: number; p: number; q: number },
  mean: { m: readonly number[]; v: readonly number[] },
  box: readonly [readonly [number, number], readonly [number, number], readonly [number, number]],
) {
  const N = 70
  const steps = box.map((b) => (b[1] - b[0]) / N)
  let total = 0
  const acc = { lt1: 0, t1: 0, lt2: 0, t2: 0, lu: 0, l1u: 0, u: 0, u2: 0 }
  const logGamma = (x: number, a: number, b: number): number =>
    a * Math.log(b) - lgammaOf(a) + (a - 1) * Math.log(x) - b * x
  const logBeta = (u: number, p: number, q: number): number =>
    lgammaOf(p + q) - lgammaOf(p) - lgammaOf(q) + (p - 1) * Math.log(u) + (q - 1) * Math.log(1 - u)
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      for (let k = 0; k < N; k++) {
        const t1 = (box[0][0] as number) + (i + 0.5) * (steps[0] as number)
        const t2 = (box[1][0] as number) + (j + 0.5) * (steps[1] as number)
        const u = (box[2][0] as number) + (k + 0.5) * (steps[2] as number)
        const rho = 2 * u - 1
        const s1 = 1 / Math.sqrt(t1)
        const s2 = 1 / Math.sqrt(t2)
        const sigma = [s1 * s1, rho * s1 * s2, rho * s1 * s2, s2 * s2]
        const lw =
          logGamma(t1, cav.a1, cav.b1) +
          logGamma(t2, cav.a2, cav.b2) +
          logBeta(u, cav.p, cav.q) +
          logJoint(sigma, mean.m, mean.v)
        const w = Math.exp(lw)
        total += w
        acc.lt1 += w * Math.log(t1)
        acc.t1 += w * t1
        acc.lt2 += w * Math.log(t2)
        acc.t2 += w * t2
        acc.lu += w * Math.log(u)
        acc.l1u += w * Math.log(1 - u)
        acc.u += w * u
        acc.u2 += w * u * u
      }
  const cell = (steps[0] as number) * (steps[1] as number) * (steps[2] as number)
  return {
    logMass: Math.log(total * cell),
    lt1: acc.lt1 / total,
    t1: acc.t1 / total,
    lt2: acc.lt2 / total,
    t2: acc.t2 / total,
    lu: acc.lu / total,
    l1u: acc.l1u / total,
    u: acc.u / total,
    uVar: acc.u2 / total - (acc.u / total) ** 2,
  }
}

describe('mvn-sd-corr', () => {
  const cav = { a1: 3, b1: 2.5, a2: 2.5, b2: 2, p: 2, q: 2 }
  const args = (meanArg: Arg): Arg[] => [
    valueArg(G, ybar),
    meanArg,
    messageArg(gamma, gammaEta(cav.a1, cav.b1)),
    messageArg(gamma, gammaEta(cav.a2, cav.b2)),
    messageArg(beta, betaEta(cav.p, cav.q)),
  ]

  it('a known mean: the messages to both precisions and the correlation (log moments) match the brute-force three-dimensional integral', () => {
    const a = args(valueArg(G, [0.1, 0.2]))
    const b = brute(cav, { m: [0.1, 0.2], v: [1e-12, 0, 0, 1e-12] }, [
      [0.02, 9],
      [0.02, 9],
      [0.002, 0.998],
    ])
    const t1 = gamma.moments(gamma.product(gammaEta(cav.a1, cav.b1), mvnSdCorr.ep?.(ctx(2, a)).message as number[]))
    const t2 = gamma.moments(gamma.product(gammaEta(cav.a2, cav.b2), mvnSdCorr.ep?.(ctx(3, a)).message as number[]))
    const u = beta.moments(
      beta.product(betaEta(cav.p, cav.q), mvnSdCorr.ep?.(ctx(4, a, 'log-moments')).message as number[]),
    )
    // the gamma that matches (E log t, E t) has exactly those moments
    expect(t1[0]).toBeCloseTo(b.lt1, 2)
    expect(t1[1]).toBeCloseTo(b.t1, 2)
    expect(t2[0]).toBeCloseTo(b.lt2, 2)
    expect(t2[1]).toBeCloseTo(b.t2, 2)
    expect(u[0]).toBeCloseTo(b.lu, 2)
    expect(u[1]).toBeCloseTo(b.l1u, 2)
    const massOf =
      gamma.logPartition(gammaEta(cav.a1, cav.b1)) +
      gamma.logPartition(gammaEta(cav.a2, cav.b2)) +
      beta.logPartition(betaEta(cav.p, cav.q))
    expect(mvnSdCorr.logEvidence?.(ctx(2, a)) as number).toBeCloseTo(b.logMass + massOf, 2)
  })

  it('a random mean: the same integral with the mean integrated out, and the message to the mean is proper', () => {
    const m = [0.1, 0.2]
    const v = [0.15, 0.04, 0.04, 0.2]
    const meta = gaussianMvEta(m, v)
    const a = args(messageArg(G, meta))
    const b = brute(cav, { m, v }, [
      [0.02, 9],
      [0.02, 9],
      [0.002, 0.998],
    ])
    const u = beta.moments(
      beta.product(betaEta(cav.p, cav.q), mvnSdCorr.ep?.(ctx(4, a, 'log-moments')).message as number[]),
    )
    const t1 = gamma.moments(gamma.product(gammaEta(cav.a1, cav.b1), mvnSdCorr.ep?.(ctx(2, a)).message as number[]))
    expect(u[0]).toBeCloseTo(b.lu, 2)
    expect(u[1]).toBeCloseTo(b.l1u, 2)
    expect(t1[1]).toBeCloseTo(b.t1, 2)
    const toMean = mvnSdCorr.ep?.(ctx(1, a)).message as number[]
    expect(G.isProper(G.product(meta, toMean))).toBe(true)
    // the data pull the mean from (0.1, 0.2) towards the sample mean
    expect(G.moments(G.product(meta, toMean))[0]).toBeGreaterThan(0.1)
  })

  it('known standard deviations make it a one-dimensional integral over the correlation', () => {
    const a: Arg[] = [
      valueArg(G, ybar),
      valueArg(G, [0, 0]),
      valueArg(gamma, 1),
      valueArg(gamma, 1),
      messageArg(beta, betaEta(1, 1)),
    ]
    const msg = mvnSdCorr.ep?.(ctx(4, a, 'log-moments')).message as number[]
    // the posterior of rho under the uniform prior is proportional to (1 - rho^2)^(-n / 2) exp(-(s11 - 2 rho s12 + s22 + n ybar...) / (2 (1 - rho^2)))
    const sxx = ROWS.reduce((s, y) => s + (y[0] as number) ** 2, 0)
    const syy = ROWS.reduce((s, y) => s + (y[1] as number) ** 2, 0)
    const sxy = ROWS.reduce((s, y) => s + (y[0] as number) * (y[1] as number), 0)
    let z = 0
    let lu = 0
    const N = 4000
    for (let i = 0; i < N; i++) {
      const rho = -1 + (2 * (i + 0.5)) / N
      const lw = -(n / 2) * Math.log(1 - rho * rho) - (sxx - 2 * rho * sxy + syy) / (2 * (1 - rho * rho))
      const w = Math.exp(lw)
      z += w
      lu += w * Math.log((1 + rho) / 2)
    }
    expect(beta.moments(beta.product(betaEta(1, 1), msg))[0]).toBeCloseTo(lu / z, 3)
  })

  /** The mean and standard deviation of `u` under a beta given by its natural parameters. */
  const betaMeanSd = (eta: readonly number[]): [number, number] => {
    const [p, q] = [(eta[0] as number) + 1, (eta[1] as number) + 1]
    return [p / (p + q), Math.sqrt((p * q) / ((p + q) ** 2 * (p + q + 1)))]
  }

  it('by default the belief has the mean and variance of u: against the three-dimensional brute-force integral, known and random mean', () => {
    const boxes = [
      [0.02, 9],
      [0.02, 9],
      [0.002, 0.998],
    ] as const
    for (const mean of [
      { arg: valueArg(G, [0.1, 0.2]), m: [0.1, 0.2], v: [1e-12, 0, 0, 1e-12] },
      {
        arg: messageArg(G, gaussianMvEta([0.1, 0.2], [0.15, 0.04, 0.04, 0.2])),
        m: [0.1, 0.2],
        v: [0.15, 0.04, 0.04, 0.2],
      },
    ]) {
      const a = args(mean.arg)
      const b = brute(cav, mean, boxes)
      const [m, sd] = betaMeanSd(beta.product(betaEta(cav.p, cav.q), mvnSdCorr.ep?.(ctx(4, a)).message as number[]))
      expect(m).toBeCloseTo(b.u, 3)
      expect(sd).toBeCloseTo(Math.sqrt(b.uVar), 3)
    }
  })

  it('known standard deviations and a strong correlation from few pairs: the spread is the exact one, the log moments are narrow', () => {
    // four pairs from a correlation near 0.97 (unit variances), under the uniform prior: p(rho | y) is known up to a constant
    const rows = [
      [1.0, 0.9],
      [-0.8, -0.85],
      [0.3, 0.4],
      [1.4, 1.25],
    ]
    const k = rows.length
    const yb = [0, 1].map((j) => rows.reduce((t, y) => t + (y[j] as number), 0) / k)
    const sc = [0, 0, 0, 0]
    for (const y of rows)
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          sc[i * 2 + j] =
            (sc[i * 2 + j] as number) + ((y[i] as number) - (yb[i] as number)) * ((y[j] as number) - (yb[j] as number))
    const sxx = rows.reduce((t, y) => t + (y[0] as number) ** 2, 0)
    const syy = rows.reduce((t, y) => t + (y[1] as number) ** 2, 0)
    const sxy = rows.reduce((t, y) => t + (y[0] as number) * (y[1] as number), 0)
    let z = 0
    let m1 = 0
    let m2 = 0
    const N = 20000
    for (let i = 0; i < N; i++) {
      const rho = -1 + (2 * (i + 0.5)) / N
      const w = Math.exp(-(k / 2) * Math.log(1 - rho * rho) - (sxx - 2 * rho * sxy + syy) / (2 * (1 - rho * rho)))
      z += w
      m1 += w * rho
      m2 += w * rho * rho
    }
    const exact = { mean: m1 / z, sd: Math.sqrt(m2 / z - (m1 / z) ** 2) }
    const a: Arg[] = [
      valueArg(G, yb),
      valueArg(G, [0, 0]),
      valueArg(gamma, 1),
      valueArg(gamma, 1),
      messageArg(beta, betaEta(1, 1)),
    ]
    const spread = (match: 'moments' | 'log-moments'): { mean: number; ratio: number } => {
      const c = { ...ctx(4, a, match), params: { count: k, scatter: sc, match } }
      const [m, sd] = betaMeanSd(beta.product(betaEta(1, 1), mvnSdCorr.ep?.(c).message as number[]))
      return { mean: 2 * m - 1, ratio: (2 * sd) / exact.sd }
    }
    expect(spread('moments').mean).toBeCloseTo(exact.mean, 4)
    // (the integral is a panelled grid: the spread is the exact one to 1e-3)
    expect(Math.abs(spread('moments').ratio - 1)).toBeLessThan(2e-3)
    // the beta of the log moments is the KL projection: right mean, a spread a fifth too small here
    expect(spread('log-moments').ratio).toBeLessThan(0.9)
    expect(Math.abs(spread('log-moments').mean - exact.mean)).toBeLessThan(0.05 * exact.sd * 4)
  })

  it('is refused at a variational rule, and the check names the dimension', () => {
    expect(mvnSdCorr.noRule?.vmp).toMatch(/closed form/)
    expect(mvnSdCorr.check?.({}, [gaussianMv(3), gaussianMv(3), gamma, gamma, beta])).toMatch(/two entries/)
    expect(mvnSdCorr.check?.({}, [G, G, gamma, gamma, beta])).toBeUndefined()
  })

  it('a flat cavity on the mean: the first message is a proper Gaussian', () => {
    const a = args(messageArg(G, G.flat as Eta))
    const toMean = mvnSdCorr.ep?.(ctx(1, a)).message as number[]
    expect(G.isProper(toMean)).toBe(true)
  })
})
