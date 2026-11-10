// The truths are checked against independent computations: the conjugate closed forms against a brute-force integral of
// the model, the grid truths against a plain-JavaScript quadrature of the same model written here, the funnel and the
// banana against the library's own target densities (`omai-engine/datasets`), the AR(1) smoothing against the Kalman
// smoother, and the models against the library's generators (`glmCurve`, `blobs`, `eloGames`), whose data they bind.
import { describe, expect, it } from 'vitest'
import { generate } from '../../data/index.ts'
import { anisotropicBlobs, bananaDensity, blobs, eloGames, funnelDensity, glmCurve } from '../../datasets/index.ts'
import { normal as normalDist, poisson as poissonDist } from '../../dists/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { key } from '../../random/index.ts'
import { item, tensor, toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { logJoint } from '../density.ts'
import { gaussianMethod } from '../inference/index.ts'
import { toProgram } from '../model/index.ts'
import { type BenchmarkData, defineBenchmark } from './benchmark.ts'
import { betaBernoulli, gammaPoisson, normalNormal } from './conjugate.ts'
import { banana, bimodal, funnelCentred } from './geometry.ts'
import { poissonRegression } from './glm.ts'
import { gaussianMixture } from './mixture.ts'
import { exactPlan } from './plans.ts'
import { trueskillThreePlayers } from './ranking.ts'
import { eightSchoolsCentred, eightSchoolsMarginal } from './schools.ts'
import { ar1StateSpace } from './state-space.ts'
import { averageOracle, mixtureOracle } from './truth.ts'

/** The integral of `exp(f(x))` over `[lo, hi]` by the trapezoid rule, as a log. */
function logIntegral(f: (x: number) => number, lo: number, hi: number, n = 20001): number {
  const h = (hi - lo) / (n - 1)
  const values = Array.from({ length: n }, (_, i) => f(lo + i * h))
  const max = Math.max(...values)
  const total = values.reduce((a, v, i) => a + Math.exp(v - max) * (i === 0 || i === n - 1 ? 0.5 : 1), 0)
  return max + Math.log(total * h)
}

const sum = (xs: readonly number[]): number => xs.reduce((a, v) => a + v, 0)
const logNormalPdf = (x: number, m: number, s: number): number =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(s) - (x - m) ** 2 / (2 * s * s)

describe('the conjugate truths equal the closed forms and a brute-force integral of the model', () => {
  it('beta-bernoulli: Beta(2 + k, 2 + n - k), and the evidence by integrating p^k (1-p)^(n-k) Beta(p; 2, 2)', () => {
    const y = betaBernoulli.data.data['y'] as number[]
    const k = sum(y)
    const truth = betaBernoulli.truth()
    expect(truth.marginals['p']?.mean).toBeCloseTo((2 + k) / (4 + y.length), 12)
    const logBeta22 = Math.log(6)
    const brute = logIntegral(
      (p) => k * Math.log(p) + (y.length - k) * Math.log(1 - p) + Math.log(p) + Math.log(1 - p) + logBeta22,
      1e-9,
      1 - 1e-9,
    )
    expect(truth.logEvidence).toBeCloseTo(brute, 5)
  })

  it('normal-normal: the precision-weighted mean, and the evidence by integrating the likelihood against the prior', () => {
    const y = normalNormal.data.data['y'] as number[]
    const truth = normalNormal.truth()
    const precision = 1 / 4 + y.length / 2.25
    expect(truth.marginals['mu']?.mean).toBeCloseTo(sum(y) / 2.25 / precision, 12)
    expect(truth.marginals['mu']?.sd).toBeCloseTo(Math.sqrt(1 / precision), 12)
    const brute = logIntegral((mu) => logNormalPdf(mu, 0, 2) + sum(y.map((v) => logNormalPdf(v, mu, 1.5))), -20, 20)
    expect(truth.logEvidence).toBeCloseTo(brute, 6)
  })

  it('gamma-poisson: Gamma(2 + sum k, 0.5 + n), and the evidence by integration', () => {
    const k = gammaPoisson.data.data['k'] as number[]
    const truth = gammaPoisson.truth()
    expect(truth.marginals['rate']?.mean).toBeCloseTo((2 + sum(k)) / (0.5 + k.length), 12)
    const logPrior = (r: number): number => 2 * Math.log(0.5) - item(lgamma(2)) + Math.log(r) - 0.5 * r
    const logLik = (r: number): number => sum(k.map((c) => c * Math.log(r) - r - item(lgamma(c + 1))))
    expect(truth.logEvidence).toBeCloseTo(
      logIntegral((r) => logPrior(r) + logLik(r), 1e-9, 40),
      6,
    )
  })
})

describe('the quadrature truths equal a plain-JavaScript quadrature of the same model', () => {
  it('poisson regression: the means of a and b and the evidence on a fine grid of (a, b)', () => {
    const x = poissonRegression.data.data['x'] as number[]
    const k = poissonRegression.data.data['k'] as number[]
    const logK = sum(k.map((c) => item(lgamma(c + 1))))
    const na = 301
    const lo = [-1, 0.3]
    const hi = [2, 1.2]
    const ha = ((hi[0] as number) - (lo[0] as number)) / (na - 1)
    const hb = ((hi[1] as number) - (lo[1] as number)) / (na - 1)
    const logPost = (a: number, b: number): number =>
      logNormalPdf(a, 0, 2) +
      logNormalPdf(b, 0, 2) +
      sum(x.map((xi, i) => (k[i] ?? 0) * (a + b * xi) - Math.exp(a + b * xi))) -
      logK
    const cells: { a: number; b: number; lp: number }[] = []
    for (let i = 0; i < na; i++)
      for (let j = 0; j < na; j++) cells.push({ a: (lo[0] as number) + i * ha, b: (lo[1] as number) + j * hb, lp: 0 })
    for (const c of cells) c.lp = logPost(c.a, c.b)
    const max = Math.max(...cells.map((c) => c.lp))
    const w = cells.map((c) => Math.exp(c.lp - max))
    const total = sum(w)
    const meanA = sum(cells.map((c, i) => c.a * (w[i] as number))) / total
    const meanB = sum(cells.map((c, i) => c.b * (w[i] as number))) / total
    const truth = poissonRegression.truth()
    expect(truth.marginals['a']?.mean).toBeCloseTo(meanA, 3)
    expect(truth.marginals['b']?.mean).toBeCloseTo(meanB, 3)
    expect(truth.logEvidence).toBeCloseTo(max + Math.log(total * ha * hb), 3)
  })

  it('eight schools: mu, tau and the school effects from the likelihood N(y; mu, sigma² + tau²) and the precision-weighted effects', () => {
    const y = eightSchoolsCentred.data.data['y'] as number[]
    const sigma = eightSchoolsCentred.data.data['sigma'] as number[]
    const truth = eightSchoolsCentred.truth(undefined, 'fast')
    const n = 241
    const mus = Array.from({ length: n }, (_, i) => -12 + (30 * i) / (n - 1))
    const taus = Array.from({ length: n }, (_, i) => 0.0005 + (45 * i) / (n - 1))
    const hm = (mus[1] as number) - (mus[0] as number)
    const ht = (taus[1] as number) - (taus[0] as number)
    let z = 0
    let mMu = 0
    let mTau = 0
    let mTheta0 = 0
    let max = Number.NEGATIVE_INFINITY
    const cells: { mu: number; tau: number; lp: number }[] = []
    for (const mu of mus)
      for (const tau of taus) {
        // Half-Cauchy(5) density on tau, normal(0, 5) on mu, N(y_j; mu, sigma_j² + tau²) likelihood.
        const lp =
          logNormalPdf(mu, 0, 5) +
          Math.log(2 / (Math.PI * 5 * (1 + (tau / 5) ** 2))) +
          sum(y.map((yj, j) => logNormalPdf(yj, mu, Math.sqrt((sigma[j] as number) ** 2 + tau * tau))))
        cells.push({ mu, tau, lp })
        max = Math.max(max, lp)
      }
    cells.forEach((c, i) => {
      // The trapezoid rule: the cells on the edge of the grid weigh half (the density is not zero at tau = 0).
      const row = Math.floor(i / n)
      const col = i % n
      const edge = (row === 0 || row === n - 1 ? 0.5 : 1) * (col === 0 || col === n - 1 ? 0.5 : 1)
      const w = Math.exp(c.lp - max) * edge
      z += w
      mMu += w * c.mu
      mTau += w * c.tau
      const s2 = (sigma[0] as number) ** 2
      const t2 = c.tau * c.tau
      mTheta0 += (w * (c.mu * s2 + (y[0] as number) * t2)) / (s2 + t2)
    })
    expect(truth.marginals['mu']?.mean).toBeCloseTo(mMu / z, 1)
    expect(truth.marginals['tau']?.mean).toBeCloseTo(mTau / z, 1)
    expect(truth.marginals['theta[0]']?.mean).toBeCloseTo(mTheta0 / z, 1)
    expect(truth.logEvidence).toBeCloseTo(max + Math.log(z * hm * ht), 1)
    // The marginal spec is what the truth integrates: the same two latents as the full model's hyperparameters.
    expect(eightSchoolsMarginal.spec.nodes.filter((nd) => nd.kind === 'latent').map((nd) => nd.name)).toEqual([
      'mu',
      'tau',
    ])
  })
})

describe('the geometry specs are the library densities', () => {
  it('the banana spec has the log density of bananaDensity', () => {
    const program = toProgram(banana.bound())
    for (const [x0, x1] of [
      [0.3, -1.5],
      [-2.2, 3.1],
      [3.5, 14.0],
    ] as const) {
      const lj = item(logJoint(program, undefined, { x0, x1 }).logJoint)
      expect(lj).toBeCloseTo(item(bananaDensity().logProb(tensor([[x0, x1]]))), 10)
    }
  })

  it('the funnel spec has the log density of funnelDensity on (v, x0), plus the other eight coordinates', () => {
    const program = toProgram(funnelCentred.bound())
    const v = 0.7
    const x = [0.4, -0.3, 0.9, 0.1, -1.2, 0.5, 0.2, -0.6, 0.8]
    const rest = sum(x.slice(1).map((xi) => logNormalPdf(xi, 0, Math.exp(v / 2))))
    const lj = item(logJoint(program, undefined, { v, x }).logJoint)
    expect(lj).toBeCloseTo(item(funnelDensity(3).logProb(tensor([[v, x[0] as number]]))) + rest, 10)
  })

  it('the two-mode target has the mixture density of two unit Gaussians on the diagonal (the label summed out)', () => {
    const program = toProgram(bimodal.bound())
    const x = [0.5, -0.2]
    // The joint at each label is log(1/2) + the two normal densities; summing the label out gives the mixture density.
    const joint = (z: number): number => item(logJoint(program, undefined, { x, z }).logJoint)
    const mode = (m: number): number =>
      Math.log(0.5) + logNormalPdf(x[0] as number, m, 1) + logNormalPdf(x[1] as number, m, 1)
    expect(joint(0)).toBeCloseTo(mode(-3), 10)
    expect(joint(1)).toBeCloseTo(mode(3), 10)
    const o = bimodal.truth().marginals['x[0]']
    expect(o?.cdf(0)).toBeCloseTo(0.5, 12)
    expect(o?.mean).toBeCloseTo(0, 12)
  })

  it('the funnel truth has the exact moments: v is N(0, 9), x has sd exp(9/4)', () => {
    const t = funnelCentred.truth()
    expect(t.marginals['v']?.sd).toBe(3)
    expect(t.marginals['x[3]']?.sd).toBeCloseTo(Math.exp(2.25), 12)
    expect(t.marginals['x[3]']?.cdf(0)).toBeCloseTo(0.5, 2)
  })
})

describe('the AR(1) smoothing truth equals the Kalman smoother', () => {
  it('the means, the sds and the evidence agree with the gaussian method in its Kalman mode', () => {
    const truth = ar1StateSpace.truth()
    const method = gaussianMethod(ar1StateSpace.program(), {})
    const s = session(method, { config: { method: 'kalman' }, steps: 3, mode: 'eager' })
    const post = method.posterior?.(s.state, s.config)
    for (let t = 0; t < 12; t++) {
      expect(item(post?.mean(`x${t}`) as never)).toBeCloseTo(truth.marginals[`x${t}`]?.mean ?? Number.NaN, 8)
      expect(item(post?.stddev(`x${t}`) as never)).toBeCloseTo(truth.marginals[`x${t}`]?.sd ?? Number.NaN, 8)
    }
    expect(post?.logEvidence?.value).toBeCloseTo(truth.logEvidence ?? Number.NaN, 8)
  })
})

describe('the models bind the library generators', () => {
  it('glmCurve data bind to the Poisson regression, whose posterior favours the generating coefficients', () => {
    const { data, truth } = generate(glmCurve, {
      key: key(5),
      knobs: { n: 60, family: 'poisson', intercept: 0.3, slope: 0.8 },
    })
    const bound = poissonRegression.definition?.bind({ x: toFlat(data.x).filter((_, i) => i < 60), k: toFlat(data.y) })
    expect(bound?.sizes).toEqual({ row: 60 })
    const program = toProgram(bound as NonNullable<typeof bound>)
    const at = (a: number, b: number): number => item(logJoint(program, undefined, { a, b }).logJoint)
    const extra = truth.extra as { intercept: number; slope: number }
    expect(at(extra.intercept, extra.slope)).toBeGreaterThan(at(-1.5, 2))
    expect(at(extra.intercept, extra.slope)).toBeGreaterThan(at(1.5, -0.5))
  })

  it('blobs data (two centres, one feature) bind to the Gaussian mixture', () => {
    const { data } = generate(blobs, {
      key: key(6),
      knobs: { n: 40, nFeatures: 1, centers: 2, clusterStd: 1, boxSize: 4 },
    })
    const bound = gaussianMixture.definition?.bind({ y: toFlat(data.x) }, { sizes: { component: 2 } })
    expect(bound?.sizes).toEqual({ row: 40, component: 2 })
    expect(
      Number.isFinite(
        item(logJoint(toProgram(bound as NonNullable<typeof bound>), undefined, {}, { key: key(1) }).logJoint),
      ),
    ).toBe(true)
    void anisotropicBlobs
  })

  it('eloGames data (three players, no draws) bind to the TrueSkill model as index data', () => {
    const { data } = generate(eloGames, { key: key(7), knobs: { players: 3, games: 30, drawWeight: 0, spread: 200 } })
    const parts = toFlat(data.participants)
    const ranks = toFlat(data.ranks)
    const a = Array.from({ length: 30 }, (_, g) => parts[2 * g] as number)
    const b = Array.from({ length: 30 }, (_, g) => parts[2 * g + 1] as number)
    const win = Array.from({ length: 30 }, (_, g) => ((ranks[2 * g] as number) < (ranks[2 * g + 1] as number) ? 1 : 0))
    const bound = trueskillThreePlayers.definition?.bind({ a, b, win }, { sizes: { player: 3 } })
    expect(bound?.sizes).toEqual({ player: 3, game: 30 })
    expect(
      Number.isFinite(
        item(logJoint(toProgram(bound as NonNullable<typeof bound>), undefined, {}, { key: key(1) }).logJoint),
      ),
    ).toBe(true)
  })
})

describe('the oracles of mixtures', () => {
  it('a mixtureOracle of many components tabulates and agrees with the exact sum', () => {
    const comps = Array.from({ length: 80 }, (_, i) => ({ weight: 1, mean: i / 10, sd: 0.5 }))
    const o = mixtureOracle('x', comps)
    const exact = (x: number): number => sum(comps.map((c) => item(normalDist(c.mean, c.sd).cdf(x)))) / comps.length
    for (const x of [0, 2, 4, 7]) expect(o.cdf(x)).toBeCloseTo(exact(x), 3)
    expect(Math.abs(o.cdf(o.quantile(0.3)) - 0.3)).toBeLessThan(1e-3)
  })

  it('an averageOracle of two marginals is their even mixture', () => {
    const lo = mixtureOracle('lo', [{ weight: 1, mean: -2, sd: 1 }])
    const hi = mixtureOracle('hi', [{ weight: 1, mean: 2, sd: 1 }])
    const both = averageOracle('mu', [
      { weight: 1, oracle: lo },
      { weight: 1, oracle: hi },
    ])
    expect(both.mean).toBe(0)
    expect(both.sd).toBeCloseTo(Math.sqrt(5), 12)
    expect(both.cdf(0)).toBeCloseTo(0.5, 12)
    expect(both.quantile(0.5)).toBeCloseTo(0, 6)
  })
})

describe('defineBenchmark checks its parts', () => {
  const parts = {
    title: 't',
    description: 'd',
    tags: [],
    references: [],
    generate: (): BenchmarkData => ({ data: {}, generating: {} }),
    truth: () => ({ kind: 'exact' as const, source: 's', marginals: {} }),
    plans: () => [exactPlan('conjugate')],
  }
  it('refuses a name that is not lower case with dashes', () => {
    expect(() => defineBenchmark({ ...parts, name: 'Bad Name', program: () => () => undefined })).toThrow(
      /lower case with dashes/,
    )
  })
  it('wants a spec or a program, not both and not neither', () => {
    expect(() => defineBenchmark({ ...parts, name: 'neither' })).toThrow(/not both or neither/)
    expect(() => poissonRegression.bound({ data: { x: [1], k: [1, 2] }, generating: {} })).toThrow(/dimension 'row'/)
    expect(() => ar1StateSpace.bound()).toThrow(/is a program/)
  })
  it('names the oracles of a poisson check: only the dists used here are imported', () => {
    expect(item(poissonDist(2).prob(0))).toBeCloseTo(Math.exp(-2), 12)
  })
})
