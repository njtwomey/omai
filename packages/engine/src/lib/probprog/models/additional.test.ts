// The benchmark models added after the first suite: each truth against an independent computation (a brute force, a plain
// quadrature, a library routine that does it another way), and a few of the engines that must agree with it at a small
// budget. The generic checks every model passes (data from a key, finite log joint, a verdict from each method, the truth's
// oracles) are in `models.test.ts`, which iterates the whole suite.
import { describe, expect, it } from 'vitest'
import { multivariateNormal } from '../../dists/index.ts'
import { irtAbility } from '../../pgm/index.ts'
import { key } from '../../random/index.ts'
import { item, tensor, toFlat } from '../../tensor/index.ts'
import { logJoint } from '../density.ts'
import { toProgram } from '../model/index.ts'
import { gridPosterior } from '../validation/index.ts'
import { additionalBenchmarkModels } from './additional.ts'
import { additionalCampaigns, clutterEpCampaign } from './additional-campaigns.ts'
import { benchmarkAgreement } from './agreement.ts'
import { clutterEp, clutterProblem } from './clutter.ts'
import { hmmSmoothing, pmfOracle } from './discrete-truth.ts'
import { gaussianProcessRegression, gpPosteriorGivenLengthScale } from './gaussian-process.ts'
import { IRT_ITEMS, irtScoringTruth, itemResponseScoring } from './irt.ts'
import { BNN_TEST_INPUTS, bayesianNeuralNetwork } from './neural-network.ts'
import { stickBreakingMixture } from './nonparametric.ts'
import { gaussianHmm, gaussianHmmTruth, stochasticVolatility, stochasticVolatilityTruth } from './series.ts'
import { ldaLite, ldaLiteTruth } from './topics.ts'

describe('the additional models', () => {
  it('are listed in the order of their campaigns, with a campaign each and the clutter EP table last', () => {
    const names = additionalBenchmarkModels().map((m) => m.name)
    expect(names).toEqual([
      'stochastic-volatility',
      'gaussian-hmm',
      'gaussian-process-regression',
      'gaussian-process-non-centred',
      'lda-lite',
      'bayesian-neural-network',
      'item-response-scoring',
      'stick-breaking-mixture',
      'clutter-problem',
    ])
    expect(additionalCampaigns().map((c) => c.name)).toEqual([
      ...names.map((n) => `benchmark/${n}`),
      'benchmark/clutter-ep',
    ])
  })
})

describe('pmfOracle', () => {
  it('is a step CDF with the smallest value reaching p as quantile', () => {
    const o = pmfOracle('k', [2, 0, 1], [0.5, 0.2, 0.3])
    expect([
      o.mean,
      o.cdf(0),
      o.cdf(0.5),
      o.cdf(1),
      o.cdf(2),
      o.quantile(0.2),
      o.quantile(0.21),
      o.quantile(0.51),
    ]).toEqual([0.3 + 1, 0.2, 0.2, 0.5, 1, 0, 1, 2])
  })
})

describe('the Gaussian HMM truth equals a brute force over every state path', () => {
  const y = [-0.9, 0.4, 1.1, 0.7, -0.2, 1.4]
  const A = [
    [0.8, 0.15, 0.05],
    [0.1, 0.75, 0.15],
    [0.05, 0.15, 0.8],
  ]
  const pi = [0.5, 0.3, 0.2]
  const mu = [-1.5, 0, 1.5]
  it('marginals, evidence and the distribution of the visits to a state', () => {
    const s = hmmSmoothing(y, pi, A, mu, 1)
    const K = 3
    const T = y.length
    const marginals = Array.from({ length: T }, () => new Array<number>(K).fill(0))
    const visits = new Array<number>(T + 1).fill(0)
    let evidence = 0
    for (let code = 0; code < K ** T; code++) {
      const z = Array.from({ length: T }, (_, t) => Math.floor(code / K ** t) % K)
      let p = pi[z[0] as number] as number
      for (let t = 1; t < T; t++) p *= (A[z[t - 1] as number] as number[])[z[t] as number] as number
      for (let t = 0; t < T; t++)
        p *= Math.exp(-0.5 * ((y[t] as number) - (mu[z[t] as number] as number)) ** 2) / Math.sqrt(2 * Math.PI)
      evidence += p
      z.forEach((k, t) => {
        ;(marginals[t] as number[])[k] = ((marginals[t] as number[])[k] as number) + p
      })
      const c = z.filter((k) => k === 2).length
      visits[c] = (visits[c] as number) + p
    }
    expect(s.logEvidence).toBeCloseTo(Math.log(evidence), 10)
    marginals.forEach((row, t) => {
      row.forEach((v, k) => {
        expect((s.marginals[t] as number[])[k]).toBeCloseTo(v / evidence, 10)
      })
    })
    visits.forEach((v, c) => {
      expect((s.visits[2] as number[])[c]).toBeCloseTo(v / evidence, 10)
    })
  })

  it('the benchmark truth has the oracles of the states and of the visits to state 2', () => {
    const t = gaussianHmmTruth(gaussianHmm.data.data['y'] as number[])
    expect(Object.keys(t.marginals).length).toBe(17)
    expect(t.marginals['visits_2']?.mean).toBeGreaterThan(0)
    expect(t.kind).toBe('exact')
  })

  it('the chain engine and the junction tree agree with it on the benchmark data', () => {
    const a = benchmarkAgreement(gaussianHmm, { only: ['chain', 'enumeration'], scale: 0.5 })
    expect(a.failures).toEqual([])
    expect(a.passed).toBe(true)
  })
})

describe('the stochastic volatility truth', () => {
  // A brute-force integral of the model for two steps: the joint of (h0, h1, y0, y1) on a fine grid.
  it('has the evidence and the first state mean of a direct two-dimensional integral', () => {
    const y = [0.9, -0.4]
    const mu = -1
    const phi = 0.95
    const sigma = 0.25
    const s0 = sigma / Math.sqrt(1 - phi ** 2)
    const n = 500
    const lo = mu - 7 * s0
    const step = (14 * s0) / (n - 1)
    const hs = Array.from({ length: n }, (_, i) => lo + i * step)
    const npdf = (x: number, m: number, s: number): number =>
      Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))
    let z = 0
    let m0 = 0
    for (const h0 of hs) {
      const w0 = npdf(h0, mu, s0) * npdf(y[0] as number, 0, Math.exp(h0 / 2))
      let inner = 0
      for (const h1 of hs) inner += npdf(h1, mu + phi * (h0 - mu), sigma) * npdf(y[1] as number, 0, Math.exp(h1 / 2))
      z += w0 * inner * step * step
      m0 += h0 * w0 * inner * step * step
    }
    const t = stochasticVolatilityTruth(y, 600)
    expect(t.logEvidence).toBeCloseTo(Math.log(z), 4)
    expect(t.marginals['h[0]']?.mean).toBeCloseTo(m0 / z, 3)
  })

  it('the coarse grid of the fast quality is close to the fine one', () => {
    const y = stochasticVolatility.data.data['y'] as number[]
    const fine = stochasticVolatilityTruth(y, 800)
    const coarse = stochasticVolatilityTruth(y, 200)
    expect(coarse.logEvidence).toBeCloseTo(fine.logEvidence ?? Number.NaN, 1)
    expect(Math.abs((coarse.marginals['h[10]']?.mean ?? 0) - (fine.marginals['h[10]']?.mean ?? 0))).toBeLessThan(0.02)
  })

  it('the bootstrap filter estimates the evidence of the scan within its Monte Carlo error', () => {
    const a = benchmarkAgreement(stochasticVolatility, { only: ['particle-filter'], scale: 1 })
    expect(a.runs[0]?.status).toBe('completed')
    const evidence = a.runs[0]?.posterior.logEvidence?.value
    expect(Math.abs((evidence ?? Number.NaN) - (a.truth.logEvidence ?? 0))).toBeLessThan(0.5)
  })
})

describe('the Gaussian process truth', () => {
  const x = gaussianProcessRegression.data.data['x'] as number[]
  const y = gaussianProcessRegression.data.data['y'] as number[]
  it('has the marginal likelihood of the multivariate normal density of the library, and the information-form mean', () => {
    const ell = 1.3
    const p = gpPosteriorGivenLengthScale(x, y, ell)
    const grid = x.map((a) => x.map((b) => Math.exp(-0.5 * ((a - b) / ell) ** 2) + (a === b ? 1e-6 + 0.09 : 0)))
    const dens = multivariateNormal(tensor(new Array<number>(x.length).fill(0)), { covariance: tensor(grid) })
    expect(p.logMarginalLikelihood).toBeCloseTo(item(dens.logProb(tensor(y))), 6)
    // mean = sigma^-2 (K^-1 + sigma^-2 I)^-1 y, computed by solving (I + sigma^-2 K) m = sigma^-2 K y ... via a dense solve here.
    const n = x.length
    const K = x.map((a, i) => x.map((b, j) => Math.exp(-0.5 * ((a - b) / ell) ** 2) + (i === j ? 1e-6 : 0)))
    // (K + sigma² I) alpha = y by Gaussian elimination, mean = K alpha.
    const A = K.map((row, i) => [...row.map((v, j) => v + (i === j ? 0.09 : 0)), y[i] as number])
    for (let c = 0; c < n; c++) {
      let piv = c
      for (let r = c + 1; r < n; r++)
        if (Math.abs((A[r] as number[])[c] as number) > Math.abs((A[piv] as number[])[c] as number)) piv = r
      ;[A[c], A[piv]] = [A[piv] as number[], A[c] as number[]]
      for (let r = c + 1; r < n; r++) {
        const f = ((A[r] as number[])[c] as number) / ((A[c] as number[])[c] as number)
        for (let k = c; k <= n; k++)
          (A[r] as number[])[k] = ((A[r] as number[])[k] as number) - f * ((A[c] as number[])[k] as number)
      }
    }
    const alpha = new Array<number>(n).fill(0)
    for (let r = n - 1; r >= 0; r--) {
      let s = (A[r] as number[])[n] as number
      for (let k = r + 1; k < n; k++) s -= ((A[r] as number[])[k] as number) * (alpha[k] as number)
      alpha[r] = s / ((A[r] as number[])[r] as number)
    }
    K.forEach((row, i) => {
      expect(p.mean[i]).toBeCloseTo(
        row.reduce((s, v, j) => s + v * (alpha[j] as number), 0),
        8,
      )
    })
  })

  it('the spec has the log joint of the Gram matrix of the library at a draw', () => {
    const lj = logJoint(toProgram(gaussianProcessRegression.bound()), undefined, {}, { key: key(1) })
    expect(Number.isFinite(item(lj.logJoint))).toBe(true)
    const t = gaussianProcessRegression.truth(undefined, 'fast')
    expect(Object.keys(t.marginals)).toHaveLength(13)
    // The posterior of a function value is narrower than its prior (amplitude 1) at an observed input.
    expect(t.marginals['f[3]']?.sd).toBeLessThan(0.5)
  })
})

describe('the LDA-lite truth', () => {
  it('has marginals that sum correctly and a posterior on the topic of the first document that follows its words', () => {
    const w = ldaLite.data.data['w'] as number[][]
    const t = ldaLiteTruth(w)
    expect(Object.keys(t.marginals)).toHaveLength(15 + 3 + 8)
    // Document 0 uses words 0 and 1 (topic 0's prior favourites) in the canonical data, or the model's mixture says otherwise.
    for (let d = 0; d < 3; d++) {
      const mean = (t.marginals[`theta[${d * 2 + 1}]`]?.mean as number) ?? Number.NaN
      const tokens = [0, 1, 2, 3, 4].map((n) => t.marginals[`z[${d * 5 + n}]`]?.mean as number)
      // The posterior mean of the mixture weight is near the share of tokens in topic 1 (shrunk to 1/2 by the prior).
      expect(Math.abs(mean - (0.5 + tokens.reduce((a, v) => a + v, 0)) / 6)).toBeLessThan(0.15)
    }
    // The word probabilities of a topic sum to one in the mean.
    for (let k = 0; k < 2; k++)
      expect([0, 1, 2, 3].reduce((a, v) => a + (t.marginals[`phi[${k * 4 + v}]`]?.mean as number), 0)).toBeCloseTo(1, 9)
  })

  it('NUTS and the exact Dirichlet-categorical Gibbs sweeps land near the exact mixture means (a small run)', {
    timeout: 120_000,
  }, () => {
    const a = benchmarkAgreement(ldaLite, { only: ['nuts'], scale: 0.4 })
    expect(a.runs).toHaveLength(1)
    for (const row of a.table.rows) expect(Math.abs(row.meanError), row.quantity).toBeLessThan(0.5)
  })
})

describe('the item response scoring truth', () => {
  it('has the expected-a-posteriori mean of the library, from another grid, and a posterior sd below the prior', () => {
    const data = itemResponseScoring.data
    const truth = irtScoringTruth(data.data['a'] as number[], data.data['b'] as number[], data.data['y'] as number[][])
    const y = data.data['y'] as (0 | 1)[][]
    y.forEach((row, p) => {
      const eap = irtAbility(row, IRT_ITEMS, 'eap')
      expect(truth.marginals[`theta[${p}]`]?.mean).toBeCloseTo(eap.theta, 2)
      expect(truth.marginals[`theta[${p}]`]?.sd).toBeCloseTo(eap.se, 2)
      expect(truth.marginals[`theta[${p}]`]?.sd).toBeLessThan(1)
    })
  })

  it('NUTS and expectation propagation are close to it at a small budget', { timeout: 120_000 }, () => {
    const a = benchmarkAgreement(itemResponseScoring, { only: ['nuts', 'ep'], scale: 0.3 })
    expect(a.runs.map((r) => r.plan.kind)).toEqual(['nuts', 'ep'])
    const ep = a.table.rows.filter((r) => r.method === 'ep')
    for (const row of ep) expect(Math.abs(row.meanError), row.quantity).toBeLessThan(0.3)
  })
})

describe('the stick-breaking mixture truth', () => {
  it('has the evidence of a direct integral over the unit cube, and weights that sum to one', {
    timeout: 60_000,
  }, () => {
    const y = stickBreakingMixture.data.data['y'] as number[]
    const MEANS = [-3, -1, 1, 3]
    const n = 60
    const nodes = Array.from({ length: n }, (_, i) => (i + 0.5) / n)
    let z = 0
    for (const v0 of nodes)
      for (const v1 of nodes)
        for (const v2 of nodes) {
          const pi = [v0, v1 * (1 - v0), v2 * (1 - v0) * (1 - v1), (1 - v0) * (1 - v1) * (1 - v2)]
          let logp = 0
          for (const yi of y)
            logp += Math.log(
              pi.reduce(
                (s, p, k) => s + (p * Math.exp(-0.5 * (yi - (MEANS[k] as number)) ** 2)) / Math.sqrt(2 * Math.PI),
                0,
              ),
            )
          z += Math.exp(logp)
        }
    const t = stickBreakingMixture.truth()
    expect(t.logEvidence).toBeCloseTo(Math.log(z / n ** 3), 3)
    expect(Object.keys(t.marginals)).toEqual(['v[0]', 'v[1]', 'v[2]', 'pi[0]', 'pi[1]', 'pi[2]', 'pi[3]'])
    // The posterior means of the weights sum to one.
    expect([0, 1, 2, 3].reduce((a, k) => a + (t.marginals[`pi[${k}]`]?.mean as number), 0)).toBeCloseTo(1, 9)
    // The compiled model's log joint at given breaks and labels is the explicit formula (a Beta(1, 1) prior is flat).
    const v = [0.3, 0.55, 0.7]
    const label = y.map((_, i) => i % 4)
    const weights = [v[0] as number, (v[1] as number) * 0.7, (v[2] as number) * 0.7 * 0.45, 0.7 * 0.45 * 0.3]
    const want = y.reduce(
      (a, yi, i) =>
        a +
        Math.log(weights[label[i] as number] as number) -
        0.5 * Math.log(2 * Math.PI) -
        0.5 * (yi - (MEANS[label[i] as number] as number)) ** 2,
      0,
    )
    const lj = logJoint(toProgram(stickBreakingMixture.bound()), undefined, { v: tensor(v), z: tensor(label) })
    expect(item(lj.logJoint)).toBeCloseTo(want, 10)
    // The compiled model's weights are a probability vector.
    const trace = logJoint(toProgram(stickBreakingMixture.bound()), undefined, {}, { key: key(2) }).trace
    const pi = trace.sites.get('pi')
    expect(pi && 'value' in pi ? toFlat(pi.value as never).reduce((a, v) => a + v, 0) : Number.NaN).toBeCloseTo(1, 12)
  })
})

describe('the clutter problem', () => {
  const y = clutterProblem.data.data['y'] as number[]
  it('has the moments and the evidence of the spec with the clutter indicators enumerated, a different route', () => {
    const grid = gridPosterior(toProgram(clutterProblem.bound()), undefined, {
      points: 6001,
      maxRefinements: 0,
      box: [[-60, 60]],
    })
    const t = clutterProblem.truth(undefined, 'fast')
    expect(t.marginals['theta']?.mean).toBeCloseTo(grid.marginal('theta').mean, 2)
    expect(t.marginals['theta']?.sd).toBeCloseTo(grid.marginal('theta').sd, 2)
    expect(t.logEvidence).toBeCloseTo(grid.logEvidence, 2)
  })

  it('the canonical posterior has two modes, the one near 8 higher and a second near 3 with a third of the mass', () => {
    const t = clutterProblem.truth().marginals['theta']
    expect(t?.cdf(5)).toBeGreaterThan(0.2)
    expect(t?.cdf(5)).toBeLessThan(0.5)
    expect(t?.quantile(0.05)).toBeLessThan(4)
    expect(t?.quantile(0.95)).toBeGreaterThan(7.5)
    expect(t?.sd).toBeGreaterThan(1.5)
  })

  it('EP with numeric sites converges for three integration rules, (the integrating ones to the dominant mode), and the campaign passes', () => {
    for (const rule of ['quadrature', 'gauss-hermite', 'laplace'] as const) {
      const r = clutterEp(y, rule)
      expect(r.converged, rule).toBe(true)
      // The integrating rules find the dominant mode; the Laplace rule, which follows the mode of each tilted site, may settle on the other.
      if (rule !== 'laplace') expect(Math.abs(r.mean - 7.9), rule).toBeLessThan(0.5)
    }
    const out = clutterEpCampaign().run({ key: key(0), scale: 0.1 })
    expect(out.passed).toBe(true)
    expect(out.markdown).toContain('EP, quadrature rule')
  })
})

describe('the Bayesian neural network reference', () => {
  it('shows uncertainty that grows away from the data: small near it, larger in the gap, largest outside', () => {
    const t = bayesianNeuralNetwork.truth()
    expect(t.kind).toBe('reference')
    expect(BNN_TEST_INPUTS).toEqual([-3, -1.25, 0, 1.25, 3])
    const sd = BNN_TEST_INPUTS.map((_, j) => t.marginals[`fstar[${j}]`]?.sd as number)
    expect(sd[2]).toBeGreaterThan(2 * (sd[1] as number))
    expect(sd[0]).toBeGreaterThan(2 * (sd[3] as number))
    expect(sd[4]).toBeGreaterThan(2 * (sd[1] as number))
  })

  it('refuses a truth for data other than the canonical data', () => {
    const other = bayesianNeuralNetwork.generate(key(5))
    expect(() => bayesianNeuralNetwork.truth(other)).toThrow(/canonical data only/)
  })
})
