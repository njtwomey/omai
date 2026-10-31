import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { iterate, runAlgorithm, stateFromJson, stateToJson } from '../../trace/index.ts'
import { forwardBackward } from '../inference/chain.ts'
import { hmmChain } from '../models/state-space.ts'
import { type HmmParameters, hmmFromParameters } from './baum-welch.ts'
import {
  fitNegativeBinomialDurations,
  fitPoissonDurations,
  geometricDurations,
  type HsmmParameters,
  hsmmAlgorithm,
  hsmmFromHmm,
  hsmmLogLikelihood,
  hsmmPosterior,
  hsmmSample,
  negativeBinomialDurations,
  poissonDurations,
} from './hsmm.ts'

// Exact enumeration of every segmentation (states, durations, the last segment running on past the data), the truth.
function enumerate(
  p: HsmmParameters,
  x: readonly number[],
): { logLikelihood: number; occupancy: number[][]; transition: number[][]; duration: number[][]; start: number[] } {
  const T = x.length
  const K = p.start.length
  const D = p.duration[0]?.length ?? 0
  const emit = (j: number, a: number, b: number): number => {
    let v = 1
    for (let t = a; t < b; t++) v *= (p.kind === 'categorical' ? p.emission[j]?.[x[t] ?? 0] : 0) ?? 0
    return v
  }
  const occ = Array.from({ length: T }, () => new Array<number>(K).fill(0))
  const trans = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  const dur = Array.from({ length: K }, () => new Array<number>(D).fill(0))
  const first = new Array<number>(K).fill(0)
  let total = 0
  type Seg = { j: number; s: number; tau: number }
  const walk = (pos: number, prev: number, segs: Seg[], prob: number): void => {
    for (let j = 0; j < K; j++) {
      const w = prev < 0 ? (p.start[j] ?? 0) : (p.transition[prev]?.[j] ?? 0)
      if (w === 0) continue
      for (let tau = 1; tau <= D; tau++) {
        const q = prob * w * (p.duration[j]?.[tau - 1] ?? 0) * emit(j, pos, Math.min(T, pos + tau))
        if (q === 0) continue
        const next = [...segs, { j, s: pos, tau }]
        if (pos + tau >= T) {
          total += q
          const weight = q
          next.forEach((g, n) => {
            for (let t = g.s; t < Math.min(T, g.s + g.tau); t++)
              (occ[t] as number[])[g.j] = (occ[t]?.[g.j] ?? 0) + weight
            ;(dur[g.j] as number[])[g.tau - 1] = (dur[g.j]?.[g.tau - 1] ?? 0) + weight
            if (n > 0) {
              const a = next[n - 1]?.j ?? 0
              ;(trans[a] as number[])[g.j] = (trans[a]?.[g.j] ?? 0) + weight
            }
          })
          first[next[0]?.j ?? 0] = (first[next[0]?.j ?? 0] ?? 0) + weight
        } else walk(pos + tau, j, next, q)
      }
    }
  }
  walk(0, -1, [], 1)
  return {
    logLikelihood: Math.log(total),
    occupancy: occ.map((r) => r.map((v) => v / total)),
    transition: trans.map((r) => r.map((v) => v / total)),
    duration: dur.map((r) => r.map((v) => v / total)),
    start: first.map((v) => v / total),
  }
}

const tiny: HsmmParameters = {
  kind: 'categorical',
  start: [0.6, 0.3, 0.1],
  transition: [
    [0, 0.7, 0.3],
    [0.4, 0, 0.6],
    [0.5, 0.5, 0],
  ],
  emission: [
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
  ],
  duration: [
    [0.2, 0.5, 0.3, 0],
    [0.6, 0.1, 0.1, 0.2],
    [0, 0.3, 0.3, 0.4],
  ],
}

describe('hidden semi-Markov forward-backward', () => {
  it('matches exact enumeration of all segmentations: log likelihood, occupancy, moves, durations, start', () => {
    for (const x of [[0], [0, 1], [0, 1, 1, 2], [2, 0, 0, 1, 2, 1], [1, 1, 0, 2, 0, 0, 1]]) {
      const post = hsmmPosterior(tiny, x)
      const truth = enumerate(tiny, x)
      expect(post.logLikelihood).toBeCloseTo(truth.logLikelihood, 11)
      for (let t = 0; t < x.length; t++)
        for (let k = 0; k < 3; k++)
          expect(post.occupancy[t]?.[k]).toBeCloseTo(truth.occupancy[t]?.[k] ?? Number.NaN, 11)
      for (let i = 0; i < 3; i++) {
        expect(post.start[i]).toBeCloseTo(truth.start[i] ?? Number.NaN, 11)
        for (let j = 0; j < 3; j++)
          expect(post.transition[i]?.[j]).toBeCloseTo(truth.transition[i]?.[j] ?? Number.NaN, 11)
        for (let u = 0; u < 4; u++) expect(post.duration[i]?.[u]).toBeCloseTo(truth.duration[i]?.[u] ?? Number.NaN, 11)
      }
    }
  })

  it('handles impossible emissions (a symbol a state never emits) exactly', () => {
    const sparse: HsmmParameters = {
      ...tiny,
      emission: [
        [0.7, 0.3, 0],
        [0, 0.8, 0.2],
        [0.2, 0.2, 0.6],
      ],
    }
    const x = [0, 0, 1, 2, 2, 1]
    const post = hsmmPosterior(sparse, x)
    const truth = enumerate(sparse, x)
    expect(post.logLikelihood).toBeCloseTo(truth.logLikelihood, 11)
    for (let t = 0; t < x.length; t++)
      for (let k = 0; k < 3; k++) expect(post.occupancy[t]?.[k]).toBeCloseTo(truth.occupancy[t]?.[k] ?? Number.NaN, 11)
  })

  it('the occupancy sums to one at every step and the counts are consistent', () => {
    const post = hsmmPosterior(tiny, [2, 0, 0, 1, 2, 1, 1, 0])
    for (const row of post.occupancy) expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
    // one more segment than moves; the expected number of segments is the sum of the duration counts
    const moves = post.transition.flat().reduce((a, v) => a + v, 0)
    const segments = post.duration.flat().reduce((a, v) => a + v, 0)
    expect(segments).toBeCloseTo(moves + 1, 11)
  })

  it('with geometric durations it is the hidden Markov model: same likelihood and state posteriors', () => {
    const hmm: HmmParameters = {
      kind: 'categorical',
      start: [0.5, 0.3, 0.2],
      transition: [
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.3, 0.3, 0.4],
      ],
      emission: tiny.kind === 'categorical' ? tiny.emission : [],
    }
    const x = [0, 1, 1, 2, 0, 0, 1, 2, 2, 1]
    const hsmm = hsmmFromHmm(hmm, x.length)
    expect(hsmm.transition.map((r, i) => r[i])).toEqual([0, 0, 0])
    const fb = forwardBackward(hmmChain(hmmFromParameters(hmm), x))
    const post = hsmmPosterior(hsmm, x)
    expect(post.logLikelihood).toBeCloseTo(fb.logLikelihood, 11)
    for (let t = 0; t < x.length; t++)
      for (let k = 0; k < 3; k++) expect(post.occupancy[t]?.[k]).toBeCloseTo(fb.posteriors[t]?.[k] ?? Number.NaN, 11)
    expect(hsmmLogLikelihood(hsmm, [x, x])).toBeCloseTo(2 * fb.logLikelihood, 10)
    expect(geometricDurations([0.5], 3)).toEqual([[0.5, 0.25, 0.25]])
  })

  it('rejects an empty or impossible sequence', () => {
    expect(() => hsmmPosterior(tiny, [])).toThrow(/empty/)
    const left: HsmmParameters = {
      ...tiny,
      emission: [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
      ],
    }
    expect(() => hsmmPosterior(left, [1])).toThrow(/probability zero/)
  })
})

describe('duration families', () => {
  it('the truncated Poisson and negative binomial fits recover the rate and the size from their own pmf', () => {
    for (const lambda of [0.4, 3.2, 7]) expect(fitPoissonDurations(poissonDurations(lambda, 12))).toBeCloseTo(lambda, 7)
    for (const [r, p] of [
      [2.5, 0.3],
      [6, 0.5],
    ] as const) {
      const fit = fitNegativeBinomialDurations(negativeBinomialDurations(r, p, 40), 1)
      expect(fit.r).toBeCloseTo(r, 3)
      expect(fit.p).toBeCloseTo(p, 4)
    }
    // a negative binomial of size one is the geometric distribution
    const geo = negativeBinomialDurations(1, 0.4, 5)
    expect((geo[1] ?? 0) / (geo[0] ?? 1)).toBeCloseTo(0.6, 12)
    expect(poissonDurations(1, 3).reduce((a, v) => a + v, 0)).toBeCloseTo(1, 14)
  })
})

describe('hidden semi-Markov EM', () => {
  const gaussian = (means: number[], lambda: number[]): HsmmParameters => ({
    kind: 'gaussian',
    start: [1 / 3, 1 / 3, 1 / 3],
    transition: [
      [0, 0.7, 0.3],
      [0.4, 0, 0.6],
      [0.5, 0.5, 0],
    ],
    means: means.map((m) => [m]),
    covariances: means.map(() => [[1]]),
    duration: lambda.map((l) => poissonDurations(l, 25)),
  })

  it('raises the log likelihood at every step, for every duration family', () => {
    const truth = gaussian([-4, 0, 4], [9, 3, 6])
    const sequences = [0, 1, 2].map((s) => hsmmSample(truth, 150, key(s)).observations)
    const start = gaussian([-3, 0.5, 3], [5, 5, 5])
    for (const family of ['nonparametric', 'poisson', 'negative-binomial'] as const) {
      const alg = hsmmAlgorithm(sequences, start, { durationFamily: family, covariancePrior: 0, tol: -1 })
      const lls: number[] = []
      for (const { state } of iterate(alg, { key: key(0), maxSteps: 12 }))
        if (state.logLikelihood !== null) lls.push(state.logLikelihood)
      expect(lls.length).toBe(12)
      for (let t = 1; t < lls.length; t++) expect(lls[t] ?? 0).toBeGreaterThanOrEqual((lls[t - 1] ?? 0) - 1e-9)
      expect((lls.at(-1) ?? 0) - (lls[0] ?? 0)).toBeGreaterThan(1)
    }
  })

  it('recovers planted Poisson durations, means and transitions', () => {
    const truth = gaussian([-4, 0, 4], [9, 3, 6])
    const sequences = [0, 1, 2, 3, 4, 5].map((s) => hsmmSample(truth, 600, key(10 + s)).observations)
    const start = gaussian([-3, 0.5, 3], [5, 5, 5])
    const run = runAlgorithm(hsmmAlgorithm(sequences, start, { durationFamily: 'poisson', tol: 1e-5 }), {
      maxSteps: 80,
      key: key(0),
    })
    const fitted = run.state.params
    if (fitted.kind !== 'gaussian') throw new Error('gaussian expected')
    // durations: the shifted-Poisson rate (mean - 1) of each state, tolerance 3 standard errors (~ 600 segments in all)
    const rate = (pmf: number[]): number => pmf.reduce((a, v, u) => a + v * u, 0)
    for (const [k, l] of [9, 3, 6].entries()) expect(Math.abs(rate(fitted.duration[k] ?? []) - l)).toBeLessThan(0.7)
    for (const [k, m] of [-4, 0, 4].entries()) expect(fitted.means[k]?.[0]).toBeCloseTo(m, 1)
    for (const [i, row] of [
      [0, 0.7, 0.3],
      [0.4, 0, 0.6],
      [0.5, 0.5, 0],
    ].entries())
      for (const [j, v] of row.entries()) expect(Math.abs((fitted.transition[i]?.[j] ?? 0) - v)).toBeLessThan(0.15)
    expect(run.state.params.duration.every((r) => Math.abs(r.reduce((a, v) => a + v, 0) - 1) < 1e-9)).toBe(true)
  })

  it('a negative binomial fit beats a Poisson fit on overdispersed durations', () => {
    const spread = (r: number, p: number): number[] => negativeBinomialDurations(r, p, 40)
    const truth: HsmmParameters = {
      ...gaussian([-4, 0, 4], [1, 1, 1]),
      duration: [spread(1.2, 0.12), spread(1.5, 0.25), spread(1.2, 0.2)],
    }
    const sequences = [0, 1, 2, 3].map((s) => hsmmSample(truth, 500, key(30 + s)).observations)
    const start = { ...gaussian([-3, 0.5, 3], [1, 1, 1]), duration: [8, 3, 5].map((l) => poissonDurations(l, 40)) }
    const ll = (family: 'poisson' | 'negative-binomial'): number => {
      const run = runAlgorithm(hsmmAlgorithm(sequences, start, { durationFamily: family, tol: 1e-4 }), {
        maxSteps: 40,
        key: key(0),
      })
      return run.state.logLikelihood ?? Number.NaN
    }
    expect(ll('negative-binomial')).toBeGreaterThan(ll('poisson') + 5)
  })

  it('serialises its state, and the tape shows the likelihood', () => {
    const start = gaussian([-3, 0.5, 3], [5, 5, 5])
    const sequences = [hsmmSample(start, 80, key(1)).observations]
    const alg = hsmmAlgorithm(sequences, start)
    const run = runAlgorithm(alg, { maxSteps: 4, key: key(0) })
    expect(stateFromJson(stateToJson(run.state))).toEqual(run.state)
    expect(run.steps).toBeGreaterThan(0)
  })
})
