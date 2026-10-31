import { describe, expect, it } from 'vitest'
import { foldIn, key, normal, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { iterate } from '../../trace/index.ts'
import { baumWelchAlgorithm, type HmmParameters, initialHmmParameters } from './baum-welch.ts'
import { type GmmHmmParameters, gmmHmmAlgorithm, gmmHmmLogLikelihood, initialGmmHmmParameters } from './gmm-hmm.ts'

const lse = (xs: number[]): number => Math.log(xs.reduce((a, v) => a + Math.exp(v), 0))

/** log N(x; mean, cov) in two dimensions by the explicit formula. */
function logN2(x: number[], mean: number[], c: number[][]): number {
  const det = (c[0]?.[0] ?? 0) * (c[1]?.[1] ?? 0) - (c[0]?.[1] ?? 0) * (c[1]?.[0] ?? 0)
  const dx = (x[0] ?? 0) - (mean[0] ?? 0)
  const dy = (x[1] ?? 0) - (mean[1] ?? 0)
  const q = ((c[1]?.[1] ?? 0) * dx * dx - 2 * (c[0]?.[1] ?? 0) * dx * dy + (c[0]?.[0] ?? 0) * dy * dy) / det
  return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * q
}

/** A draw from a Gaussian-mixture HMM (diagonal components). */
function sample(p: GmmHmmParameters, T: number, seed: number): number[][] {
  const u = Array.from(toFlat(uniform(key(seed), [3 * T])))
  const z = Array.from(toFlat(normal(foldIn(key(seed), 1), [T, p.means[0]?.[0]?.length ?? 1])))
  let next = 0
  const draw = (w: readonly number[]): number => {
    const r = u[next++] ?? 0
    let acc = 0
    for (let i = 0; i < w.length; i++) {
      acc += w[i] ?? 0
      if (r < acc) return i
    }
    return w.length - 1
  }
  const out: number[][] = []
  let state = draw(p.start)
  const d = p.means[0]?.[0]?.length ?? 1
  for (let t = 0; t < T; t++) {
    if (t > 0) state = draw(p.transition[state] ?? [])
    const m = draw(p.weights[state] ?? [])
    out.push(
      (p.means[state]?.[m] ?? []).map(
        (mu, a) => mu + Math.sqrt(p.covariances[state]?.[m]?.[a]?.[a] ?? 1) * (z[t * d + a] ?? 0),
      ),
    )
  }
  return out
}

describe('Gaussian-mixture HMM likelihood', () => {
  it('matches the explicit sum over state paths and mixture components (two dimensions, full covariances)', () => {
    const p: GmmHmmParameters = {
      start: [0.6, 0.4],
      transition: [
        [0.7, 0.3],
        [0.2, 0.8],
      ],
      weights: [
        [0.3, 0.7],
        [0.5, 0.5],
      ],
      means: [
        [
          [0, 0],
          [2, 1],
        ],
        [
          [-1, 2],
          [3, -1],
        ],
      ],
      covariances: [
        [
          [
            [1, 0.3],
            [0.3, 0.8],
          ],
          [
            [0.5, -0.1],
            [-0.1, 1.2],
          ],
        ],
        [
          [
            [0.7, 0],
            [0, 0.7],
          ],
          [
            [1, 0.5],
            [0.5, 1],
          ],
        ],
      ],
    }
    const x = [
      [0.1, 0.2],
      [1.5, 1],
      [-0.5, 1.8],
    ]
    // brute force over the 2^3 state paths, each emission the explicit mixture sum
    const emit = (k: number, o: number[]): number =>
      lse(
        [0, 1].map(
          (m) => Math.log(p.weights[k]?.[m] ?? 0) + logN2(o, p.means[k]?.[m] ?? [], p.covariances[k]?.[m] ?? []),
        ),
      )
    let total = 0
    for (let code = 0; code < 8; code++) {
      const z = [code & 1, (code >> 1) & 1, (code >> 2) & 1]
      let lp = Math.log(p.start[z[0] ?? 0] ?? 0) + emit(z[0] ?? 0, x[0] ?? [])
      for (let t = 1; t < 3; t++)
        lp += Math.log(p.transition[z[t - 1] ?? 0]?.[z[t] ?? 0] ?? 0) + emit(z[t] ?? 0, x[t] ?? [])
      total += Math.exp(lp)
    }
    expect(gmmHmmLogLikelihood(p, [x])).toBeCloseTo(Math.log(total), 12)
  })
})

describe('Gaussian-mixture HMM EM', () => {
  it('with one component per state it is the Gaussian Baum-Welch: the same log likelihoods, step by step', () => {
    const truth: GmmHmmParameters = {
      start: [0.5, 0.5],
      transition: [
        [0.9, 0.1],
        [0.2, 0.8],
      ],
      weights: [[1], [1]],
      means: [[[0, 0]], [[3, 2]]],
      covariances: [
        [
          [
            [1, 0],
            [0, 1],
          ],
        ],
        [
          [
            [0.5, 0],
            [0, 2],
          ],
        ],
      ],
    }
    const x = sample(truth, 150, 1)
    const bwStart = initialHmmParameters([x], { states: 2, emission: 'gaussian' }, key(0)) as Extract<
      HmmParameters,
      { kind: 'gaussian' }
    >
    const start: GmmHmmParameters = {
      start: bwStart.start,
      transition: bwStart.transition,
      weights: [[1], [1]],
      means: bwStart.means.map((m) => [m]),
      covariances: bwStart.covariances.map((c) => [c]),
    }
    const collect = (alg: { init: (k: ReturnType<typeof key>) => unknown }): number[] => {
      const out: number[] = []
      for (const { state } of iterate(alg as never, { key: key(0), maxSteps: 6 })) {
        const ll = (state as { logLikelihood: number | null }).logLikelihood
        if (ll !== null) out.push(ll)
      }
      return out
    }
    const a = collect(baumWelchAlgorithm([x], bwStart, { tol: -1 }))
    const b = collect(gmmHmmAlgorithm([x], start, { tol: -1, minCovar: 0 }))
    expect(b.length).toBe(a.length)
    for (const [i, v] of b.entries()) expect(v).toBeCloseTo(a[i] ?? Number.NaN, 8)
  })

  const truth: GmmHmmParameters = {
    start: [0.5, 0.5],
    transition: [
      [0.9, 0.1],
      [0.1, 0.9],
    ],
    // state 0 is bimodal (at -4 and +4), state 1 sits in between at 0
    weights: [
      [0.5, 0.5],
      [1, 0],
    ],
    means: [
      [[-4], [4]],
      [[0], [10]],
    ],
    covariances: [
      [[[0.25]], [[0.25]]],
      [[[0.25]], [[1]]],
    ],
  }
  const sequences = [sample(truth, 400, 11), sample(truth, 400, 12)]
  const start: GmmHmmParameters = {
    start: [0.5, 0.5],
    transition: [
      [0.7, 0.3],
      [0.3, 0.7],
    ],
    weights: [
      [0.5, 0.5],
      [0.5, 0.5],
    ],
    means: [
      [[-3], [3]],
      [[-1], [1]],
    ],
    covariances: [
      [[[1]], [[1]]],
      [[[1]], [[1]]],
    ],
  }

  it('raises the log likelihood at every step (zero covariance prior)', () => {
    const alg = gmmHmmAlgorithm(sequences, start, { covariancePrior: 0, tol: -1 })
    const lls: number[] = []
    for (const { state } of iterate(alg, { key: key(0), maxSteps: 15 }))
      if (state.logLikelihood !== null) lls.push(state.logLikelihood)
    expect(lls.length).toBe(15)
    for (let t = 1; t < lls.length; t++) expect(lls[t] ?? 0).toBeGreaterThanOrEqual((lls[t - 1] ?? 0) - 1e-9)
    expect((lls.at(-1) ?? 0) - (lls[0] ?? 0)).toBeGreaterThan(50)
  })

  it('recovers a planted bimodal state: the two modes, their weights and the transitions', () => {
    const alg = gmmHmmAlgorithm(sequences, start, { covariancePrior: 0, tol: 1e-6 })
    let last = start
    for (const { state } of iterate(alg, { key: key(0), maxSteps: 100 })) last = state.params
    const modes = (last.means[0] ?? []).map((m) => m[0] ?? 0).sort((a, b) => a - b)
    expect(modes[0]).toBeCloseTo(-4, 0)
    expect(modes[1]).toBeCloseTo(4, 0)
    // each mode carries about half of state 0
    expect(Math.abs((last.weights[0]?.[0] ?? 0) - 0.5)).toBeLessThan(0.1)
    expect(Math.abs((last.transition[0]?.[0] ?? 0) - 0.9)).toBeLessThan(0.05)
    expect(Math.abs((last.transition[1]?.[1] ?? 0) - 0.9)).toBeLessThan(0.05)
    // one Gaussian per state cannot explain the bimodal state: the mixture gains at least a hundred nats
    const single = gmmHmmAlgorithm(
      sequences,
      { ...start, weights: [[1], [1]], means: [[[-3]], [[0]]], covariances: [[[[1]]], [[[1]]]] },
      { covariancePrior: 0, tol: 1e-6 },
    )
    let ll1 = Number.NEGATIVE_INFINITY
    for (const { state } of iterate(single, { key: key(0), maxSteps: 100 })) ll1 = state.logLikelihood ?? ll1
    expect(gmmHmmLogLikelihood(last, sequences)).toBeGreaterThan(ll1 + 100)
  })

  it('initial parameters are valid, ordered and deterministic from the key', () => {
    const a = initialGmmHmmParameters(sequences, { states: 2, mixtures: 3 }, key(0))
    expect(a).toEqual(initialGmmHmmParameters(sequences, { states: 2, mixtures: 3 }, key(0)))
    expect(a.weights[0]).toEqual([1 / 3, 1 / 3, 1 / 3])
    expect(a.means[0]?.[0]?.[0] ?? 0).toBeLessThan(a.means[1]?.[0]?.[0] ?? 0)
    expect(() => initialGmmHmmParameters([], { states: 2, mixtures: 1 }, key(0))).toThrow()
    expect(Number.isFinite(gmmHmmLogLikelihood(a, sequences))).toBe(true)
  })
})
