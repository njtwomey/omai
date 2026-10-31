import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { iterate } from '../../trace/index.ts'
import {
  type FactorialHmmParameters,
  factorialHmmAlgorithm,
  factorialHmmExact,
  factorialHmmSample,
  factorialHmmVariational,
} from './factorial-hmm.ts'

// Two chains (2 and 3 states), two-dimensional observations.
const small: FactorialHmmParameters = {
  start: [
    [0.6, 0.4],
    [0.2, 0.5, 0.3],
  ],
  transition: [
    [
      [0.8, 0.2],
      [0.3, 0.7],
    ],
    [
      [0.6, 0.3, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.3, 0.6],
    ],
  ],
  loadings: [
    [
      [0, 0],
      [2, 1],
    ],
    [
      [0, 0],
      [1, -1],
      [-1, 2],
    ],
  ],
  bias: [0.5, -0.5],
  variances: [0.8, 1.2],
}

/** Every joint path of the two chains, with its probability: the independent truth for the exact posterior. */
function enumerate(p: FactorialHmmParameters, x: number[][]): { total: number; gamma: number[][][] } {
  const T = x.length
  const gamma = p.start.map((s) => Array.from({ length: T }, () => new Array<number>(s.length).fill(0)))
  let total = 0
  const paths: [number[], number[]][] = []
  const rec = (t: number, a: number[], b: number[]): void => {
    if (t === T) {
      paths.push([a, b])
      return
    }
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) rec(t + 1, [...a, i], [...b, j])
  }
  rec(0, [], [])
  for (const [a, b] of paths) {
    let pr = (p.start[0]?.[a[0] ?? 0] ?? 0) * (p.start[1]?.[b[0] ?? 0] ?? 0)
    for (let t = 1; t < T; t++)
      pr *= (p.transition[0]?.[a[t - 1] ?? 0]?.[a[t] ?? 0] ?? 0) * (p.transition[1]?.[b[t - 1] ?? 0]?.[b[t] ?? 0] ?? 0)
    for (let t = 0; t < T; t++)
      for (let d = 0; d < 2; d++) {
        const mean = (p.bias[d] ?? 0) + (p.loadings[0]?.[a[t] ?? 0]?.[d] ?? 0) + (p.loadings[1]?.[b[t] ?? 0]?.[d] ?? 0)
        const v = p.variances[d] ?? 1
        pr *= Math.exp(-(((x[t]?.[d] ?? 0) - mean) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)
      }
    total += pr
    for (let t = 0; t < T; t++) {
      ;(gamma[0]?.[t] as number[])[a[t] ?? 0] = (gamma[0]?.[t]?.[a[t] ?? 0] ?? 0) + pr
      ;(gamma[1]?.[t] as number[])[b[t] ?? 0] = (gamma[1]?.[t]?.[b[t] ?? 0] ?? 0) + pr
    }
  }
  return { total, gamma: gamma.map((c) => c.map((r) => r.map((v) => v / total))) }
}

describe('factorial HMM exact inference', () => {
  it('matches the sum over all joint paths: log likelihood and the chain marginals', () => {
    const x = [
      [0.4, -0.2],
      [2.1, 0.3],
      [0.7, 1.4],
      [-0.5, 1.9],
    ]
    const truth = enumerate(small, x)
    const exact = factorialHmmExact(small, x)
    expect(exact.objective).toBeCloseTo(Math.log(truth.total), 11)
    for (let m = 0; m < 2; m++)
      for (let t = 0; t < x.length; t++)
        for (let k = 0; k < (small.start[m]?.length ?? 0); k++)
          expect(exact.gamma[m]?.[t]?.[k]).toBeCloseTo(truth.gamma[m]?.[t]?.[k] ?? Number.NaN, 11)
    // the summed joint occupancy of the two chains has the total mass T, and its margins are the chains' occupancies
    const cross = exact.cross[0]?.[1] ?? []
    expect(cross.flat().reduce((a, v) => a + v, 0)).toBeCloseTo(x.length, 11)
    for (let k = 0; k < 2; k++)
      expect((cross[k] ?? []).reduce((a, v) => a + v, 0)).toBeCloseTo(
        exact.gamma[0]?.reduce((s, r) => s + (r[k] ?? 0), 0) ?? Number.NaN,
        11,
      )
  })
})

describe('structured variational inference', () => {
  const x = [
    [0.4, -0.2],
    [2.1, 0.3],
    [0.7, 1.4],
    [-0.5, 1.9],
    [2.2, 0.1],
  ]

  it('with one chain the bound is the log likelihood and the marginals are exact', () => {
    const one: FactorialHmmParameters = {
      start: [small.start[1] ?? []],
      transition: [small.transition[1] ?? []],
      loadings: [small.loadings[1] ?? []],
      bias: small.bias,
      variances: small.variances,
    }
    const v = factorialHmmVariational(one, x, { key: key(0) })
    const e = factorialHmmExact(one, x)
    expect(v.objective).toBeCloseTo(e.objective, 9)
    for (let t = 0; t < x.length; t++)
      for (let k = 0; k < 3; k++) expect(v.gamma[0]?.[t]?.[k]).toBeCloseTo(e.gamma[0]?.[t]?.[k] ?? Number.NaN, 8)
  })

  it('the bound lies below the log likelihood and rises with every sweep', () => {
    const v = factorialHmmVariational(small, x, { key: key(1), tol: 0, maxSweeps: 30 })
    const e = factorialHmmExact(small, x)
    expect(v.objective).toBeLessThanOrEqual(e.objective + 1e-9)
    expect(v.history.length).toBeGreaterThan(3) // it stops once a sweep gains nothing
    for (let s = 1; s < v.history.length; s++)
      expect(v.history[s] ?? 0).toBeGreaterThanOrEqual((v.history[s - 1] ?? 0) - 1e-10)
    expect(v.objective).toBeGreaterThan(e.objective - 3)
  })

  it('on well-separated, low-noise data the factorised marginals are close to the exact ones', () => {
    const sharp: FactorialHmmParameters = {
      ...small,
      variances: [0.05, 0.05],
      loadings: [
        [
          [0, 0],
          [4, 0],
        ],
        [
          [0, 0],
          [0, 4],
          [-4, -4],
        ],
      ],
    }
    const data = factorialHmmSample(sharp, 12, key(3)).observations
    const v = factorialHmmVariational(sharp, data, { key: key(2), maxSweeps: 40 })
    const e = factorialHmmExact(sharp, data)
    let worst = 0
    for (let m = 0; m < 2; m++)
      for (let t = 0; t < data.length; t++)
        for (let k = 0; k < (sharp.start[m]?.length ?? 0); k++)
          worst = Math.max(worst, Math.abs((v.gamma[m]?.[t]?.[k] ?? 0) - (e.gamma[m]?.[t]?.[k] ?? 0)))
    expect(worst).toBeLessThan(0.05)
    expect(e.objective - v.objective).toBeLessThan(1)
  })
})

describe('factorial HMM EM', () => {
  const truth: FactorialHmmParameters = {
    start: [
      [0.5, 0.5],
      [0.5, 0.5],
    ],
    transition: [
      [
        [0.9, 0.1],
        [0.1, 0.9],
      ],
      [
        [0.8, 0.2],
        [0.2, 0.8],
      ],
    ],
    loadings: [
      [[0], [3]],
      [[0], [1.2]],
    ],
    bias: [0.5],
    variances: [0.1],
  }
  const sequences = [0, 1, 2].map((s) => factorialHmmSample(truth, 150, key(40 + s)).observations)
  const start: FactorialHmmParameters = {
    start: [
      [0.5, 0.5],
      [0.5, 0.5],
    ],
    transition: [
      [
        [0.7, 0.3],
        [0.3, 0.7],
      ],
      [
        [0.7, 0.3],
        [0.3, 0.7],
      ],
    ],
    loadings: [
      [[0], [2]],
      [[0], [1.8]],
    ],
    bias: [0.3],
    variances: [1],
  }
  const meansOf = (p: FactorialHmmParameters): number[] =>
    [0, 1]
      .flatMap((a) =>
        [0, 1].map((b) => (p.bias[0] ?? 0) + (p.loadings[0]?.[a]?.[0] ?? 0) + (p.loadings[1]?.[b]?.[0] ?? 0)),
      )
      .sort((u, v) => u - v)

  it('exact EM raises the log likelihood at every step and recovers the planted joint means and chains', () => {
    const lls: number[] = []
    let last = start
    for (const { state } of iterate(factorialHmmAlgorithm(sequences, start, { tol: -1 }), {
      key: key(0),
      maxSteps: 40,
    })) {
      if (state.objective !== null) lls.push(state.objective)
      last = state.params
    }
    for (let t = 1; t < lls.length; t++) expect(lls[t] ?? 0).toBeGreaterThanOrEqual((lls[t - 1] ?? 0) - 1e-8)
    expect((lls.at(-1) ?? 0) - (lls[0] ?? 0)).toBeGreaterThan(20)
    // the four joint means (bias plus both loadings): planted 0.5, 1.7, 3.5, 4.7 (the sums are identified, the split is not)
    for (const [i, m] of meansOf(last).entries()) expect(Math.abs(m - (meansOf(truth)[i] ?? 0))).toBeLessThan(0.15)
    expect(Math.abs((last.variances[0] ?? 0) - 0.1)).toBeLessThan(0.05)
    const stays = [
      last.transition[0]?.[0]?.[0],
      last.transition[0]?.[1]?.[1],
      last.transition[1]?.[0]?.[0],
      last.transition[1]?.[1]?.[1],
    ].map((v) => v ?? 0)
    // a chain may be labelled either way round, and the chains may swap: the stay probabilities are a set {0.9, 0.9, 0.8, 0.8}
    expect(Math.abs(stays.reduce((a, v) => a + v, 0) - 3.4)).toBeLessThan(0.25)
  })

  it('variational EM raises the bound at every step and finds the same joint means', () => {
    const bounds: number[] = []
    let last = start
    for (const { state } of iterate(
      factorialHmmAlgorithm(sequences, start, { inference: 'variational', sweeps: 15, tol: -1 }, key(5)),
      {
        key: key(0),
        maxSteps: 30,
      },
    )) {
      if (state.objective !== null) bounds.push(state.objective)
      last = state.params
    }
    for (let t = 1; t < bounds.length; t++) expect(bounds[t] ?? 0).toBeGreaterThanOrEqual((bounds[t - 1] ?? 0) - 1e-8)
    expect((bounds.at(-1) ?? 0) - (bounds[0] ?? 0)).toBeGreaterThan(20)
    for (const [i, m] of meansOf(last).entries()) expect(Math.abs(m - (meansOf(truth)[i] ?? 0))).toBeLessThan(0.3)
    // the bound of the fitted model stays below its exact log likelihood
    const exact = sequences.reduce((a, s) => a + factorialHmmExact(last, s).objective, 0)
    expect(bounds.at(-1) ?? 0).toBeLessThanOrEqual(exact + 1e-6)
  })

  it('samples are deterministic from the key and the product chain has a size limit', () => {
    expect(factorialHmmSample(truth, 8, key(1))).toEqual(factorialHmmSample(truth, 8, key(1)))
    const wide: FactorialHmmParameters = {
      start: Array.from({ length: 7 }, () => [0.25, 0.25, 0.25, 0.25]),
      transition: Array.from({ length: 7 }, () => Array.from({ length: 4 }, () => [0.25, 0.25, 0.25, 0.25])),
      loadings: Array.from({ length: 7 }, () => [[0], [1], [2], [3]]),
      bias: [0],
      variances: [1],
    }
    expect(() => factorialHmmExact(wide, [[0], [1]])).toThrow(/too many/)
  })
})
