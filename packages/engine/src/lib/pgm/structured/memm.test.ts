import { describe, expect, it } from 'vitest'
import { analyticObjective, lbfgs } from '../../optim/index.ts'
import { key, normal } from '../../random/index.ts'
import { asTensor, toFlat, type Value } from '../../tensor/index.ts'
import { forwardBackward } from '../inference/chain.ts'
import {
  type CrfInput,
  crfDecode,
  crfObjective,
  crfParameters,
  type LabelledSequence,
  linearChainCrf,
  withCrfParameters,
} from './crf.ts'
import {
  type MemmStructure,
  memmChain,
  memmDecode,
  memmLogLikelihood,
  memmObjective,
  memmSmoothProblem,
} from './memm.ts'

const randomVector = (seed: number, n: number, scale = 1): number[] =>
  Array.from(toFlat(normal(key(seed), [n])), (v) => scale * v)

const labels = ['A', 'B', 'C']
const attributes = ['f', 'g', 'h']
const x: CrfInput = [{ f: 1, g: 2 }, { h: 1 }, { f: 1, h: 1.5 }, { g: 1 }]

/** Every labelling of n items. */
function labellings(n: number, K: number): number[][] {
  let all: number[][] = [[]]
  for (let t = 0; t < n; t++) all = all.flatMap((p) => Array.from({ length: K }, (_, k) => [...p, k]))
  return all
}

describe('the MEMM is a locally normalised model', () => {
  const crf = withCrfParameters(linearChainCrf(labels, attributes), randomVector(1, 3 * 3 + 9, 0.8))

  it('its labellings have probabilities that sum to one, and its marginals are the enumerated ones', () => {
    const all = labellings(x.length, 3)
    const p = all.map((y) => Math.exp(memmLogLikelihood(crf, x, y)))
    expect(p.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
    const posteriors = forwardBackward(memmChain(crf, x)).posteriors
    for (let t = 0; t < x.length; t++)
      for (let k = 0; k < 3; k++) {
        const exact = all.reduce((a, y, i) => a + (y[t] === k ? (p[i] ?? 0) : 0), 0)
        expect(posteriors[t]?.[k]).toBeCloseTo(exact, 12)
      }
  })

  it('Viterbi finds the most probable labelling', () => {
    const all = labellings(x.length, 3)
    const ll = all.map((y) => memmLogLikelihood(crf, x, y))
    const best = Math.max(...ll)
    const r = memmDecode(crf, x)
    expect(r.logProbability).toBeCloseTo(best, 12)
    expect(memmLogLikelihood(crf, x, r.labels)).toBeCloseTo(best, 12)
  })

  it('differs from the CRF: the same weights give another distribution', () => {
    const chainLoglik = crfObjective(crf, [{ x, y: ['A', 'B', 'C', 'A'] }]).value
    expect(-memmLogLikelihood(crf, x, ['A', 'B', 'C', 'A'])).not.toBeCloseTo(chainLoglik, 3)
  })
})

describe('the MEMM objective', () => {
  const bigramCrf = withCrfParameters(
    linearChainCrf(labels, attributes, { bigramAttributes: ['f'] }),
    randomVector(2, 3 * 3 + 9 + 9, 0.7),
  )
  const edges: CrfInput = {
    unigram: [{ f: 1 }, { g: 1, h: 2 }, { f: 2 }, { h: 1 }],
    bigram: [{}, { f: 1 }, { f: 0.5 }, {}],
  }
  const data: LabelledSequence[] = [
    { x: edges, y: ['A', 'C', 'C', 'B'] },
    { x: [{ g: 1 }, { f: 1 }], y: ['B', 'A'] },
  ]

  it('has the gradient of finite differences, with bigram attributes and an L2 penalty', () => {
    const f = (w: ArrayLike<number>): { value: number; gradient: Float64Array } =>
      memmObjective(withCrfParameters(bigramCrf, w), data, { c2: 0.1 })
    const w = Array.from(crfParameters(bigramCrf))
    const g = f(w).gradient
    for (let q = 0; q < w.length; q++) {
      const h = 1e-6
      const up = [...w]
      const down = [...w]
      up[q] = (up[q] ?? 0) + h
      down[q] = (down[q] ?? 0) - h
      expect((f(up).value - f(down).value) / (2 * h)).toBeCloseTo(g[q] ?? Number.NaN, 6)
    }
  })

  it('equals the negative log-likelihood of the data', () => {
    const r = memmObjective(bigramCrf, data)
    const nll = data.reduce((a, d) => a - memmLogLikelihood(bigramCrf, d.x, d.y), 0)
    expect(r.value).toBeCloseTo(nll, 12)
  })

  it('is a smooth problem for the solver slot: batches are unbiased and L-BFGS fits the data', () => {
    const problem = memmSmoothProblem({ crf: bigramCrf, data, c2: 0.05 })
    const f = problem.objective as (x: unknown, batch?: readonly number[]) => Value
    const w = randomVector(9, crfParameters(bigramCrf).length, 0.3)
    const value = (b?: readonly number[]): number => toFlat(f(asTensor(Float64Array.from(w)), b))[0] ?? Number.NaN
    const direct = memmObjective(withCrfParameters(bigramCrf, w), data, { c2: 0.05 }).value
    expect(value()).toBeCloseTo(direct, 10)
    expect(problem.batches?.n).toBe(2)
    // the likelihood part averages out over the singleton batches; the penalty is not rescaled
    const penalty = 0.05 * w.reduce((a, v) => a + v * v, 0)
    const mean = (value([0]) + value([1])) / 2
    expect(mean).toBeCloseTo(direct, 8)
    expect(penalty).toBeGreaterThan(0)
    const r = lbfgs(problem.objective as never, problem.x0 as never, { gradTol: 1e-8, maxIterations: 200 })
    const fitted = withCrfParameters(bigramCrf, toFlat(r.x as Value))
    expect(memmDecode(fitted, edges).labels).toEqual(['A', 'C', 'C', 'B'])
  })
})

describe('the label-bias problem (Lafferty, McCallum and Pereira 2001)', () => {
  // States 1..5 over the words "rib" and "rob": "rib" is labelled 1 2 3 and "rob" is 4 5 3. After the shared first letter
  // the states 1 and 4 each have a single successor, and so do 2 and 5.
  const names = ['1', '2', '3', '4', '5']
  const vocabulary = ['r', 'i', 'o', 'b']
  const rib: CrfInput = [{ r: 1 }, { i: 1 }, { b: 1 }]
  const rob: CrfInput = [{ r: 1 }, { o: 1 }, { b: 1 }]
  const next = (...to: number[]): boolean[] => names.map((_, k) => to.includes(k))
  const structure: MemmStructure = {
    start: next(0, 3),
    transition: [next(1), next(2), next(), next(4), next(2)],
  }
  const data: LabelledSequence[] = [
    { x: rib, y: ['1', '2', '3'] },
    { x: rob, y: ['4', '5', '3'] },
  ]
  const start = linearChainCrf(names, vocabulary)
  const fit = (
    objective: (w: Float64Array) => { value: number; gradient: Float64Array },
  ): ReturnType<typeof withCrfParameters> => {
    const r = lbfgs(
      analyticObjective({
        value: (w) => objective(Float64Array.from(w)).value,
        gradient: (w) => objective(Float64Array.from(w)).gradient,
      }),
      crfParameters(start),
      { gradTol: 1e-8, maxIterations: 300 },
    )
    return withCrfParameters(start, toFlat(r.x as Value))
  }

  it('the MEMM cannot tell the paths apart: after the first letter the observations do not matter, for any weights', () => {
    for (const seed of [1, 2, 3]) {
      const crf = withCrfParameters(start, randomVector(seed, crfParameters(start).length, 2))
      for (const path of [
        ['1', '2', '3'],
        ['4', '5', '3'],
      ]) {
        // the probability of a path is the same whichever second letter is observed
        expect(memmLogLikelihood(crf, rib, path, structure)).toBeCloseTo(
          memmLogLikelihood(crf, rob, path, structure),
          12,
        )
      }
      // so "rib" and "rob" are decoded to the same path, and one of the two is wrong
      const a = memmDecode(crf, rib, structure).labels
      const b = memmDecode(crf, rob, structure).labels
      expect(a).toEqual(b)
      expect(a.join('') === '123' && b.join('') === '453').toBe(false)
    }
  })

  it('trained on both words, the MEMM still gives the wrong path as much probability as the right one', () => {
    const crf = fit((w) => memmObjective(withCrfParameters(start, w), data, { c2: 0.01, structure }))
    // the first label decides everything and "r" does not tell 1 from 4: each path has probability 1/2, on both inputs
    for (const input of [rib, rob])
      for (const path of [
        ['1', '2', '3'],
        ['4', '5', '3'],
      ])
        expect(Math.exp(memmLogLikelihood(crf, input, path, structure))).toBeCloseTo(0.5, 6)
    expect(memmDecode(crf, rib, structure).labels).toEqual(memmDecode(crf, rob, structure).labels)
  })

  it('the CRF, trained on the same data with the same features, labels both words correctly', () => {
    const crf = fit((w) => crfObjective(withCrfParameters(start, w), data, { c2: 0.01 }))
    expect(crfDecode(crf, rib).labels).toEqual(['1', '2', '3'])
    expect(crfDecode(crf, rob).labels).toEqual(['4', '5', '3'])
  })
})
