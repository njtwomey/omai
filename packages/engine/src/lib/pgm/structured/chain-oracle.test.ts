import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { asTensor, toFlat, type Value } from '../../tensor/index.ts'
import { type ChainInput, type ChainLabels, chainOracle } from './chain-oracle.ts'
import { chainVocabulary, compileChainData, compileChainInput, isCompiledChainInput } from './compiled.ts'
import {
  type CrfInput,
  type CrfObservation,
  crfDecode,
  crfExpectedFeatureCounts,
  crfFeatureCounts,
  crfObjective,
  crfScore,
  linearChainCrf,
  withCrfParameters,
} from './crf.ts'
import { structuredObjective, structuredSmoothProblem } from './objective.ts'
import type { StructuredOracle } from './oracle.ts'

const labels = ['A', 'B', 'C']
const attributes = ['f', 'g', 'h']
const bigramAttributes = ['e1', 'e2']
const base = linearChainCrf(labels, attributes, { bigramAttributes })
const P = base.stateWeights.length + base.transitionWeights.length + base.bigramWeights.length
const attributeNames = [...attributes, 'unseen']
const edgeNames = [...bigramAttributes, 'unseen-edge']

/** A sequence input from a table of values (0 means the attribute is absent), with edge attributes. */
function observation(items: readonly (readonly number[])[], edges: readonly (readonly number[])[]): CrfObservation {
  const record = (names: readonly string[], row: readonly number[]): Record<string, number> => {
    const out: Record<string, number> = {}
    for (const [i, v] of row.entries()) if (v !== 0) out[names[i] ?? ''] = v
    return out
  }
  return {
    unigram: items.map((r) => record(attributeNames, r)),
    bigram: edges.map((r) => record(edgeNames, r)),
  }
}

const value = fc.oneof(fc.constant(0), fc.double({ min: -2, max: 2, noNaN: true }))
const sequence = fc
  .integer({ min: 1, max: 4 })
  .chain((T) =>
    fc.tuple(
      fc.array(fc.array(value, { minLength: 4, maxLength: 4 }), { minLength: T, maxLength: T }),
      fc.array(fc.array(value, { minLength: 3, maxLength: 3 }), { minLength: T, maxLength: T }),
      fc.array(fc.integer({ min: 0, max: 2 }), { minLength: T, maxLength: T }),
    ),
  )
  .map(([items, edges, y]) => ({ x: observation(items, edges), y }))
const weights = fc.array(fc.double({ min: -1.5, max: 1.5, noNaN: true }), { minLength: P, maxLength: P })

/** Every labelling of a sequence of length T. */
function allLabellings(T: number): number[][] {
  let all: number[][] = [[]]
  for (let t = 0; t < T; t++) all = all.flatMap((p) => labels.map((_, k) => [...p, k]))
  return all
}

const hamming = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((n, k, t) => n + (k === b[t] ? 0 : 1), 0)

describe('the chain oracle and its compiled form', () => {
  const oracle = chainOracle(base)

  it('the layout is the parameter order of the model', () => {
    expect(oracle.size).toBe(P)
  })

  it('features, expectations and decoding agree between plain and compiled inputs, and with crf.ts', () => {
    fc.assert(
      fc.property(sequence, weights, ({ x, y }, w) => {
        const [c] = compileChainData(base, [{ x, y }])
        if (c === undefined) throw new Error('no compiled sequence')
        expect(isCompiledChainInput(c.x)).toBe(true)
        const model = withCrfParameters(base, w)
        // features
        const phi = crfFeatureCounts(base, x, y)
        expect(Array.from(oracle.features(x, y))).toEqual(Array.from(phi))
        expect(Array.from(oracle.features(c.x, c.y))).toEqual(Array.from(phi))
        // expectations
        const want = crfExpectedFeatureCounts(model, x)
        const got = oracle.expectations(w, c.x)
        expect(got.logPartition).toBeCloseTo(want.logPartition, 12)
        for (const [q, v] of got.counts.entries()) expect(v).toBeCloseTo(want.counts[q] ?? Number.NaN, 12)
        // decoding: the labels come back as ids for compiled input and as names for plain input
        const plain = oracle.decode(w, x)
        const fast = oracle.decode(w, c.x)
        expect(plain.y).toEqual(crfDecode(model, x).labels)
        expect(fast.y.map((k) => labels[k as number])).toEqual(plain.y)
        expect(fast.score).toBeCloseTo(plain.score, 12)
        // gradientInto: the negative log-likelihood and its gradient, plain and compiled
        const direct = crfObjective(model, [{ x, y }])
        for (const input of [x, c.x]) {
          const g = new Float64Array(P)
          const nll = oracle.gradientInto?.(w, input, input === x ? y : c.y, g) ?? Number.NaN
          expect(nll).toBeCloseTo(direct.value, 10)
          for (const [q, v] of g.entries()) expect(v).toBeCloseTo(direct.gradient[q] ?? Number.NaN, 10)
        }
      }),
      { numRuns: 60 },
    )
  })

  it('the expected counts and log-partition function equal the enumeration of every labelling', () => {
    fc.assert(
      fc.property(sequence, weights, ({ x }, w) => {
        const model = withCrfParameters(base, w)
        const T = x.unigram.length
        const all = allLabellings(T)
        const scores = all.map((y) => crfScore(model, x, y))
        const m = Math.max(...scores)
        const logZ = m + Math.log(scores.reduce((a, s) => a + Math.exp(s - m), 0))
        const expected = new Float64Array(P)
        all.forEach((y, i) => {
          const p = Math.exp((scores[i] ?? 0) - logZ)
          for (const [q, v] of oracle.features(x, y).entries()) expected[q] = (expected[q] ?? 0) + p * v
        })
        const [c] = compileChainData(base, [{ x, y: all[0] ?? [] }])
        const got = oracle.expectations(w, c?.x ?? x)
        expect(got.logPartition).toBeCloseTo(logZ, 10)
        for (const [q, v] of got.counts.entries()) expect(v).toBeCloseTo(expected[q] ?? Number.NaN, 10)
      }),
      { numRuns: 40 },
    )
  })

  it('decoding returns the best labelling, and against a gold labelling the best by score plus Hamming cost', () => {
    fc.assert(
      fc.property(sequence, weights, ({ x, y }, w) => {
        const model = withCrfParameters(base, w)
        const all = allLabellings(y.length)
        const score = (l: readonly number[]): number => crfScore(model, x, l)
        const best = Math.max(...all.map(score))
        const plain = oracle.decode(w, x)
        expect(plain.score).toBeCloseTo(best, 10)
        expect(score(plain.y.map((l) => labels.indexOf(l as string)))).toBeCloseTo(best, 10)
        const [c] = compileChainData(base, [{ x, y }])
        for (const input of [x, c?.x ?? x]) {
          const augmented = oracle.decode(w, input, { against: y })
          const bestAugmented = Math.max(...all.map((l) => score(l) + hamming(l, y)))
          expect(augmented.score).toBeCloseTo(bestAugmented, 10)
          const picked = augmented.y.map((l) => (typeof l === 'number' ? l : labels.indexOf(l)))
          expect(score(picked) + hamming(picked, y)).toBeCloseTo(bestAugmented, 10)
        }
        // the loss is the Hamming distance, on names or ids
        expect(oracle.loss(y, plain.y)).toBe(
          hamming(
            y,
            plain.y.map((l) => labels.indexOf(l as string)),
          ),
        )
        expect(
          oracle.loss(
            y.map((k) => labels[k] ?? ''),
            y,
          ),
        ).toBe(0)
      }),
      { numRuns: 40 },
    )
  })

  it('an empty sequence has no features, a zero log-partition function and an empty decoding', () => {
    const empty: CrfInput = []
    expect(Array.from(oracle.features(empty, []))).toEqual(new Array(P).fill(0))
    expect(oracle.expectations(new Float64Array(P), empty).logPartition).toBe(0)
    expect(oracle.decode(new Float64Array(P), empty)).toEqual({ y: [], score: 0 })
    const [c] = compileChainData(base, [{ x: empty, y: [] }])
    expect(oracle.decode(new Float64Array(P), c?.x ?? empty)).toEqual({ y: [], score: 0 })
  })

  it('rejects a compiled input from another vocabulary and labels of the wrong length', () => {
    const other = linearChainCrf(labels, ['f'])
    const [c] = compileChainData(other, [{ x: [{ f: 1 }], y: ['A'] }])
    expect(() => oracle.decode(new Float64Array(P), c?.x ?? [])).toThrow(/vocabulary/)
    expect(() => oracle.features([{ f: 1 }], ['A', 'B'])).toThrow(/one label per item/)
    const compiled = compileChainInput(chainVocabulary(base), [{ f: 1 }])
    expect(() => oracle.features(compiled, ['A', 'B'])).toThrow(/one label per item/)
    expect(() => oracle.loss(['A'], ['Z'])).toThrow(/unknown label/)
  })
})

describe('structuredObjective', () => {
  const oracle = chainOracle(base)
  const data = [
    {
      x: observation(
        [
          [1, 0, 2, 0],
          [0, 1, 0, 0],
          [1, 1, 0, 1],
        ],
        [
          [0, 0, 0],
          [1, 0, 0],
          [0, 2, 1],
        ],
      ),
      y: ['A', 'C', 'B'],
    },
    {
      x: observation(
        [
          [0, 1, 1, 0],
          [2, 0, 0, 0],
        ],
        [
          [0, 0, 0],
          [0, 1, 0],
        ],
      ),
      y: ['B', 'B'],
    },
    { x: observation([[1, 0, 0, 0]], [[0, 0, 0]]), y: ['C'] },
  ]
  const w = Array.from({ length: P }, (_, q) => 0.4 * Math.sin(1 + 1.7 * q))
  const compiled = compileChainData(base, data)

  it('equals crfObjective in value and gradient, on plain and compiled data, with both penalties', () => {
    const penalty = { c2: 0.07, c1: 0.03 }
    const want = crfObjective(withCrfParameters(base, w), data, penalty)
    for (const rows of [data, compiled]) {
      const got = structuredObjective(oracle, rows, w, penalty)
      expect(got.value).toBeCloseTo(want.value, 10)
      for (const [q, v] of got.gradient.entries()) expect(v).toBeCloseTo(want.gradient[q] ?? Number.NaN, 10)
    }
  })

  it('has a gradient that matches central finite differences of the value (L2 penalty)', () => {
    const penalty = { c2: 0.1 }
    const g = structuredObjective(oracle, compiled, w, penalty).gradient
    for (let q = 0; q < P; q += 3) {
      const h = 1e-6
      const up = Float64Array.from(w)
      const dn = Float64Array.from(w)
      up[q] = (up[q] ?? 0) + h
      dn[q] = (dn[q] ?? 0) - h
      const fd =
        (structuredObjective(oracle, compiled, up, penalty).value -
          structuredObjective(oracle, compiled, dn, penalty).value) /
        (2 * h)
      expect(g[q]).toBeCloseTo(fd, 6)
    }
  })

  it('without gradientInto, and with a clamped function, the generic accumulation gives the same objective', () => {
    const { gradientInto: _unused, ...plain } = oracle
    const generic: StructuredOracle<ChainInput, ChainLabels> = plain
    const want = structuredObjective(oracle, data, w, { c2: 0.1 })
    const got = structuredObjective(generic, data, w, { c2: 0.1 })
    expect(got.value).toBeCloseTo(want.value, 10)
    for (const [q, v] of got.gradient.entries()) expect(v).toBeCloseTo(want.gradient[q] ?? Number.NaN, 10)
    // a clamped function that clamps to the labelling itself is the same objective
    const clamped: typeof generic = {
      ...generic,
      clamped: (wv, x, y) => ({
        counts: generic.features(x, y),
        logPartition: Array.from(generic.features(x, y)).reduce((a, v, q) => a + v * (wv[q] ?? 0), 0),
      }),
    }
    expect(structuredObjective(clamped, data, w, { c2: 0.1 }).value).toBeCloseTo(want.value, 10)
  })

  it('batch estimates are unbiased: the mean over singleton batches is the full objective', () => {
    const full = structuredObjective(oracle, compiled, w, { c2: 0.1 })
    const singles = compiled.map((_, i) => structuredObjective(oracle, compiled, w, { c2: 0.1 }, [i]))
    expect(singles.reduce((a, r) => a + r.value, 0) / 3).toBeCloseTo(full.value, 10)
    const mean = Float64Array.from(full.gradient, (_, q) => singles.reduce((a, r) => a + (r.gradient[q] ?? 0), 0) / 3)
    for (const [q, v] of mean.entries()) expect(v).toBeCloseTo(full.gradient[q] ?? Number.NaN, 10)
  })

  it('the smooth problem evaluates the same value and gradient through the tensor wrapper and its batches', () => {
    const problem = structuredSmoothProblem({ oracle, data: compiled, penalty: { c2: 0.1 } })
    expect(problem.batches?.n).toBe(3)
    expect(problem.x0).toHaveLength(P)
    const f = problem.objective as (x: unknown, batch?: readonly number[]) => Value
    const at = (batch?: readonly number[]): number => toFlat(f(asTensor(Float64Array.from(w)), batch))[0] ?? Number.NaN
    expect(at()).toBeCloseTo(structuredObjective(oracle, compiled, w, { c2: 0.1 }).value, 10)
    expect(at([0, 2])).toBeCloseTo(structuredObjective(oracle, compiled, w, { c2: 0.1 }, [0, 2]).value, 10)
    // the gradient of the wrapped objective is the analytic one
    const g = grad((x: Value) => f(x))(asTensor(Float64Array.from(w)))
    expect(Array.from(toFlat(g))[1]).toBeCloseTo(
      structuredObjective(oracle, compiled, w, { c2: 0.1 }).gradient[1] ?? Number.NaN,
      10,
    )
    expect(structuredSmoothProblem({ oracle, data: compiled, start: w }).x0).toEqual(w)
  })
})
