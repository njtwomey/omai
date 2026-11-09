import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { sparseCodedSignal } from '../datasets/index.ts'
import { dictionaryRecovery } from '../explain/index.ts'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  type DictionaryUpdate,
  dictionaryLearning,
  dictionaryLearningAlgorithm,
  dictLearning,
} from './dictionary-learning.ts'

const planted = generate(sparseCodedSignal, {
  key: key(1),
  knobs: { n: 200, components: 8, dim: 24, nonzeros: 2, noise: 0 },
})
const truth = planted.truth.extra?.['dictionary'] as number[][]
const x = toArray(planted.data.x) as number[][]
// The planted atoms nudged by a fixed pseudo-random wiggle: close enough for alternating minimisation to find the truth.
const nudged = truth.map((row, j) => row.map((v, q) => v + 0.04 * Math.sin(1.7 * j + 0.9 * q)))
const omp = { fitAlgorithm: 'omp', fitNonzeroCoefs: 2 } as const

describe('dictionary learning of planted signals', () => {
  for (const dictUpdate of ['ksvd', 'mod'] as const)
    it(`${dictUpdate} with OMP recovers the planted atoms and codes exactly from a nearby start`, () => {
      const f = dictionaryLearning
        .with({ nComponents: 8, dictUpdate, init: 'custom', dictInit: nudged, maxIter: 60, tol: 0, ...omp })
        .fit({ x })
      expect(f.params.reconstructionErr).toBeLessThan(1e-6)
      expect(dictionaryRecovery(truth, f.params.components).similarity.every((s) => s > 0.99999)).toBe(true)
      for (const row of f.params.codes) expect(row.filter((v) => v !== 0).length).toBeLessThanOrEqual(2)
    })

  it('K-SVD and MOD both bring the error down from the random-signal start, K-SVD at least as far in the end', () => {
    const ends: Record<string, number> = {}
    for (const dictUpdate of ['ksvd', 'mod'] as const) {
      const f = dictionaryLearning
        .with({ nComponents: 8, dictUpdate, init: 'samples', maxIter: 40, tol: 0, ...omp })
        .fit({ x }, { key: key(5), logs: 'all' })
      const error = metricSeries(f.run, 'solver/error').values
      expect(error.at(-1) as number).toBeLessThan((error[1] as number) * 0.9)
      expect(error.at(-1) as number).toBeLessThan((error[0] as number) * 0.5)
      ends[dictUpdate] = error.at(-1) as number
    }
    expect(ends['ksvd']).toBeLessThanOrEqual((ends['mod'] as number) * 1.5)
  })

  it('block coordinate descent with the lasso lowers the objective and keeps atoms in the unit ball', () => {
    const f = dictionaryLearning
      .with({ nComponents: 8, dictUpdate: 'cd', init: 'custom', dictInit: nudged, alpha: 0.05, maxIter: 15, tol: 0 })
      .fit({ x }, { logs: 'all' })
    const objective = metricSeries(f.run, 'solver/objective').values
    for (let t = 1; t < objective.length; t++)
      expect(objective[t] as number).toBeLessThanOrEqual((objective[t - 1] as number) * (1 + 1e-9))
    expect(objective.at(-1) as number).toBeLessThan(objective[0] as number)
    for (const atom of f.params.components) expect(Math.hypot(...atom)).toBeLessThanOrEqual(1 + 1e-12)
    expect(dictionaryRecovery(truth, f.params.components).mean).toBeGreaterThan(0.99)
  })

  it('MOD and K-SVD leave unit atoms', () => {
    for (const dictUpdate of ['ksvd', 'mod'] as const) {
      const f = dictionaryLearning
        .with({ nComponents: 8, dictUpdate, init: 'samples', maxIter: 5, tol: 0, ...omp })
        .fit({ x }, { key: key(2) })
      for (const atom of f.params.components) expect(Math.hypot(...atom)).toBeCloseTo(1, 9)
    }
  })
})

describe('unused atoms', () => {
  it('are replaced by the worst-represented signals, and counted on the tape', () => {
    // Three-dimensional signals and five atoms: the SVD start has two zero atoms nobody can use.
    const y = [
      [2, 0, 0],
      [0, 3, 0],
      [0, 0, 1],
      [1, 1, 0],
      [0, 1, 1],
      [2, 0, 1],
      [1, 2, 0],
      [0, 2, 3],
    ]
    for (const dictUpdate of ['cd', 'mod', 'ksvd'] as DictionaryUpdate[]) {
      const f = dictionaryLearning
        .with({
          nComponents: 5,
          dictUpdate,
          alpha: 0.1,
          maxIter: 3,
          tol: 0,
          fitAlgorithm: dictUpdate === 'cd' ? 'lassoLars' : 'omp',
          fitNonzeroCoefs: 1,
        })
        .fit({ x: y }, { key: key(0), logs: 'all' })
      expect(metricSeries(f.run, 'solver/unused').values[1]).toBeGreaterThanOrEqual(2)
      expect(f.params.components.every((a) => Math.hypot(...a) > 0.5)).toBe(true)
    }
  })
})

describe('the estimator', () => {
  const small = dictionaryLearning.with({ nComponents: 6, alpha: 0.2, maxIter: 8, tol: 0 })

  it('transforms with OMP by default, inverts, splits signs, and is keyed', () => {
    const f = small.fit({ x })
    const z = f.transform(x)
    expect(z.shape).toEqual([200, 6])
    expect(f.inverseTransform(z).shape).toEqual([200, 24])
    expect(small.with({ splitSign: true }).fit({ x }).transform(x).shape).toEqual([200, 12])
    const a = small.with({ init: 'random' }).fit({ x }, { key: key(7) }).params.components
    expect(small.with({ init: 'random' }).fit({ x }, { key: key(7) }).params.components).toEqual(a)
    expect(small.with({ init: 'random' }).fit({ x }, { key: key(8) }).params.components).not.toEqual(a)
  })

  it('exposes the atoms as a record, and a session probes atoms and usage at any step', () => {
    const f = small.fit({ x })
    const record = f.exposition.find((e) => e.kind === 'components')
    expect((record as unknown as { components: number[][] }).components.length).toBe(6)
    const s = small.session({ x }, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    const atoms = s.probe(2, 'dict/atoms') as { kind: string; series: unknown[] }
    expect(atoms.kind).toBe('curves')
    expect(atoms.series.length).toBe(6)
    const usage = s.probe(3, 'dict/usage') as { series: { y: number[] }[] }
    expect(usage.series[0]?.y.every((v) => v >= 0 && v <= 1)).toBe(true)
    expect(s.fittedAt(2).params.components.length).toBe(6)
    const coherence = metricSeries(s.run(), 'solver/coherence').values
    expect(coherence.every((v) => v >= 0 && v <= 1 + 1e-12)).toBe(true)
  })

  it('traces objective, error, sparsity and coherence, and stops when the objective stops falling', () => {
    const f = dictionaryLearning
      .with({ nComponents: 8, alpha: 0.3, maxIter: 200, tol: 1e-4 })
      .fit({ x }, { logs: 'all' })
    expect(f.params.nIter).toBeLessThan(200)
    expect(metricSeries(f.run, 'solver/sparsity').values.length).toBeGreaterThan(2)
    expect(f.params.reconstructionErr).toBeGreaterThan(0)
  })

  it('rejects the options the updates cannot honour, and a custom start of the wrong shape', () => {
    expect(() => dictionaryLearningAlgorithm(x, truth, { dictUpdate: 'ksvd', positiveCode: true })).toThrow(/positive/)
    expect(() => dictionaryLearningAlgorithm(x, truth, { dictUpdate: 'mod', positiveDict: true })).toThrow(/positive/)
    expect(() => small.with({ init: 'custom' }).fit({ x })).toThrow(/dictInit/)
    expect(() => small.with({ init: 'custom', dictInit: [[1, 2]] }).fit({ x })).toThrow(/dictInit/)
  })

  it('keeps atoms and codes non-negative when asked', () => {
    const pos = generate(sparseCodedSignal, {
      key: key(9),
      knobs: { n: 60, components: 4, dim: 8, nonzeros: 2, signs: 'positive' },
    })
    const f = dictionaryLearning
      .with({
        nComponents: 4,
        alpha: 0.05,
        maxIter: 6,
        tol: 0,
        fitAlgorithm: 'lassoCd',
        positiveCode: true,
        positiveDict: true,
        init: 'samples',
      })
      .fit(pos.data, { key: key(1) })
    expect(f.params.components.flat().every((v) => v >= 0)).toBe(true)
    expect(f.params.codes.flat().every((v) => v >= 0)).toBe(true)
  })

  it('states round-trip through JSON', () => {
    const alg = dictionaryLearningAlgorithm(x, nudged, { dictUpdate: 'ksvd', ...omp })
    const { state } = runAlgorithm(alg, { maxSteps: 2 })
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})

describe('dictLearning', () => {
  it("returns scikit-learn's triple: codes, atoms and the objective after every iteration", () => {
    const r = dictLearning(x, 6, 0.2, { maxIter: 5, tol: 0 })
    expect(r.errors.length).toBe(r.nIter)
    expect(r.dictionary.length).toBe(6)
    expect(r.code.length).toBe(200)
    expect(r.dictionary.flat().length).toBe(144)
  })
})
