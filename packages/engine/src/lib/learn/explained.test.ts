import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares } from '../../testing/learn.ts'
import { coefficientTable } from '../exposition/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import { collecting, decisionSurfaceOf, defineExplainedEstimator, ExplainedEstimatorClass } from './explained.ts'

const counting = defineExplainedEstimator(
  {
    name: 'counting',
    kind: 'other',
    fit: (data, h: { readonly scale: number }) => {
      const n = (data.x as number[][]).length
      return { params: { n: n * h.scale }, exposition: [coefficientTable({ names: ['n'], estimates: [n * h.scale] })] }
    },
    methods: () => ({}),
  },
  { scale: 1 },
)

describe('explained estimators', () => {
  it('fit carries records, revive has none, with re-applies the spec', () => {
    const f = counting.fit({ x: [[1], [2]] })
    expect(f.params).toEqual({ n: 2 })
    expect(f.exposition.map((r) => r.kind)).toEqual(['coefficientTable'])
    expect(counting.revive({ n: 3 }).exposition).toEqual([])
    const g = counting.with({ scale: 2 }).fit({ x: [[1], [2]] })
    expect(g.params).toEqual({ n: 4 })
    expect(g.exposition).toHaveLength(1)
  })

  it('the class wrapper exposes the latest records', () => {
    const c = new ExplainedEstimatorClass(counting).fit([[0], [0], [0]])
    expect(c.exposition).toHaveLength(1)
  })
})

describe('decision surfaces', () => {
  it('classifiers give labels and probabilities on the grid', () => {
    const x = [
      [0, 0],
      [0, 1],
      [3, 3],
      [3, 4],
    ]
    const f = gaussianNB().fit({ x, y: [0, 0, 1, 1] })
    const s = decisionSurfaceOf(f, x, { resolution: 5 })
    expect(s.x).toHaveLength(5)
    expect(s.labels?.[0]?.[0]).toBe(0)
    expect(s.labels?.[4]?.[4]).toBe(1)
    expect(s.probabilities).toHaveLength(2)
    expect(s.classes).toEqual(['0', '1'])
  })

  it('regressors give values; one feature gives one row', () => {
    const f = leastSquares().fit({ x: [[0], [1]], y: [0, 2] })
    const s = decisionSurfaceOf(f, [[0], [1]], { resolution: 3, padding: 0 })
    expect(s.values).toHaveLength(1)
    expect(s.values?.[0]?.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([0, 1, 2])
  })
})

describe('collecting', () => {
  it('collects every state whatever the tracer', () => {
    const c = collecting({ name: 'up', config: {}, init: () => 0, step: (s: number) => s + 1 }, (s) => s * 10)
    const r = runAlgorithm(c.algorithm, { maxSteps: 3, tracer: recorder() })
    expect(r.state).toBe(3)
    expect(c.history).toEqual([0, 10, 20, 30])
    runAlgorithm(c.algorithm, { maxSteps: 1 })
    expect(c.history).toEqual([0, 10])
  })
})
