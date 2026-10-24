import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { choice, contains, dict, int, real } from '../space/index.ts'
import {
  createStudy,
  functionObjective,
  hypervolumeSubsetSelection,
  leavesOfSpace,
  multiObjective,
  parzenEstimator,
  parzenLogPdf,
  parzenSample,
  tpeSampler,
  tune,
} from './index.ts'

describe('Parzen estimators', () => {
  it('integrate to one over a stepped leaf and an option leaf', () => {
    const leaves = leavesOfSpace(dict({ k: int(1, 6), c: choice(['a', 'b', 'c']) }))
    const pe = parzenEstimator(leaves, [
      [2, 0],
      [5, 2],
      [3, 0],
    ])
    let total = 0
    for (let k = 1; k <= 6; k++) for (let c = 0; c < 3; c++) total += Math.exp(parzenLogPdf(pe, [[k, c]])[0] ?? 0)
    expect(total).toBeCloseTo(1, 12)
  })
  it('draw inside the bounds and on the step', () => {
    const leaves = leavesOfSpace(dict({ s: real(0, 2, { step: 0.5 }), lr: real(1e-3, 1, { log: true }) }))
    const pe = parzenEstimator(leaves, [
      [0.5, 0.01],
      [1.5, 0.1],
    ])
    for (const [s, lr] of parzenSample(pe, key(0), 200)) {
      expect([0, 0.5, 1, 1.5, 2]).toContain(s)
      expect(lr).toBeGreaterThanOrEqual(1e-3)
      expect(lr).toBeLessThanOrEqual(1)
    }
  })
  it('reject invalid weights', () => {
    const leaves = leavesOfSpace(real(0, 1))
    expect(() => parzenEstimator(leaves, [[0.5]], { priorWeight: -1 })).toThrow()
    expect(() => parzenEstimator(leaves, [[0.5]], { predeterminedWeights: [1, 2] })).toThrow()
    expect(() => parzenEstimator(leaves, [[0.5]], { predeterminedWeights: [-1] })).toThrow()
  })
})

describe('the TPE sampler', () => {
  it('handles conditional spaces, sampling inactive keys never', () => {
    const space = dict({
      kind: choice(['linear', 'tree']),
      alpha: real(1e-4, 1, { log: true, when: { key: 'kind', in: ['linear'] } }),
      depth: int(1, 10, { when: { key: 'kind', in: ['tree'] } }),
    })
    const objective = functionObjective((p) => {
      const v = p as { kind: string; alpha?: number; depth?: number }
      return v.kind === 'tree' ? Math.abs((v.depth ?? 0) - 7) : 1 + Math.abs(Math.log10(v.alpha ?? 1) + 2)
    })
    for (const multivariate of [true, false]) {
      const r = tune(objective, space, { sampler: tpeSampler({ multivariate }), nTrials: 60, seed: 0 })
      for (const t of r.study.trials) expect(contains(space, t.params)).toBe(true)
      expect(r.best?.params).toEqual({ kind: 'tree', depth: 7 })
    }
  })
  it('optimises two objectives (MOTPE) towards the front', () => {
    const both = multiObjective([
      functionObjective((p) => (p as number[])[0] as number),
      functionObjective((p) => {
        const [a = 0, b = 0] = p as number[]
        return (1 + b) * (1 - Math.sqrt(a))
      }),
    ])
    const r = tune(both, { kind: 'box', low: 0, high: 1, shape: [2] }, { sampler: tpeSampler(), nTrials: 80, seed: 0 })
    expect(r.bestTrials.length).toBeGreaterThan(3)
    expect(Math.min(...r.bestTrials.map((t) => (t.params as number[])[1] ?? 1))).toBeLessThan(0.1)
  })
  it('selects hypervolume subsets greedily, distinct points first', () => {
    expect(
      hypervolumeSubsetSelection(
        [
          [1, 1],
          [1, 1],
          [2, 0],
        ],
        2,
        [3, 3],
      ),
    ).toEqual([0, 2])
    expect(
      hypervolumeSubsetSelection(
        [
          [1, 1],
          [1, 1],
        ],
        2,
        [3, 3],
      ),
    ).toEqual([0, 1])
    expect(
      hypervolumeSubsetSelection(
        [
          [1, 1],
          [1, 1],
          [0, 2],
        ],
        3,
        [3, 3],
      ),
    ).toEqual([0, 1, 2])
    expect(createStudy(real(0, 1)).trials).toEqual([])
  })
})
