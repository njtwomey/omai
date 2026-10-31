import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  affinityPropagation,
  affinityPropagationAlgorithm,
  affinityPropagationFit,
  affinitySimilarities,
} from './exemplar.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 60, centers: 3, clusterStd: 0.5 } })

describe('affinity propagation', () => {
  it('finds one exemplar per blob and predicts to the nearest exemplar', () => {
    const f = affinityPropagation.with({ preference: -50 }).fit(data, { key: key(0) })
    expect(f.params.exemplarIndices.length).toBe(3)
    expect(adjustedRandScore(data.y, f.params.labels).estimate.value).toBeGreaterThan(0.99)
    expect(toFlat(f.predict(data.x))).toEqual(f.params.labels)
  })

  it('breaks exact ties with keyed noise, as sklearn does, and converges at damping 0.5', () => {
    // Three pairs of points: every pair's two members are equally good exemplars.
    const x = [
      [0, 0],
      [0.1, 0],
      [5, 5],
      [5.1, 5],
      [0, 5],
      [0.1, 5],
    ]
    for (const k of [0, 1, 2]) {
      const f = affinityPropagationFit(x, { damping: 0.5, noiseKey: key(k) })
      expect(f.labels).toEqual([0, 0, 1, 1, 2, 2])
      expect(f.nIter).toBeLessThan(100)
    }
  })

  it('a session shows the exemplars emerging; the state round-trips', () => {
    const s = affinityPropagation.with({ preference: -50 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect((s.probe(3, 'model/assignments') as number[]).length).toBe(60)
    const { n, s: sim } = affinitySimilarities(data.x, { preference: -50 })
    const state = runAlgorithm(affinityPropagationAlgorithm(sim, n), { maxSteps: 2 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})
