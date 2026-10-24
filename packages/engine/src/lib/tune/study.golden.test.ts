import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { real } from '../space/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { bestTrials, createStudy, type StudyDirection, studyHypervolume, type Trial } from './index.ts'

const fixture = loadFixture('tune/study')

describe('pareto fronts and hypervolume against optuna', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const v = decode(c.inputs[0]) as Tensor
      const [n = 0, m = 0] = v.shape
      const flat = toFlat(v)
      const directions = c.params['directions'] as StudyDirection[]
      const trials: Trial[] = Array.from({ length: n }, (_, i) => ({
        number: i,
        params: 0,
        state: 'complete',
        values: flat.slice(i * m, (i + 1) * m),
        intermediate: [],
        source: 'fixture',
      }))
      for (let k = 0; k < Number(c.params['failed']); k++)
        trials.push({ number: n + k, params: 0, state: 'failed', values: [], intermediate: [], source: 'fixture' })
      const study = createStudy(real(0, 1), { directions, trials })
      const out = c.output as { best: number[]; hypervolume: number }
      expect(bestTrials(study).map((t) => t.number)).toEqual(out.best)
      const hv = studyHypervolume(study, c.params['reference'] as number[])
      expect(Math.abs(hv - out.hypervolume)).toBeLessThanOrEqual(1e-12 + 1e-12 * out.hypervolume)
    })
  }
})
