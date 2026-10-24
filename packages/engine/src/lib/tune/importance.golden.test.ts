import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { createStudy, pedAnovaImportance, type StudyDirection, type Trial } from './index.ts'

const fixture = loadFixture('tune/importance')

const space = (conditional: boolean) =>
  dict({
    x: real(-5, 5),
    lr: real(1e-4, 1, { log: true }),
    k: int(1, 20),
    kl: int(1, 512, { log: true }),
    s: real(0, 5, { step: 0.25 }),
    act: choice(['a', 'b', 'c']),
    ...(conditional ? { depth: int(1, 8, { when: { key: 'act', in: ['c'] } }) } : {}),
  })

describe('PED-ANOVA against optuna', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const trials = (c.inputs[0] as Omit<Trial, 'source'>[]).map((t) => ({ ...t, source: 'fixture' }))
      const study = createStudy(space(c.params['conditional'] === true), {
        directions: [c.params['direction'] as StudyDirection],
        trials,
      })
      const got = pedAnovaImportance(study, {
        targetQuantile: Number(c.params['targetQuantile']),
        regionQuantile: Number(c.params['regionQuantile']),
        evaluateOnLocal: c.params['evaluateOnLocal'] === true,
      })
      const want = c.output as Record<string, number>
      for (const [k, w] of Object.entries(want)) {
        const g = got[k] ?? Number.NaN
        expect(Math.abs(g - w) <= c.atol + c.rtol * Math.abs(w), `${k}: ${g} vs ${w}`).toBe(true)
      }
      expect(Object.keys(got).filter((k) => (got[k] ?? 0) > 0)).toEqual(
        Object.keys(want).filter((k) => (want[k] ?? 0) > 0),
      )
    })
  }
})
