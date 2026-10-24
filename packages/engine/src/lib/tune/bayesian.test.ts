import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { choice, dict, real } from '../space/index.ts'
import { asTensor, item, sum, toFlat, type Value } from '../tensor/index.ts'
import {
  acquisitions,
  expectedImprovement,
  fitGpSurrogate,
  functionObjective,
  gpSampler,
  logExpectedImprovement,
  maximizeAcquisition,
  tune,
} from './index.ts'

describe('acquisitions', () => {
  it('log EI agrees with log of EI where EI is representable, and stays finite beyond', () => {
    const m = [-1, 0, 1, 3, 8]
    const s = [1, 1, 0.5, 0.5, 0.2]
    const ei = toFlat(expectedImprovement(m, s, 0))
    const lei = toFlat(logExpectedImprovement(m, s, 0))
    for (const [i, v] of ei.entries()) if (v > 1e-200) expect(lei[i]).toBeCloseTo(Math.log(v), 8)
    expect(Number.isFinite(toFlat(logExpectedImprovement([60], [1], 0))[0])).toBe(true)
  })
  it('have finite gradients on both sides of the tail switch', () => {
    for (const mu of [10, 30]) {
      const g = grad((m: Value) => sum(logExpectedImprovement(m, 1, 0)))(asTensor([mu])) as Value
      expect(Number.isFinite(item(g))).toBe(true)
      expect(item(g)).toBeLessThan(0)
    }
  })
})

describe('the Gaussian-process surrogate', () => {
  it('interpolates smooth data and is uncertain away from it', () => {
    const xs = [0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => [v])
    const gp = fitGpSurrogate(
      xs,
      xs.map(([v]) => Math.sin(5 * (v ?? 0))),
    )
    const near = gp.predict([[0.3]])
    expect(near.mean[0]).toBeCloseTo(Math.sin(1.5), 1)
    const far = gp.predict([[3]])
    expect(far.std[0] ?? 0).toBeGreaterThan(near.std[0] ?? 0)
  })
  it('handles a constant objective without NaN', () => {
    const xs = [[0.1], [0.5], [0.9]]
    const gp = fitGpSurrogate(xs, [2, 2, 2])
    const p = gp.predict([[0.3]])
    expect(Number.isFinite(p.mean[0])).toBe(true)
    expect(Number.isFinite(p.std[0])).toBe(true)
    const { point } = maximizeAcquisition(gp, acquisitions.logEI, 2, key(0), { nCandidates: 16 })
    expect(point[0]).toBeGreaterThanOrEqual(0)
  })
  it('rejects mismatched data', () => {
    expect(() => fitGpSurrogate([], [])).toThrow()
    expect(() => fitGpSurrogate([[0]], [1, 2])).toThrow()
  })
})

describe('the GP sampler', () => {
  it('optimises a mixed space with an option', () => {
    const space = dict({ x: real(-2, 2), c: choice(['a', 'b', 'c']) })
    const f = functionObjective((p) => {
      const { x, c } = p as { x: number; c: string }
      return (x - 0.5) ** 2 + (c === 'c' ? 0 : 1)
    })
    const r = tune(f, space, { sampler: gpSampler({ nStartupTrials: 5 }), nTrials: 15, seed: 1 })
    expect(r.best?.params).toMatchObject({ c: 'c' })
    expect(r.best?.values[0]).toBeLessThan(0.05)
  })
  it('accepts every named acquisition and a custom one', () => {
    for (const acquisition of ['EI', 'PI', 'LCB', (m: Value) => m] as const) {
      const r = tune(
        functionObjective((x) => (x as number) ** 2),
        real(-1, 1),
        {
          sampler: gpSampler({ nStartupTrials: 3, acquisition, search: { nCandidates: 32, nLocal: 1 } }),
          nTrials: 5,
          seed: 0,
        },
      )
      expect(r.study.trials.length).toBe(5)
    }
  })
})
