// ADVI's step-size sequence (Kucukelbir et al. 2017) against its formula, written out here without the library: the
// update of step k is eta k^(-1/2 + epsilon) g_k / (tau + sqrt(s_k)) with s_1 = g_1^2 and s_k = alpha g_k^2 + (1 - alpha) s_(k-1).
import { describe, expect, it } from 'vitest'
import { tensor, toArray } from '../tensor/index.ts'
import { advi, adviDecay, applyUpdates, optimisers, scaleByAdvi, schedules } from './index.ts'

const gradients = [
  [3, -0.5],
  [1, 2],
  [-2, 0.1],
  [0.4, -4],
  [5, 0.3],
]

describe('ADVI step-size sequence', () => {
  it('matches the formula, coordinate by coordinate', () => {
    const eta = 0.7
    const epsilon = 0.05
    const alpha = 0.1
    const tau = 1
    const opt = advi({ learningRate: adviDecay(eta, epsilon), alpha, tau })
    let state = opt.init(tensor([0, 0]))
    const s: number[] = [0, 0]
    gradients.forEach((g, i) => {
      const k = i + 1
      const r = opt.update(tensor(g), state, tensor([0, 0]))
      state = r.state as typeof state
      const got = toArray(r.updates as never) as number[]
      g.forEach((gj, j) => {
        s[j] = k === 1 ? gj * gj : alpha * gj * gj + (1 - alpha) * (s[j] ?? 0)
        const expected = -(eta * k ** (-0.5 + epsilon) * gj) / (tau + Math.sqrt(s[j] ?? 0))
        expect(got[j], `step ${k} coordinate ${j}`).toBeCloseTo(expected, 12)
      })
    })
  })

  it('scaleByAdvi starts at the first squared gradient, where RMSprop starts at zero', () => {
    const t = scaleByAdvi(0.1, 1)
    expect(toArray(t.update(tensor([3]), t.init(tensor([0]))).updates as never)).toEqual([0.75])
    expect(adviDecay(2)(3)).toBeCloseTo(1, 12)
    expect(adviDecay(1, 0.1)(0)).toBe(1)
  })

  it('is in the tables, so a method takes it as the optimiser with the decay as its schedule', () => {
    expect(optimisers.names).toContain('advi')
    expect(schedules.names).toContain('advi')
    const lr = schedules.make({ kind: 'advi', epsilon: 0.01 })(0.5)
    const opt = optimisers.make({ kind: 'advi', alpha: 0.2, tau: 2 })(lr)
    const step = opt.update(tensor([4]), opt.init(tensor([0])))
    expect((toArray(applyUpdates(tensor([0]), step.updates) as never) as number[])[0]).toBeCloseTo(
      -(0.5 * 4) / (2 + 4),
      12,
    )
  })
})
