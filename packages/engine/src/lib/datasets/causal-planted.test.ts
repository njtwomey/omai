import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { syntheticControlPanel, upliftData } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)

describe('syntheticControlPanel', () => {
  const { data, truth } = generate(syntheticControlPanel, {
    key: key(1),
    knobs: { donors: 12, periods: 20, start: 14, effect: 2, growth: 0.5, noise: 0 },
  })
  const extra = truth.extra as { donorWeights: number[]; effectPath: number[]; counterfactual: number[]; start: number }
  const y = Array.from(toFlat(data.y))

  it('has weights on the simplex and a treated path equal to its counterfactual plus the effect', () => {
    expect(extra.donorWeights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(extra.donorWeights.filter((w) => w > 0).length).toBe(3)
    for (let t = 0; t < 20; t++) {
      expect(y[t]).toBeCloseTo((extra.counterfactual[t] ?? 0) + (extra.effectPath[t] ?? 0), 12)
      expect(extra.effectPath[t]).toBe(t >= 14 ? 2 * (1 + 0.5 * (t - 14)) : 0)
    }
    expect(truth.effects?.att).toBeCloseTo(mean(extra.effectPath.slice(14)), 12)
  })

  it('without noise the donors reproduce the counterfactual exactly with the planted weights', () => {
    for (let t = 0; t < 20; t++) {
      let synthetic = 0
      extra.donorWeights.forEach((w, j) => {
        synthetic += w * (y[(j + 1) * 20 + t] ?? 0)
      })
      expect(synthetic).toBeCloseTo(extra.counterfactual[t] ?? 0, 10)
    }
  })

  it('marks only the treated unit after the start as treated', () => {
    const x = tensor(Array.from(toFlat(data.x)) as number[]) as Tensor
    const flat = toFlat(x)
    let treated = 0
    for (let r = 0; r < 12 * 20 + 20; r++) treated += flat[r * 3 + 2] ?? 0
    expect(treated).toBe(6)
  })
})

describe('upliftData', () => {
  const { data, truth } = generate(upliftData, { key: key(2), knobs: { n: 20000, covariates: 3 } })
  const x = toFlat(data.x)
  const n = 20000
  const t = Array.from({ length: n }, (_, i) => x[i * 4 + 3] ?? 0)
  const x0 = Array.from({ length: n }, (_, i) => x[i * 4] ?? 0)

  it('assigns treatment independently of the covariates, at the requested share', () => {
    expect(mean(t)).toBeGreaterThan(0.48)
    expect(mean(t)).toBeLessThan(0.52)
    const treatedMean = mean(x0.filter((_, i) => t[i] === 1))
    expect(Math.abs(treatedMean - mean(x0.filter((_, i) => t[i] === 0)))).toBeLessThan(0.05)
  })

  it('has individual effects in {-1, 0, 1}, harmed units, and an uplift function that matches them on average', () => {
    const ite = Array.from(toFlat(truth.effects?.ite as Tensor))
    expect(new Set(ite)).toEqual(new Set([-1, 0, 1]))
    expect(ite.filter((v) => v === -1).length).toBeGreaterThan(0)
    expect(Math.abs(mean(ite) - (truth.effects?.ate ?? 0))).toBeLessThan(0.01)
    const cate = truth.effects?.cate
    const grid = Array.from(
      toFlat(
        (cate as (v: Tensor) => Tensor)(
          tensor([
            [-2, 0, 0, 0],
            [3, 0, 0, 0],
          ]) as Tensor,
        ),
      ),
    )
    expect(grid[0]).toBeLessThan(0)
    expect(grid[1]).toBeGreaterThan(0)
  })

  it('the observed outcome is the potential outcome of the arm', () => {
    const y = Array.from(toFlat(data.y))
    const po = truth.effects?.potentialOutcomes
    const c = Array.from(toFlat(po?.control as Tensor))
    const tr = Array.from(toFlat(po?.treated as Tensor))
    for (let i = 0; i < 500; i++) expect(y[i]).toBe(t[i] === 1 ? tr[i] : c[i])
  })
})
