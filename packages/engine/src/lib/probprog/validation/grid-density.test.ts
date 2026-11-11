// The grid posterior's marginal density and joint on the constrained scale: they integrate to one and agree with each other.
import { describe, expect, it } from 'vitest'
import { gamma, normal } from '../../dists/index.ts'
import { observe, sample } from '../index.ts'
import { gridPosterior } from './grid.ts'

const model = (): void => {
  const m = sample('m', normal(0, 1))
  sample('s', gamma(2, 2))
  observe('y', normal(m, 1), 1.2)
}

describe('gridPosterior density and joint', () => {
  const g = gridPosterior(model, undefined, { points: 41, maxRefinements: 0 })

  it('the marginal density of a positive coordinate integrates to one and has the oracle mean', () => {
    const one = gridPosterior(() => void sample('s', gamma(2, 2)), undefined)
    const { x, p } = one.density('s')
    let area = 0
    let mean = 0
    for (let i = 1; i < x.length; i++) {
      const w = (((p[i] ?? 0) + (p[i - 1] ?? 0)) / 2) * ((x[i] ?? 0) - (x[i - 1] ?? 0))
      area += w
      mean += w * (((x[i] ?? 0) + (x[i - 1] ?? 0)) / 2)
    }
    expect(area).toBeCloseTo(1, 2)
    expect(mean / area).toBeCloseTo(one.marginal('s').mean, 2)
  })

  it('the joint integrates to one over the constrained plane', () => {
    const j = g.joint('m', 's')
    let total = 0
    for (let a = 1; a < j.x.length; a++)
      for (let b = 1; b < j.y.length; b++)
        total +=
          (((j.p[a]?.[b] ?? 0) + (j.p[a - 1]?.[b] ?? 0) + (j.p[a]?.[b - 1] ?? 0) + (j.p[a - 1]?.[b - 1] ?? 0)) / 4) *
          ((j.x[a] ?? 0) - (j.x[a - 1] ?? 0)) *
          ((j.y[b] ?? 0) - (j.y[b - 1] ?? 0))
    expect(total).toBeCloseTo(1, 1)
  })
})
