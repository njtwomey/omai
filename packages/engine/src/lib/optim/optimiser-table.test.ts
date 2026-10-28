import { describe, expect, it } from 'vitest'
import { defaults, grid } from '../space/index.ts'
import { tensor, toArray } from '../tensor/index.ts'
import { optimisers } from './optimiser-table.ts'

describe('optimisers', () => {
  it('is one slot with SGD as the default, and every option builds a transformation that descends', () => {
    expect(defaults(optimisers.space)).toMatchObject({ kind: 'sgd', momentum: 0, nesterov: false })
    for (const config of grid(optimisers.space, 2)) {
      const t = optimisers.make(config)(0.1)
      const g = tensor([1, -2])
      const { updates } = t.update(g, t.init(tensor([0, 0])), tensor([0, 0]))
      const u = toArray(updates as never) as number[]
      // A step against the gradient: the update's sign is opposite to the gradient's, coordinate by coordinate.
      expect(u[0], JSON.stringify(config)).toBeLessThan(0)
      expect(u[1], JSON.stringify(config)).toBeGreaterThan(0)
    }
  })

  it('takes a schedule as the learning rate, and fills the knobs a configuration leaves out', () => {
    const t = optimisers.make({ kind: 'adam' })((step) => 0.5 / (1 + step))
    const first = toArray(t.update(tensor([1]), t.init(tensor([0]))).updates as never) as number[]
    expect(first[0]).toBeCloseTo(-0.5, 6)
    expect(optimisers.make({ kind: 'nope' })(0.1).name).toContain('addDecayedWeights')
  })

  it('records what each option does, for validity rules', () => {
    expect(optimisers.where((p) => p.decoupledDecay)).toEqual(['adamw', 'lion'])
    expect(optimisers.where((p) => !p.adaptive)).toEqual(['sgd', 'lion'])
  })
})
