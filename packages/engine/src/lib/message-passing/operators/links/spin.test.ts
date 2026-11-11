// The spin site against its closed form: the tilted distribution of a Gaussian cavity times exp(field x) on the points
// x = -1 and x = +1 has the mean tanh(field + h) and variance 1 - mean^2, and the mass is the sum of the two weights.
import { describe, expect, it } from 'vitest'
import { messageArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { spin } from './spin.ts'

const ctx = (eta: readonly number[], field: number) => ({
  mode: 'ep' as const,
  to: 0,
  args: [messageArg(gaussian, eta)],
  params: { field },
  power: 1,
})

describe('spin', () => {
  it('sends the Gaussian that turns the cavity into the two-point tilted distribution', () => {
    const [h, j, field] = [0.3, 2, 0.5]
    const mean = Math.tanh(field + h)
    const variance = 1 - mean * mean
    const message = spin.ep?.(ctx([h, j], field)).message
    expect(message?.[0]).toBeCloseTo(mean / variance - h, 12)
    expect(message?.[1]).toBeCloseTo(1 / variance - j, 12)
  })

  it('has the mass of the two weights', () => {
    const [h, j, field] = [0.3, 2, 0.5]
    const plus = Math.exp(field + h - j / 2)
    const minus = Math.exp(-field - h - j / 2)
    expect(spin.logEvidence?.(ctx([h, j], field))).toBeCloseTo(Math.log(plus + minus), 12)
  })

  it('has no evidence against a cavity without a density', () => {
    expect(spin.logEvidence?.(ctx([0, 0], 0))).toBeNaN()
    expect(spin.logEvidence?.(ctx(gaussianEta(0, 1), 0))).toBeCloseTo(-0.5 + Math.log(2), 12)
  })

  it('declares exact EP and no variational rule', () => {
    expect(spin.support).toEqual({ ep: 'exact' })
    expect(spin.vmp).toBeUndefined()
    expect(spin.noRule?.vmp).toContain('two point masses')
  })
})
