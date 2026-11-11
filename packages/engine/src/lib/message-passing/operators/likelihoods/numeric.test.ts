// The numeric operator against closed forms: the factor exp(-x^2 / 2) over a Gaussian cavity has a Gaussian tilted
// distribution, so the quadrature must give its moments, its message and its mass to quadrature accuracy.
import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { numericOperator } from './numeric.ts'

const bell = numericOperator({ name: 'bell', arity: 1, logDensity: ([x]) => -0.5 * (x as number) ** 2 })
const at = (mean: number, variance: number, power = 1) => ({
  mode: 'ep' as const,
  to: 0,
  args: [messageArg(gaussian, gaussianEta(mean, variance))],
  params: {},
  power,
})

describe('numericOperator', () => {
  it('finds the moments of the tilted distribution by quadrature', () => {
    // N(m, v) times exp(-x^2 / 2): precision 1 / v + 1, mean (m / v) / (1 / v + 1); the moments are E x and E x^2
    const [m, v] = [1, 2]
    const precision = 1 / v + 1
    const mean = m / v / precision
    const t = bell.tilted?.(at(m, v))
    expect(t?.moments[0]).toBeCloseTo(mean, 8)
    expect(t?.moments[1]).toBeCloseTo(1 / precision + mean * mean, 8)
  })

  it('has the mass of the factor against the unnormalised cavity', () => {
    // the cavity exp(-x^2 / 2) times the factor exp(-x^2 / 2) integrates to sqrt(pi)
    expect(bell.logEvidence?.(at(0, 1))).toBeCloseTo(0.5 * Math.log(Math.PI), 6)
  })

  it('supports a power: the factor raised to it is what is integrated', () => {
    // the factor to the power 2 is exp(-x^2): over the cavity exp(-x^2 / 2) the mass is sqrt(2 pi / 3)
    expect(bell.logEvidence?.(at(0, 1, 2))).toBeCloseTo(0.5 * Math.log((2 * Math.PI) / 3), 6)
  })

  it('takes observed arguments and refuses an arity it cannot integrate', () => {
    const observed = numericOperator({ name: 'obs', arity: 1, logDensity: ([x]) => -0.5 * (x as number) ** 2 })
    expect(
      observed.logEvidence?.({ mode: 'ep', to: 0, args: [valueArg(gaussian, 2)], params: {}, power: 1 }),
    ).toBeCloseTo(-2, 10)
    expect(() => numericOperator({ name: 'x', arity: 4, logDensity: () => 0 })).toThrow('the arity must be 1, 2 or 3')
  })
})
