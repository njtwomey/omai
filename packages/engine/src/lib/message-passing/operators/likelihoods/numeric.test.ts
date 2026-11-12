// The numeric operator against closed forms: the factor exp(-x^2 / 2) over a Gaussian cavity has a Gaussian tilted
// distribution, so the quadrature must give its moments, its message and its mass to quadrature accuracy.
import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { projectSite } from '../../core/define.ts'
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

  it('projects a narrow mass at a large location through its centred variance, not through the second moment', () => {
    // a sharp factor (sd 1e-3) at 1e6 under a wide cavity: the tilted variance is about 1e-6 where the second moment is about 1e12, so
    // the variance recovered as `E x^2 - (E x)^2` is lost to cancellation (about 1e-4) and the message comes out improper or wrong
    const location = 1e6
    const sharp = numericOperator({
      name: 'sharp',
      arity: 1,
      logDensity: ([x]) => -0.5 * (((x as number) - location) / 1e-3) ** 2,
    })
    const context = at(location, 1)
    const message = projectSite(context.args[0] as never, sharp.tilted?.(context) as never, 1)
    // the tilted precision is 1e6 + 1, so the message to the cavity of precision 1 has precision 1e6
    expect(message[1]).toBeCloseTo(1e6, -1)
    expect(Math.abs((message[1] as number) - 1e6)).toBeLessThan(1e3)
    expect((message[0] as number) / (message[1] as number)).toBeCloseTo(location, 3)
  })
})
