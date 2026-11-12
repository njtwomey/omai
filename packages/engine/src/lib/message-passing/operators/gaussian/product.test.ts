import { describe, expect, it } from 'vitest'
import { logNormal, massOf, messageArg, valueArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import type { Arg, OperatorContext } from '../../core/types.ts'
import { product } from './gaussian.ts'

/*
 * The product y = a b when the output's incoming message is so far from the pushforward (more than about 38 sd) that the unshifted
 * Gaussian weights of the grid underflow to zero: the mass was 0, the mean NaN, the message an ImproperMessageError and the evidence
 * minus infinity. With b known the posterior of a is Gaussian in closed form, so the messages and the evidence are exact references:
 * a ~ N(0, 1), b = 1e-4, y observed through a message N(0.05, 1e-6): the likelihood is 49 sd away at every node (about e^-1225).
 */

const ctx = (to: number, args: readonly Arg[]): OperatorContext => ({ mode: 'ep', to, args, params: {}, power: 1 })

const A = { mean: 0, variance: 1 }
const B = 1e-4
const Y = { mean: 0.05, variance: 1e-6 }

const args = (): Arg[] => [
  messageArg(gaussian, gaussianEta(Y.mean, Y.variance)),
  messageArg(gaussian, gaussianEta(A.mean, A.variance)),
  valueArg(gaussian, B),
]

describe('product: an output message far in the tail of the pushforward', () => {
  it('the message to a factor is the exact conjugate one [b y / v, b^2 / v]', () => {
    const m = product.ep?.(ctx(1, args())).message as readonly number[]
    // (the 96-node grid resolves the Gaussian tilt to about 5e-4)
    expect(Math.abs((m[0] as number) / ((B * Y.mean) / Y.variance) - 1)).toBeLessThan(1e-3)
    // (the precision of the tilted a, cavity 1 plus the site, is the one that is compared: the site is a hundredth of it)
    expect(Math.abs((1 + (m[1] as number)) / (1 + (B * B) / Y.variance) - 1)).toBeLessThan(1e-3)
  })

  it('the message to the output is the pushforward N(0, b^2): [0, 1 / b^2]', () => {
    const m = product.ep?.(ctx(0, args())).message as readonly number[]
    expect(Math.abs(m[0] as number)).toBeLessThan(1e-3 * (Y.mean / Y.variance))
    expect(Math.abs((m[1] as number) * B * B - 1)).toBeLessThan(1e-3)
  })

  it('the evidence is the log of N(y; b a, v_y + b^2 v_a) with the masses of the cavities, finite', () => {
    const a = args()
    const overlap = logNormal(Y.mean, B * A.mean, Y.variance + B * B * A.variance)
    const want = a.reduce((s, x) => s + massOf(x), 0) + overlap
    const got = product.logEvidence?.(ctx(0, a)) as number
    expect(overlap).toBeLessThan(-1200)
    expect(Number.isFinite(got)).toBe(true)
    expect(Math.abs(got - want)).toBeLessThan(2e-3)
  })
})
