import { describe, expect, it } from 'vitest'
import { gradCheck } from '../../../lib/autodiff/index.ts'
import { normal, key as rngKey, uniform } from '../../../lib/random/index.ts'
import { type Tensor, tensor, toArray, toFlat, type Value } from '../../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { categoricalProjection, quantileHuberLoss } from './distributional.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]

describe('distributional kernels against numpy transcriptions of the papers', () => {
  for (const c of loadFixture('methods/rl/agents/distributional').cases) {
    it(c.name, () => {
      const [a, b, d] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor, Tensor]
      const got =
        c.fn === 'categoricalProjection'
          ? tensor(
              categoricalProjection(
                rows(a),
                toFlat(b),
                toFlat(d),
                c.params['vMin'] as number,
                c.params['vMax'] as number,
              ),
            )
          : quantileHuberLoss(a, b, d, { kappa: c.params['kappa'] as number, reduction: 'none' })
      expectMatches(got, c.output, Math.max(c.rtol, 1e-12), Math.max(c.atol, 1e-12), c.name)
    })
  }
})

describe('the quantile Huber kernel', () => {
  it('has the gradient of its definition in quantiles, targets and fractions', () => {
    const targets = normal(rngKey(1), [3, 4])
    const taus = uniform(rngKey(2), [3, 5])
    const theta = normal(rngKey(3), [3, 5])
    for (const kappa of [0.5, 1, 2]) {
      expect(gradCheck((x: Value) => quantileHuberLoss(x, targets, taus, { kappa }), theta).passed).toBe(true)
      expect(
        gradCheck((t: Value) => quantileHuberLoss(theta, t, taus, { kappa, reduction: 'sum' }), targets).passed,
      ).toBe(true)
      expect(gradCheck((t: Value) => quantileHuberLoss(theta, targets, t, { kappa }), taus).passed).toBe(true)
    }
  })
})
