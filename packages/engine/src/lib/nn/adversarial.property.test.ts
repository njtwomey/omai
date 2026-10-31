import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, uniform } from '../random/index.ts'
import { type Tensor, toFlat, type Value } from '../tensor/index.ts'
import * as NN from './index.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)

describe('PGD stays inside its ball', () => {
  it('for any network, radius, norm, step count and range', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 1000 }),
        fc.double({ min: 0.01, max: 2, noNaN: true }),
        fc.constantFrom('linf' as const, 'l2' as const),
        fc.integer({ min: 1, max: 8 }),
        fc.boolean(),
        fc.boolean(),
        (seed, epsilon, norm, steps, targeted, ranged) => {
          const net = NN.mlp([3, 5, 3], { key: key(seed) })
          const f: NN.LogitsFunction = (v) => net.apply(net.params, v, { train: false })
          const x = uniform(key(seed + 1), [4, 3], { low: -1.4, high: 1.4 }) as Tensor
          const labels = [0, 1, 2, 1]
          const range = ranged ? ([-1.5, 1.5] as const) : undefined
          const adv = flat(
            NN.pgd(f, x, labels, {
              epsilon,
              norm,
              steps,
              targeted,
              key: key(seed + 2),
              ...(range === undefined ? {} : { range }),
            }),
          )
          const x0 = flat(x)
          for (let r = 0; r < 4; r++) {
            const d = [0, 1, 2].map((k) => (adv[3 * r + k] as number) - (x0[3 * r + k] as number))
            const size = norm === 'linf' ? Math.max(...d.map(Math.abs)) : Math.hypot(...d)
            // The range clip is applied after the projection and can only shrink the clipped coordinates when the input is inside it.
            const slack =
              range === undefined
                ? 1e-12
                : 1e-12 + Math.max(0, ...x0.slice(3 * r, 3 * r + 3).map((v) => Math.abs(v) - 1.5))
            expect(size).toBeLessThanOrEqual(epsilon + slack)
            if (range !== undefined) expect(adv.slice(3 * r, 3 * r + 3).every((v) => v >= -1.5 && v <= 1.5)).toBe(true)
          }
        },
      ),
      { numRuns: 60 },
    )
  })

  it('the margin loss is positive exactly when the label is lost', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: 4, maxLength: 4 }), {
          minLength: 1,
          maxLength: 5,
        }),
        fc.integer({ min: 0, max: 3 }),
        (logits, label) => {
          const m = flat(
            NN.marginLoss(
              logits,
              logits.map(() => label),
            ),
          )
          logits.forEach((row, i) => {
            const lost = row.some((v, j) => j !== label && v > (row[label] as number))
            expect((m[i] as number) > 0).toBe(lost)
          })
        },
      ),
    )
  })
})
