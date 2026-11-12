import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { ndtr } from '../numerics/special/index.ts'
import { key } from '../random/index.ts'
import { add, arange, less, reshape, sub, sum, type Tensor, take, tensor, toArray, toFlat } from '../tensor/index.ts'
import { ordinalRatings, ratingCuts } from './index.ts'

describe('ratingCuts', () => {
  it('starts at the first cut and widens by the tilt', () => {
    expect(toArray(ratingCuts(5, -2, 1, 0))).toEqual([-2, -1, 0, 1])
    const tilted = toFlat(ratingCuts(5, 0, 1, 0.5))
    expect(tilted.length).toBe(4)
    expect((tilted[3] ?? 0) - (tilted[2] ?? 0)).toBeGreaterThan((tilted[1] ?? 0) - (tilted[0] ?? 0))
  })
})

describe('ordinalRatings', () => {
  const { data, truth } = generate(ordinalRatings, {
    key: key(2),
    knobs: { raters: 40, items: 50, density: 0.5 },
  })
  const cuts = truth.extra?.['cuts'] as Tensor
  const values = toFlat(data.values)

  it('rates about the stated fraction of the pairs, each on the scale of 1 to 5', () => {
    expect(values.length).toBeGreaterThan(0.45 * 40 * 50)
    expect(values.length).toBeLessThan(0.55 * 40 * 50)
    expect(Math.min(...values)).toBeGreaterThanOrEqual(1)
    expect(Math.max(...values)).toBeLessThanOrEqual(5)
    expect(data.nUsers).toBe(40)
    expect(data.nItems).toBe(50)
  })

  it('has counts that follow the ordered probit of the true latents', () => {
    const quality = truth.latents?.['quality'] as Tensor
    const generosity = truth.latents?.['generosity'] as Tensor
    const mean = add(take(quality, data.items), take(generosity, data.users))
    // the probability that a rating is at most k, for each rating and each cut point
    const upTo = ndtr(sub(reshape(cuts, [1, 4]), reshape(mean, [-1, 1]))) as Tensor
    const expected = toFlat(sum(upTo, 0))
    const observed = toFlat(sum(less(reshape(tensor(values), [-1, 1]), add(1, arange(1, 5))), 0))
    // the rating is at most k exactly when it is below k + 1
    const n = values.length
    for (const [k, e] of expected.entries()) {
      const p = e / n
      expect(Math.abs((observed[k] ?? 0) - e)).toBeLessThan(5 * Math.sqrt(n * p * (1 - p)))
    }
  })

  it('is a function of the key and keeps its truth', () => {
    const again = generate(ordinalRatings, { key: key(2), knobs: { raters: 40, items: 50, density: 0.5 } })
    expect(toFlat(again.data.values)).toEqual(values)
    expect(toFlat(truth.latents?.['quality'] as Tensor).length).toBe(50)
    expect(toFlat(cuts).length).toBe(4)
  })
})
