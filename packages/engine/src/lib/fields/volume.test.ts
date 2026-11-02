import { describe, expect, it } from 'vitest'
import { gradCheck } from '../autodiff/index.ts'
import { add, mul, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import { volumeRender } from './volume.ts'

const uniformRay = (n: number, length: number): { t: Value; delta: Value } => {
  const h = length / n
  return {
    t: tensor([Array.from({ length: n }, (_, i) => i * h)]),
    delta: tensor([Array.from({ length: n }, () => h)]),
  }
}
const red = (n: number): Value => tensor([Array.from({ length: n }, () => [1, 0, 0])])

describe('volumeRender', () => {
  it('has the closed form for a constant density: alpha = 1 - exp(-sigma L), the weights sum to alpha', () => {
    for (const [sigma, L, n] of [
      [1, 1, 2],
      [0.5, 4, 40],
      [3, 2, 17],
    ] as const) {
      const { t, delta } = uniformRay(n, L)
      const r = volumeRender(tensor([Array.from({ length: n }, () => sigma)]), red(n), t, { delta })
      const alpha = 1 - Math.exp(-sigma * L)
      expect(toFlat(r.alpha)[0]).toBeCloseTo(alpha, 12)
      expect(toFlat(r.weights).reduce((a, b) => a + b, 0)).toBeCloseTo(alpha, 12)
      expect(toFlat(r.colour)).toEqual([expect.closeTo(alpha, 12), 0, 0])
      // The geometric weights: w_i = exp(-sigma h i) (1 - exp(-sigma h)).
      const h = L / n
      for (const [i, w] of toFlat(r.weights).entries())
        expect(w).toBeCloseTo(Math.exp(-sigma * h * i) * (1 - Math.exp(-sigma * h)), 12)
      // Transmittance in front of the first sample is one.
      expect(toFlat(r.transmittance)[0]).toBe(1)
    }
  })

  it('returns an opaque slab’s colour and the depth of its front face', () => {
    const t = tensor([[1, 2, 3, 4, 5, 6]])
    const density = tensor([[0, 0, 1e4, 1e4, 0, 0]])
    const colour = tensor([
      [
        [0, 0, 0],
        [0, 0, 0],
        [0.2, 0.5, 0.9],
        [0.9, 0.9, 0.9],
        [0, 0, 0],
        [0, 0, 0],
      ],
    ])
    const r = volumeRender(density, colour, t, { delta: tensor([[1, 1, 1, 1, 1, 1]]) })
    expect(toFlat(r.colour).map((v) => Math.round(v * 1e6) / 1e6)).toEqual([0.2, 0.5, 0.9])
    expect(toFlat(r.depth)[0]).toBeCloseTo(3, 6)
    expect(toFlat(r.alpha)[0]).toBeCloseTo(1, 6)
  })

  it('has non-negative weights summing to at most one, and transmittance decreasing from one', () => {
    const density = tensor([
      [0.3, 2, 0, 5, 0.1, 1],
      [0, 0, 0, 0, 0, 0],
    ])
    const colour = tensor([
      Array.from({ length: 6 }, () => [0.5, 0.5, 0.5]),
      Array.from({ length: 6 }, () => [0.5, 0.5, 0.5]),
    ])
    const t = tensor([
      [1, 2, 3, 4, 5, 6],
      [1, 2, 3, 4, 5, 6],
    ])
    const r = volumeRender(density, colour, t)
    const w = toFlat(r.weights)
    expect(w.every((v) => v >= 0)).toBe(true)
    expect(w.slice(0, 6).reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(1)
    expect(w.slice(6).every((v) => v === 0)).toBe(true)
    const T = toFlat(r.transmittance).slice(0, 6)
    expect(T[0]).toBe(1)
    for (const [i, v] of T.entries()) if (i > 0) expect(v).toBeLessThanOrEqual(T[i - 1] as number)
  })

  it('by default spaces samples by their gaps, and the last one absorbs what is left', () => {
    const r = volumeRender(
      tensor([[0, 0, 1]]),
      tensor([
        [
          [0, 0, 0],
          [0, 0, 0],
          [1, 1, 1],
        ],
      ]),
      tensor([[1, 2, 4]]),
    )
    expect(toFlat(r.alpha)[0]).toBe(1)
    const s = volumeRender(
      tensor([[1, 0]]),
      tensor([
        [
          [1, 0, 0],
          [0, 0, 0],
        ],
      ]),
      tensor([[0, 3]]),
    )
    expect(toFlat(s.alpha)[0]).toBeCloseTo(1 - Math.exp(-3), 12)
  })

  it('composites onto a white or a given background', () => {
    const t = tensor([[1, 2]])
    const empty = volumeRender(
      tensor([[0, 0]]),
      tensor([
        [
          [0, 0, 0],
          [0, 0, 0],
        ],
      ]),
      t,
      { whiteBackground: true },
    )
    expect(toFlat(empty.colour)).toEqual([1, 1, 1])
    const half = volumeRender(
      tensor([[1, 0]]),
      tensor([
        [
          [1, 0, 0],
          [0, 0, 0],
        ],
      ]),
      t,
      { delta: tensor([[Math.LN2, 1]]), background: [0, 0, 1] },
    )
    expect(toFlat(half.colour).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([0.5, 0, 0.5])
  })

  it('is differentiable in the densities, the colours and the distances', () => {
    const t = tensor([[1, 2.2, 3.1, 4.5]])
    const density = tensor([[0.4, 1.3, 0.2, 0.9]])
    const colour = tensor([
      [
        [0.1, 0.2, 0.3],
        [0.9, 0.5, 0.1],
        [0.4, 0.4, 0.8],
        [0.2, 0.7, 0.6],
      ],
    ])
    const scalar = (d: Value, c: Value, tt: Value): Value => {
      const r = volumeRender(d, c, tt, { whiteBackground: true })
      return add(add(sum(mul(r.colour, tensor([[1, 2, 3]]))), mul(sum(r.depth), 0.3)), sum(r.alpha))
    }
    expect(gradCheck((d: Value) => scalar(d, colour, t), density).passed).toBe(true)
    expect(gradCheck((c: Value) => scalar(density, c, t), colour).passed).toBe(true)
    expect(gradCheck((tt: Value) => scalar(density, colour, tt), t, { eps: 1e-6 }).passed).toBe(true)
  })

  it('checks shapes', () => {
    expect(() => volumeRender(tensor([1, 2]), tensor([[[0, 0, 0]]]), tensor([[1]]))).toThrow(/\[rays, n\]/)
    expect(() => volumeRender(tensor([[1, 2]]), tensor([[[0, 0, 0]]]), tensor([[1, 2]]))).toThrow(/colours must be/)
  })
})
