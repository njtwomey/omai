import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, mul, norm, sum, toFlat, type Value } from '../../tensor/index.ts'
import {
  alibiBias,
  alibiSlopes,
  applyRotary,
  learnedPositions,
  rotaryAngles,
  sinusoidalPositions,
} from './positional.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('position encodings', () => {
  it('sinusoidal encodings alternate sin and cos with geometric frequencies', () => {
    const pe = toFlat(sinusoidalPositions(3, 4, { base: 100 }))
    expect(pe[4 * 2 + 2]).toBeCloseTo(Math.sin(2 / 10), 14)
    expect(pe[4 * 2 + 3]).toBeCloseTo(Math.cos(2 / 10), 14)
    expect(() => sinusoidalPositions(2, 3)).toThrow()
  })

  it('rotary embeddings keep norms and make dot products depend on the offset only', () => {
    for (const interleaved of [false, true]) {
      const q = n(0, [1, 8])
      const k = n(1, [1, 8])
      const rot = (x: Value, p: number): Value => applyRotary(x, rotaryAngles(1, 8, { positions: [p] }), interleaved)
      expect(item(norm(rot(q, 5)))).toBeCloseTo(item(norm(q)), 12)
      const dot = (a: number, b: number): number => item(sum(mul(rot(q, a), rot(k, b))))
      expect(dot(7, 3)).toBeCloseTo(dot(14, 10), 12)
      expect(dot(7, 3)).not.toBeCloseTo(dot(7, 4), 6)
    }
  })

  it('partial rotary leaves the tail unchanged', () => {
    const x = n(2, [3, 6])
    const y = toFlat(applyRotary(x, rotaryAngles(3, 4)))
    const xs = toFlat(x)
    for (const r of [0, 1, 2]) for (const c of [4, 5]) expect(y[r * 6 + c]).toBe(xs[r * 6 + c])
    expect(() => applyRotary(n(3, [2, 2]), rotaryAngles(2, 4))).toThrow()
  })

  it('ALiBi slopes follow the paper for non-powers of two', () => {
    expect(alibiSlopes(8)).toEqual(Array.from({ length: 8 }, (_, i) => 2 ** -(i + 1)))
    const six = alibiSlopes(6)
    expect(six.slice(0, 4)).toEqual(alibiSlopes(4))
    expect(six.slice(4)).toEqual([alibiSlopes(8)[0], alibiSlopes(8)[2]])
    const b = toFlat(alibiBias(1, 1, 3, false))
    expect(b.map((v) => v / (alibiSlopes(1)[0] ?? 1))).toEqual([-2, -1, -0])
  })

  it('rotations and learned positions are differentiable', () => {
    const angles = rotaryAngles(3, 4, { base: 50, offset: 2 })
    expect(checkPrimitive((x) => applyRotary(x, angles), [n(4, [2, 3, 4])]).failures).toEqual([])
    expect(checkPrimitive((x) => applyRotary(x, angles, true), [n(5, [3, 4])]).failures).toEqual([])
    expect(checkPrimitive((t) => learnedPositions(t, 2, 1), [n(6, [4, 3])]).failures).toEqual([])
    expect(() => learnedPositions(n(6, [4, 3]), 4, 1)).toThrow()
  })
})
