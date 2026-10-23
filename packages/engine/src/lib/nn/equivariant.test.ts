import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { key, normal } from '../random/index.ts'
import { mean, toArray, type Value } from '../tensor/index.ts'
import {
  equivarianceError,
  invarianceError,
  OrientationPool,
  orientation,
  orientationPool,
  P4GroupConv,
  P4LiftingConv,
  p4GroupConv,
  p4LiftingConv,
  p4Transform,
  rotate90,
} from './equivariant.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('p4 group convolutions', () => {
  it('rotate90 composes to the identity after four turns', () => {
    const x = n(0, [2, 3, 3])
    expect(toArray(rotate90(x, 4))).toEqual(toArray(x))
    expect(toArray(rotate90(rotate90(x, 3), 1))).toEqual(toArray(x))
    expect(toArray(rotate90(x, -1))).toEqual(toArray(rotate90(x, 3)))
    expect(() => rotate90([1, 2])).toThrow()
  })

  it('lifting convolution: rotating the input rotates and shifts the output', () => {
    const w = n(1, [2, 1, 3, 3])
    for (const padding of [0, 1]) {
      const f = (x: Value): Value => p4LiftingConv(x, w, { padding, bias: [0.3, -0.2] })
      expect(
        equivarianceError(
          f,
          n(2, [1, 1, 6, 6]),
          (x) => rotate90(x),
          (y) => p4Transform(y),
        ),
      ).toBeLessThan(1e-12)
    }
  })

  it('group convolution commutes with the p4 action, for every rotation', () => {
    const w = n(3, [3, 2, 4, 3, 3])
    const f = (x: Value): Value => p4GroupConv(x, w, { padding: 1 })
    for (const k of [1, 2, 3]) {
      expect(
        equivarianceError(
          f,
          n(4, [1, 2, 4, 5, 5]),
          (x) => p4Transform(x, k),
          (y) => p4Transform(y, k),
        ),
      ).toBeLessThan(1e-12)
    }
  })

  it('orientation and global pooling make a rotation-invariant network', () => {
    const lift = new P4LiftingConv(1, 2, 3, { padding: 1, key: key(5) })
    const group = new P4GroupConv(2, 2, 3, { padding: 1, key: key(6) })
    const pool = new OrientationPool('mean')
    const net = (x: Value): Value => mean(pool.forward(group.forward(lift.forward(x))), [2, 3])
    expect(invarianceError(net, n(7, [2, 1, 6, 6]), (x) => rotate90(x))).toBeLessThan(1e-12)
    // Orientation pooling alone is equivariant to plain rotations.
    const y = lift.forward(n(8, [1, 1, 4, 4]))
    expect(
      equivarianceError(
        (v) => orientationPool(v),
        y,
        (v) => p4Transform(v),
        (v) => rotate90(v),
      ),
    ).toBeLessThan(1e-12)
    expect(lift.forward(n(9, [1, 1, 6, 6])).shape).toEqual(lift.outputShape([1, 1, 6, 6]))
    expect(new P4GroupConv(2, 3, 3, { stride: 2 }).outputShape([1, 2, 4, 7, 7])).toEqual([1, 3, 4, 3, 3])
    expect(orientation(y, 1).shape).toEqual([1, 2, 4, 4])
    expect(pool.outputShape([1, 2, 4, 4, 4])).toEqual([1, 2, 4, 4])
  })

  it('pass the gradient conformance check', () => {
    expect(checkPrimitive((x, w) => p4LiftingConv(x, w), [n(10, [1, 1, 4, 4]), n(11, [1, 1, 2, 2])]).failures).toEqual(
      [],
    )
    expect(
      checkPrimitive((x, w) => p4GroupConv(x, w, { padding: 1 }), [n(12, [1, 1, 4, 3, 3]), n(13, [1, 1, 4, 2, 2])])
        .failures,
    ).toEqual([])
    expect(() => p4GroupConv(n(0, [1, 1, 3, 3, 3]), n(1, [1, 1, 4, 2, 2]))).toThrow()
  })
})
