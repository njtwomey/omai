import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { matmul, mul, stack, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const coord = fc.double({ min: -3, max: 3, noNaN: true })
const closeAll = (a: readonly number[], b: readonly number[], digits = 12): void => {
  for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? Number.NaN, digits)
}

describe('coordinate systems', () => {
  it('round-trips polar, cylindrical and spherical coordinates', () => {
    fc.assert(
      fc.property(coord, coord, coord, (x, y, z) => {
        if (Math.hypot(x, y) < 1e-3) return
        closeAll(toFlat(F.polarToCartesian(F.cartesianToPolar([x, y]))), [x, y])
        closeAll(toFlat(F.cylindricalToCartesian(F.cartesianToCylindrical([x, y, z]))), [x, y, z])
        const s = toFlat(F.cartesianToSpherical([x, y, z]))
        expect(s[1]).toBeGreaterThanOrEqual(0)
        expect(s[1]).toBeLessThanOrEqual(Math.PI)
        closeAll(toFlat(F.sphericalToCartesian(s)), [x, y, z])
      }),
    )
  })

  it('places geographic points on the sphere', () => {
    const p = toFlat(F.geographicToCartesian([Math.PI / 2, 1.3], 6371))
    closeAll(p, [0, 0, 6371], 9)
    const q = toFlat(F.geographicToCartesian([0.3, 0.4]))
    expect(Math.hypot(...q)).toBeCloseTo(1, 15)
    expect(() => F.cartesianToPolar([1, 2, 3])).toThrow()
  })

  it('homogeneous coordinates turn affine maps into matrix products', () => {
    const m = F.homogeneousMatrix(
      [
        [0, -1],
        [1, 0],
      ],
      [3, 4],
    )
    expect(
      toArray(
        F.applyHomogeneous(m, [
          [1, 0],
          [0, 2],
        ]),
      ),
    ).toEqual([
      [3, 5],
      [1, 4],
    ])
    const composed = matmul(m, m)
    closeAll(toFlat(F.applyHomogeneous(composed, [1, 0])), toFlat(F.applyHomogeneous(m, F.applyHomogeneous(m, [1, 0]))))
    // Scaling a homogeneous point changes nothing.
    closeAll(toFlat(F.fromHomogeneous(mul(F.toHomogeneous([2, 5]), 7))), [2, 5])
    expect(() => F.homogeneousMatrix([[1, 2, 3]])).toThrow()
  })
})

describe('barycentric coordinates', () => {
  it('sum to one, rebuild the point and are positive exactly inside', () => {
    const tet = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    fc.assert(
      fc.property(coord, coord, coord, (x, y, z) => {
        const w = toFlat(F.barycentric([x, y, z], tet))
        expect(w.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
        closeAll(toFlat(F.fromBarycentric(w, tet)), [x, y, z], 11)
        const margin = Math.min(x, y, z, 1 - x - y - z)
        if (Math.abs(margin) > 1e-9) expect(w.every((v) => v > 0)).toBe(margin > 0)
      }),
    )
    expect(
      toArray(
        F.barycentric(
          [[1, 0]],
          [
            [0, 0],
            [2, 0],
            [0, 2],
          ],
        ),
      ),
    ).toEqual([[0.5, 0.5, 0]])
    expect(() =>
      F.barycentric(
        [0, 0],
        [
          [0, 0],
          [1, 0],
        ],
      ),
    ).toThrow()
  })
})

describe('change of basis', () => {
  it('composes and diagonalises', () => {
    const b = [
      [1, 1],
      [1, -1],
    ]
    const c = [
      [2, 0],
      [1, 1],
    ]
    const v = [3, -2]
    const inB = toFlat(F.coordinatesInBasis(v, b))
    const inC = toFlat(F.coordinatesInBasis(v, c))
    closeAll(toFlat(matmul(F.changeOfBasis(b, c), inB)), inC)
    closeAll(toFlat(matmul(b, inB)), v)
    expect(
      toArray(
        F.matrixInBasis(
          [
            [1, 0],
            [0, 1],
          ],
          c,
        ),
      ),
    ).toEqual([
      [1, 0],
      [0, 1],
    ])
  })
})

describe('Jacobian determinants', () => {
  it('match the closed forms and any user map', () => {
    expect(F.coordinateJacobianDeterminant('cylindrical', [3, 0.2, -1])).toBeCloseTo(3, 14)
    expect(F.coordinateJacobianDeterminant('spherical', [2, 0.5, 1])).toBeCloseTo(4 * Math.sin(0.5), 14)
    const shear = (p: Value): Value =>
      matmul(
        tensor([
          [2, 1],
          [0, 3],
        ]),
        p,
      )
    expect(F.coordinateJacobianDeterminant(shear, [1, 1])).toBeCloseTo(6, 14)
    expect(() =>
      F.coordinateJacobianDeterminant((p: Value) => stack([toArray(p) as unknown as number]), [1, 2]),
    ).toThrow()
    expect(F.coordinateJacobian('polar', [1, Math.PI / 2])[0]?.[0]).toBeCloseTo(0, 15)
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    const p2 = tensor([
      [1, 2],
      [-0.5, 0.7],
    ])
    const p3 = tensor([
      [1, 2, -0.4],
      [-0.5, 0.7, 1.1],
    ])
    expectConformant(F.cartesianToPolar, [p2])
    expectConformant(F.polarToCartesian, [p2])
    expectConformant(F.cartesianToCylindrical, [p3])
    expectConformant(F.cylindricalToCartesian, [p3])
    expectConformant(F.cartesianToSpherical, [p3])
    expectConformant(F.sphericalToCartesian, [p3])
    expectConformant((x) => F.geographicToCartesian(x, 2), [p2])
    expectConformant(F.toHomogeneous, [p2])
    expectConformant(F.homogeneousMatrix, [
      tensor([
        [1, 2],
        [0.5, -1],
      ]),
      tensor([0.3, 2]),
    ])
    expectConformant(F.fromHomogeneous, [p3])
    expectConformant(
      (m, x) => F.applyHomogeneous(m, x),
      [
        tensor([
          [1, 0.2, 1],
          [0.1, 2, -1],
          [0.05, 0.1, 1],
        ]),
        p2,
      ],
    )
    expectConformant(F.barycentric, [
      p2,
      tensor([
        [0, 0],
        [2, 0.3],
        [0.2, 1.5],
      ]),
    ])
    expectConformant(F.fromBarycentric, [
      tensor([0.2, 0.3, 0.5]),
      tensor([
        [0, 0],
        [2, 0.3],
        [0.2, 1.5],
      ]),
    ])
    expectConformant(F.coordinatesInBasis, [
      tensor([1, 2]),
      tensor([
        [2, 1],
        [0.5, 3],
      ]),
    ])
    expectConformant(F.changeOfBasis, [
      tensor([
        [1, 0.4],
        [0.2, 2],
      ]),
      tensor([
        [2, 1],
        [0.5, 3],
      ]),
    ])
    expectConformant(F.matrixInBasis, [
      tensor([
        [1, 0.4],
        [0.2, 2],
      ]),
      tensor([
        [2, 1],
        [0.5, 3],
      ]),
    ])
  })
})
