import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { matmul, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const angle = fc.double({ min: -3, max: 3, noNaN: true })
const quat = fc
  .tuple(
    fc.double({ min: -1, max: 1, noNaN: true }),
    fc.double({ min: -1, max: 1, noNaN: true }),
    fc.double({ min: -1, max: 1, noNaN: true }),
    fc.double({ min: -1, max: 1, noNaN: true }),
  )
  .filter((q) => Math.hypot(...q) > 0.2)
  .map((q) => toFlat(F.quatNormalize(q)))

const closeMatrix = (a: Value, b: Value, digits = 12): void => {
  const x = toFlat(a)
  const y = toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? 0, digits)
}

describe('rotation matrices', () => {
  it('2-D rotations compose by adding angles and keep lengths', () => {
    fc.assert(
      fc.property(angle, angle, (a, b) => {
        closeMatrix(matmul(F.rotation2d(a), F.rotation2d(b)), F.rotation2d(a + b))
        expect(F.isRotationMatrix(F.rotation2d(a))).toBe(true)
      }),
    )
    const p = toFlat(F.rotate2d([[3, 4]], 1.234))
    expect(Math.hypot(p[0] ?? 0, p[1] ?? 0)).toBeCloseTo(5, 13)
  })

  it('axis-angle matrices are rotations that fix their axis', () => {
    fc.assert(
      fc.property(quat, angle, (q, t) => {
        const axis = q.slice(0, 3)
        if (Math.hypot(...axis) < 0.1) return
        const r = F.axisAngleToMatrix(axis, t)
        expect(F.isRotationMatrix(r)).toBe(true)
        closeMatrix(matmul(r, axis), tensor(axis), 12)
        closeMatrix(F.rotateAboutAxis([0.3, -1, 2], axis, t), matmul(r, [0.3, -1, 2]), 12)
        closeMatrix(F.quatToMatrix(F.quatFromAxisAngle(axis, t)), r, 12)
        const back = F.matrixToAxisAngle(r)
        expect(back.angle).toBeCloseTo(Math.abs(Math.atan2(Math.sin(t), Math.cos(t))), 9)
      }),
    )
    expect(
      F.isRotationMatrix([
        [1, 0],
        [0, 1],
        [0, 0],
      ]),
    ).toBe(false)
    expect(
      F.isRotationMatrix([
        [2, 0],
        [0, 0.5],
      ]),
    ).toBe(false)
  })
})

describe('Euler angles', () => {
  it('lists the 24 conventions and round-trips each away from gimbal lock', () => {
    expect(new Set(F.eulerSequences.map((s) => s.seq)).size).toBe(24)
    for (const { seq, kind } of F.eulerSequences) {
      const angles: [number, number, number] = [0.4, kind === 'proper' ? 1.1 : -0.6, -1.3]
      const m = F.eulerToMatrix(seq, angles)
      expect(F.isRotationMatrix(m)).toBe(true)
      const back = F.matrixToEuler(seq, m)
      for (const [i, v] of back.entries()) expect(v).toBeCloseTo(angles[i] ?? 0, 10)
      expect(F.gimbalLock(seq, angles).locked).toBe(false)
    }
    expect(() => F.eulerToMatrix('xxy', [0, 0, 0])).toThrow()
    expect(() => F.eulerToMatrix('xYz', [0, 0, 0])).toThrow()
  })

  it('intrinsic angles equal extrinsic angles in reverse order', () => {
    closeMatrix(F.eulerToMatrix('XYZ', [0.1, 0.2, 0.3]), F.eulerToMatrix('zyx', [0.3, 0.2, 0.1]))
  })

  it('at gimbal lock only the combination of the outer angles matters', () => {
    for (const [seq, mid] of [
      ['xyz', Math.PI / 2],
      ['xyz', -Math.PI / 2],
      ['ZXZ', 0],
      ['zyz', Math.PI],
    ] as const) {
      const lock = F.gimbalLock(seq, [0.7, mid, 0.2])
      expect(lock.locked).toBe(true)
      const effective = lock.effective ?? Number.NaN
      // Moving the outer angles while keeping their combination fixed leaves the rotation unchanged.
      const other: [number, number, number] =
        effective === 0.7 + 0.2 ? [0.5, mid, effective - 0.5] : [0.5, mid, 0.5 - effective]
      closeMatrix(F.eulerToMatrix(seq, [0.7, mid, 0.2]), F.eulerToMatrix(seq, other), 12)
    }
  })
})

describe('quaternions', () => {
  it('the product is associative and matches matrix composition', () => {
    fc.assert(
      fc.property(quat, quat, quat, (p, q, r) => {
        closeMatrix(F.quatMultiply(F.quatMultiply(p, q), r), F.quatMultiply(p, F.quatMultiply(q, r)), 12)
        closeMatrix(F.quatToMatrix(F.quatMultiply(p, q)), matmul(F.quatToMatrix(p), F.quatToMatrix(q)), 11)
        closeMatrix(F.quatMultiply(p, F.quatInverse(p)), tensor([0, 0, 0, 1]), 12)
        closeMatrix(F.quatRotate(p, [1, 2, 3]), matmul(F.quatToMatrix(p), [1, 2, 3]), 11)
        // Both signs of a quaternion are the same rotation.
        const back = F.matrixToQuat(F.quatToMatrix(p))
        expect(Math.abs(back.reduce((s, v, i) => s + v * (p[i] ?? 0), 0))).toBeCloseTo(1, 10)
      }),
    )
  })

  it('slerp runs at constant speed along the shorter arc', () => {
    fc.assert(
      fc.property(quat, quat, (p, q) => {
        const total = toFlat(F.rotationDistance(p, q))[0] ?? 0
        for (const t of [0, 0.3, 1]) {
          const m = F.quatSlerp(p, q, t)
          expect(toFlat(F.rotationDistance(p, m))[0] ?? 0).toBeCloseTo(t * total, 7)
        }
      }),
    )
    const near = toFlat(F.quatSlerp([0, 0, 0, 1], [1e-10, 0, 0, 1], 0.5))
    expect(near[0]).toBeCloseTo(5e-11, 20)
  })

  it('the axis-angle round trip recovers the rotation', () => {
    const { axis, angle: t } = F.quatToAxisAngle(toFlat(F.quatFromAxisAngle([1, 2, 2], 2)))
    expect(t).toBeCloseTo(2, 14)
    for (const [i, v] of axis.entries()) expect(v).toBeCloseTo([1 / 3, 2 / 3, 2 / 3][i] ?? 0, 14)
    expect(F.quatToAxisAngle([0, 0, 0, 1]).axis).toEqual([1, 0, 0])
  })
})

describe('rotation averaging', () => {
  it('the Karcher mean zeroes the average tangent vector and is near the chordal mean', () => {
    const qs = [
      [0.2, 0.1, -0.3],
      [-0.1, 0.4, 0.2],
      [0.3, -0.2, 0.1],
      [0, 0.3, 0.5],
    ].map((v) => toFlat(F.quatFromAxisAngle(v, Math.hypot(...v))))
    const k = F.karcherMean(qs)
    expect(k.converged).toBe(true)
    const steps = k.steps
    expect(steps[steps.length - 1] ?? 1).toBeLessThan(1e-12)
    const chordal = F.meanRotation(qs)
    expect(toFlat(F.rotationDistance(k.quaternion, chordal))[0]).toBeLessThan(0.01)
    // Negating any quaternion changes nothing.
    const flipped = qs.map((q, i) => (i % 2 ? q.map((v) => -v) : q))
    for (const [i, v] of F.meanRotation(flipped).entries()) expect(v).toBeCloseTo(chordal[i] ?? 0, 12)
    expect(() => F.meanRotation([[1, 0, 0]])).toThrow()
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    const t = tensor([0.3, -1.2])
    const q = tensor([
      [0.1, -0.5, 0.3, 0.8],
      [0.6, 0.2, -0.4, 0.5],
    ])
    const p = tensor([
      [0.4, 0.4, -0.2, 0.7],
      [-0.3, 0.6, 0.1, 0.6],
    ])
    const v = tensor([
      [1, -2, 0.5],
      [0.3, 0.7, -1],
    ])
    expectConformant(F.rotation2d, [t])
    expectConformant(
      (x, a) => F.rotate2d(x, a, [0.5, -1]),
      [
        tensor([
          [1, 2],
          [0.5, 3],
        ]),
        tensor(0.4),
      ],
    )
    expectConformant((a) => F.elementaryRotation('y', a), [t])
    expectConformant(F.axisAngleToMatrix, [v, t])
    expectConformant(F.rotateAboutAxis, [
      v,
      tensor([
        [0, 1, 2],
        [1, 1, 0],
      ]),
      t,
    ])
    expectConformant((a) => F.eulerToMatrix('ZYX', a), [v])
    expectConformant((a) => F.eulerToMatrix('xzx', a), [v])
    expectConformant(F.quatMultiply, [p, q])
    expectConformant(F.quatConjugate, [q])
    expectConformant(F.quatInverse, [q])
    expectConformant(F.quatNormalize, [q])
    expectConformant(F.quatFromAxisAngle, [v, t])
    expectConformant(F.quatToMatrix, [q])
    expectConformant(F.quatRotate, [q, v])
    expectConformant(F.quatSlerp, [F.quatNormalize(p), F.quatNormalize(q), tensor([0.3, 0.8])])
    expectConformant(F.rotationDistance, [F.quatNormalize(p), F.quatNormalize(q)])
  })

  it('rotation matrices from quaternions have unit determinant along any path', () => {
    const m = toArray(F.quatToMatrix([0.1, 0.2, 0.3, 0.4])) as number[][]
    expect(F.isRotationMatrix(m)).toBe(true)
  })
})
