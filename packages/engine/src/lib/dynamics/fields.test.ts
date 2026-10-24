import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { mul, neg, sin, sliceAxis, stack, sub, sum, toArray, toFlat, type Value } from '../tensor/index.ts'
import {
  classifyFixedPoint,
  fieldCurl,
  fieldDivergence,
  fieldJacobian,
  fixedPoints,
  flowMap,
  invariantManifold,
  nullclines,
  streamline,
} from './fields.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))
// The damped pendulum: equilibria at (k pi, 0), stable spirals at even k, saddles at odd k.
const pendulum = (_t: number, y: Value): Value => stack([at(y, 1), sub(neg(sin(at(y, 0))), mul(at(y, 1), 0.5))])

describe('vector fields', () => {
  it('Jacobian, divergence and curl', () => {
    expect(toArray(fieldJacobian(pendulum, [0, 0]))).toEqual([
      [0, 1],
      [-1, -0.5],
    ])
    expect(fieldDivergence(pendulum, [0.3, 0.1])).toBeCloseTo(-0.5, 14)
    expect(fieldCurl(pendulum, [0, 0])).toEqual([-2])
    const rot = (_t: number, y: Value): Value => stack([neg(at(y, 1)), at(y, 0), mul(at(y, 2), 0)])
    expect(fieldCurl(rot, [1, 2, 3])).toEqual([0, 0, 2])
    expect(() => fieldCurl((_t, y) => y, [1, 2, 3, 4])).toThrow()
  })

  it('classifies every kind of equilibrium', () => {
    const cases: [number[][], string][] = [
      [
        [
          [-1, 0],
          [0, -2],
        ],
        'stable-node',
      ],
      [
        [
          [1, 0],
          [0, 2],
        ],
        'unstable-node',
      ],
      [
        [
          [-0.1, 1],
          [-1, -0.1],
        ],
        'stable-spiral',
      ],
      [
        [
          [0.1, 1],
          [-1, 0.1],
        ],
        'unstable-spiral',
      ],
      [
        [
          [1, 0],
          [0, -1],
        ],
        'saddle',
      ],
      [
        [
          [0, 1],
          [-1, 0],
        ],
        'center',
      ],
      [
        [
          [0, 0],
          [0, -1],
        ],
        'degenerate',
      ],
    ]
    for (const [j, type] of cases) expect(classifyFixedPoint(j).type).toBe(type)
  })

  it('finds the pendulum equilibria from random seeds', () => {
    const found = fixedPoints(pendulum, {
      box: [
        [-4, 4],
        [-1, 1],
      ],
      count: 30,
      key: key(1),
    })
    const xs = found
      .map((p) => Math.round(((p.point.data[0] ?? 0) / Math.PI) * 1e6) / 1e6)
      .filter((k) => Math.abs(k) < 1.5)
      .sort((a, b) => a - b)
    expect(xs).toEqual([-1, 0, 1])
    const zero = found.find((p) => Math.abs(p.point.data[0] ?? 1) < 1e-8)
    expect(zero?.type).toBe('stable-spiral')
    expect(found.find((p) => (p.point.data[0] ?? 0) > 1)?.type).toBe('saddle')
    expect(() => fixedPoints(pendulum, {})).toThrow()
  })

  it('nullclines, streamlines, flow maps and manifolds', () => {
    const lines = nullclines(pendulum, [-1, 1], [-1, 1], 41)
    // The first nullcline is y = 0.
    for (const line of lines[0] ?? []) for (const [, y] of line) expect(Math.abs(y)).toBeLessThan(1e-9)
    const s = streamline(pendulum, [1, 0], { tEnd: 30, points: 50, both: true })
    expect(s.points.shape).toEqual([99, 2])
    const end = toFlat(s.points).slice(-2)
    expect(Math.hypot(end[0] ?? 1, end[1] ?? 1)).toBeLessThan(1e-2)
    const phi = toArray(flowMap((_t, y) => mul(y, 2), [[1], [3]], 0.5)) as number[][]
    expect(phi[1]?.[0]).toBeCloseTo(3 * Math.E, 7)
    const unstable = invariantManifold(pendulum, [Math.PI, 0], 'unstable', { tEnd: 20 })
    // Both branches of the saddle's unstable manifold fall into the neighbouring stable spirals.
    const ends = unstable.map((b) => toFlat(b.points).slice(-2)[0] ?? 0).sort((a, b) => a - b)
    expect(Math.abs((ends[0] ?? 0) - 0)).toBeLessThan(0.1)
    expect(Math.abs((ends[1] ?? 0) - 2 * Math.PI)).toBeLessThan(0.1)
    expect(() => invariantManifold(pendulum, [0, 0], 'unstable')).toThrow()
  })
})
