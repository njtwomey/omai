import { describe, expect, it } from 'vitest'
import { flattenSvg, flattenSvgPath, normaliseOutline, resampleClosed, stitchStrokes, svgOutline } from './outlines.ts'

describe('outlines', () => {
  it('flattens relative commands, smooth curves and arcs to their end points', () => {
    const [s] = flattenSvgPath('m1 1l2 0c0 1 1 1 1 0s1-1 1 0a1 1 0 0 1 2 0z')
    const last = s?.points.at(-1)
    expect(s?.closed).toBe(true)
    expect(last?.[0]).toBeCloseTo(7)
    expect(last?.[1]).toBeCloseTo(1)
    expect(() => flattenSvg([['blink', {}]])).toThrow()
  })

  it('stitches strokes into one path that visits every stroke', () => {
    const a = {
      points: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ] as [number, number][],
      closed: true,
    }
    const b = {
      points: [
        [3, 0],
        [4, 0],
      ] as [number, number][],
      closed: false,
    }
    const path = stitchStrokes([b, a])
    for (const p of [...a.points, ...b.points]) expect(path).toContainEqual(p)
  })

  it('resamples uniformly by arc length and normalises to unit radius', () => {
    const p = resampleClosed(
      [
        [0, 0],
        [3, 0],
        [3, 1],
        [0, 1],
      ],
      8,
    )
    const gaps = p.map((q, i) => Math.hypot(q[0] - (p[(i + 1) % 8]?.[0] ?? 0), q[1] - (p[(i + 1) % 8]?.[1] ?? 0)))
    for (const g of gaps) expect(g).toBeCloseTo(1, 10)
    const n = normaliseOutline(p)
    expect(Math.max(...n.map(([x, y]) => Math.hypot(x, y)))).toBeCloseTo(1, 12)
    expect(n.reduce((s, q) => s + q[0], 0)).toBeCloseTo(0, 12)
    expect(svgOutline([['circle', { cx: '5', cy: '5', r: '2' }]], 64)).toHaveLength(64)
  })
})
