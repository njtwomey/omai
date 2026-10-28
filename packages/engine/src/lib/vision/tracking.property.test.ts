import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import * as V from './index.ts'

const texture = (dx: number, dy: number, n = 32): number[][] =>
  Array.from({ length: n }, (_, i) =>
    Array.from(
      { length: n },
      (_, j) => Math.sin((j - dx) / 3) * Math.cos((i - dy) / 4) + 0.3 * Math.sin((i - dy + j - dx) / 5),
    ),
  )

describe('SORT', () => {
  it('keeps one identity per object as objects move and cross the field', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -2, max: 2, noNaN: true }),
        fc.double({ min: -2, max: 2, noNaN: true }),
        (vx, vy) => {
          const frames = Array.from({ length: 8 }, (_, t) => [
            [10 + vx * t, 10 + vy * t, 30 + vx * t, 30 + vy * t],
            [80 - 2 * t, 60, 100 - 2 * t, 90],
          ])
          const out = V.sortTrack(frames)
          expect(out.length).toBe(8)
          for (const f of out) expect(f.map((o) => o.id).sort()).toEqual([1, 2])
          // Reported boxes follow the detections.
          const last = out[7]?.find((o) => o.id === 1)?.box ?? [0, 0, 0, 0]
          expect(Math.abs(last[0] - (10 + vx * 7))).toBeLessThan(1.5)
        },
      ),
      { numRuns: 10 },
    )
  })

  it('ends tracks without detections and starts new ones', () => {
    const frames = [[[0, 0, 10, 10]], [[0, 0, 10, 10]], [], [], [[50, 50, 60, 60]], [[51, 50, 61, 60]]]
    const out = V.sortTrack(frames, { maxAge: 1, minHits: 1 })
    // As the reference SORT, a new track is reported from its second hit on.
    expect(out[4]).toEqual([])
    expect(out[5]?.map((o) => o.id)).toEqual([2])
    const tracer = recorder()
    runAlgorithm(V.sortAlgorithm(frames), { maxSteps: 5, tracer, policy: policies.minimal })
    expect(tracer.events.some((e) => e.type === 'metric' && e.name === 'tracks')).toBe(true)
  })
})

describe('point and window tracking', () => {
  it('tracks translated texture with pyramidal Lucas-Kanade', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -3, max: 3, noNaN: true }),
        fc.double({ min: -3, max: 3, noNaN: true }),
        (dx, dy) => {
          const r = V.calcOpticalFlowPyrLK(texture(0, 0), texture(dx, dy), [
            [16, 16],
            [12, 18],
          ])
          for (const [i, [x, y]] of r.points.entries()) {
            const [x0, y0] = i === 0 ? [16, 16] : [12, 18]
            expect(r.status[i]).toBe(1)
            expect(Math.abs(x - x0 - dx)).toBeLessThan(0.15)
            expect(Math.abs(y - y0 - dy)).toBeLessThan(0.15)
          }
        },
      ),
      { numRuns: 10 },
    )
    const flat = Array.from({ length: 20 }, () => new Array<number>(20).fill(0.5))
    expect(V.calcOpticalFlowPyrLK(flat, flat, [[10, 10]]).status).toEqual([0])
    const s = runAlgorithm(V.kltAlgorithm([texture(0, 0), texture(1, 0), texture(2, 0)], [[16, 16]]), {
      maxSteps: 5,
    }).state
    expect(s.points[0]?.[0]).toBeCloseTo(18, 0)
  })

  it('follows a moving blob with mean shift', () => {
    const blob = (cx: number): number[][] =>
      Array.from({ length: 30 }, (_, i) =>
        Array.from({ length: 40 }, (_, j) => Math.exp(-((i - 15) ** 2 + (j - cx) ** 2) / 8)),
      )
    let w: V.Rect = [8, 11, 8, 8]
    for (const cx of [12, 14, 16, 18, 20]) w = V.meanShift(blob(cx), w).window
    expect(Math.abs(w[0] + w[2] / 2 - 20)).toBeLessThanOrEqual(1)
    expect(() => V.meanShift(blob(10), [0, 0, 0, 5])).toThrow()
  })
})

describe('background subtraction', () => {
  it('flags only what moved, by both models', () => {
    const bg = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => 0.3 + 0.01 * ((i * j) % 3)))
    const fg = bg.map((r, i) => r.map((v, j) => (i > 2 && i < 5 && j > 2 && j < 5 ? 0.9 : v)))
    const frames = [...new Array(12).fill(bg), fg]
    for (const method of ['mixture', 'running-average'] as const) {
      const masks = V.subtractBackground(frames, { method })
      const last = toFlat(masks[12] as never)
      expect(last.reduce((a, b) => a + b, 0)).toBe(4)
      expect(toFlat(masks[5] as never).reduce((a, b) => a + b, 0)).toBe(0)
    }
    expect(V.subtractBackground([])).toEqual([])
  })
})
