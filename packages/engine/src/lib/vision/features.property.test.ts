import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../random/index.ts'
import { image } from '../signal/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder } from '../trace/index.ts'
import * as V from './index.ts'

const noise = (seed: number, h: number, w: number, s = 1): number[][] => {
  const e = toFlat(uniform(key(seed), [h * w]))
  return Array.from({ length: h }, (_, i) => Array.from({ length: w }, (_, j) => s * (e[i * w + j] ?? 0)))
}

/** An 8 x 8 window with a dark left half and a light right half, plus noise. */
const edgeWindow = (seed: number): number[][] =>
  noise(seed, 8, 8, 0.2).map((r) => r.map((v, j) => v + (j < 4 ? 0.1 : 0.7)))

/** An 8 x 8 window with a light square in the middle, plus noise. */
const blobWindow = (seed: number): number[][] =>
  noise(seed, 8, 8, 0.1).map((r, i) => r.map((v, j) => v + (i >= 2 && i < 6 && j >= 2 && j < 6 ? 0.8 : 0.3)))

describe('Haar features and cascades', () => {
  it('a window away from the origin equals the same window of the cropped image', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 4 }), fc.integer({ min: 0, max: 4 }), (r, c) => {
        const img = noise(r * 7 + c, 10, 10)
        const crop = img.slice(r).map((row) => row.slice(c))
        const a = toFlat(V.haarLikeFeature(image.integralImage(img), r, c, 5, 4))
        const b = toFlat(V.haarLikeFeature(image.integralImage(crop), 0, 0, 5, 4))
        for (const [k, v] of a.entries()) expect(v).toBeCloseTo(b[k] ?? 0, 10)
      }),
      { numRuns: 10 },
    )
    expect(V.haarFeatures(4, 4).length).toBe(V.HAAR_TYPES.reduce((s, t) => s + V.haarFeatures(4, 4, t).length, 0))
    expect(() => V.haarLikeFeature(image.integralImage(noise(1, 4, 4)), 2, 2, 4, 4)).toThrow()
  })

  it('boosting lowers the training error and a cascade rejects most negatives', () => {
    const pos = Array.from({ length: 20 }, (_, i) => edgeWindow(i))
    const neg = Array.from({ length: 40 }, (_, i) => noise(100 + i, 8, 8))
    const features = V.haarFeatures(8, 8, ['type-2-x', 'type-2-y']).filter((_, i) => i % 7 === 0)
    const tracer = recorder()
    const cascade = V.trainCascade(pos, neg, { stages: [2, 4], features, tracer, policy: policies.minimal })
    expect(tracer.events.some((e) => e.type === 'metric' && e.name === 'trainingError')).toBe(true)
    const held = Array.from({ length: 10 }, (_, i) => edgeWindow(500 + i))
    expect(held.filter((x) => V.cascadeClassify(cascade, x).pass).length).toBeGreaterThanOrEqual(9)
    const negHeld = Array.from({ length: 20 }, (_, i) => noise(900 + i, 8, 8))
    expect(negHeld.filter((x) => V.cascadeClassify(cascade, x).pass).length).toBeLessThanOrEqual(4)
    // Detection finds the edge pasted at twice the size into a noisy scene, at both scales, and nothing else.
    const scene = noise(7, 24, 32, 0.2).map((r) => r.map((v) => v + 0.4))
    for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) (scene[4 + i] as number[])[10 + j] = j < 8 ? 0.15 : 0.85
    const found = V.detectMultiScale(scene, cascade, { scaleFactor: 2, stepFraction: 0.25 })
    expect(new Set(found.boxes.map(([x1 = 0, , x2 = 0]) => x2 - x1))).toEqual(new Set([8, 16]))
    for (const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] of found.boxes) {
      expect(x1 < 18 && x2 > 18).toBe(true)
      expect(y2 > 4 && y1 < 20).toBe(true)
    }
    expect(() => V.cascadeClassify(cascade, noise(1, 6, 6))).toThrow()
  })
})

describe('HOG detection and line segments', () => {
  it('finds the windows that match a template-trained linear detector', () => {
    const hog = { pixelsPerCell: [4, 4], cellsPerBlock: [2, 2] } as const
    const pos = hogMean(
      Array.from({ length: 8 }, (_, i) => blobWindow(i)),
      hog,
    )
    const neg = hogMean(
      Array.from({ length: 8 }, (_, i) => noise(50 + i, 8, 8)),
      hog,
    )
    const weights = pos.map((v, i) => v - (neg[i] ?? 0))
    const bias = -weights.reduce((s, v, i) => s + (v * ((pos[i] ?? 0) + (neg[i] ?? 0))) / 2, 0)
    const scene = noise(3, 16, 32, 0.1).map((r) => r.map((v) => v + 0.3))
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) (scene[6 + i] as number[])[18 + j] = 0.8
    const out = V.hogDetectMultiScale(scene, { window: [8, 8], weights, bias, hog }, { stride: [4, 4], maxLevels: 1 })
    expect(out.boxes[0]).toEqual([16, 4, 24, 12])
    expect(() => V.hogDetectMultiScale(scene, { window: [8, 8], weights: [1], bias: 0, hog })).toThrow()
  })

  it('detects a straight edge at any angle and nothing in noise', () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: Math.PI, noNaN: true }), (a) => {
        const n = 48
        const [nx, ny] = [Math.cos(a), Math.sin(a)]
        const img = Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => ((j - 24) * nx + (i - 24) * ny > 0 ? 0.8 : 0.2)),
        )
        const r = V.lineSegmentDetector(image.gaussianFilter(img, 1))
        expect(r.lines.length).toBeGreaterThanOrEqual(1)
        // The longest segment runs along the edge (perpendicular to the normal) and passes near the centre.
        const longest = r.lines.reduce((b, l) =>
          Math.hypot((l[2] ?? 0) - (l[0] ?? 0), (l[3] ?? 0) - (l[1] ?? 0)) >
          Math.hypot((b[2] ?? 0) - (b[0] ?? 0), (b[3] ?? 0) - (b[1] ?? 0))
            ? l
            : b,
        )
        const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = longest
        const len = Math.hypot(x2 - x1, y2 - y1)
        expect(len).toBeGreaterThan(20)
        expect(Math.abs(((x2 - x1) * nx + (y2 - y1) * ny) / len)).toBeLessThan(0.1)
        for (const v of r.logNfa) expect(v).toBeGreaterThan(0)
      }),
      { numRuns: 8 },
    )
    const flat = toFlat(normal(key(3), [32 * 32])).map((v) => 0.5 + 0.01 * v)
    const quiet = Array.from({ length: 32 }, (_, i) => flat.slice(i * 32, (i + 1) * 32))
    expect(V.lineSegmentDetector(quiet).lines).toEqual([])
  })
})

function hogMean(
  windows: number[][][],
  hog: { readonly pixelsPerCell: readonly [4, 4]; readonly cellsPerBlock: readonly [2, 2] },
): number[] {
  const f = V.hogFeatures(windows, hog)
  return (f[0] ?? []).map((_, k) => f.reduce((s, r) => s + (r[k] ?? 0), 0) / f.length)
}
