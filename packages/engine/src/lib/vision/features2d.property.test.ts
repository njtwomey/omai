import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const blob = (cx: number, cy: number, s: number, n = 40): number[][] =>
  Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Math.exp(-((i - cy) ** 2 + (j - cx) ** 2) / (2 * s * s))),
  )

const texture = (dx = 0, n = 48): number[][] =>
  Array.from({ length: n }, (_, i) =>
    Array.from(
      { length: n },
      (_, j) => Math.sin(i * i * 0.05 + (j - dx) * 0.9) * Math.cos((j - dx) ** 2 * 0.03 - i * 0.4) * 0.5 + 0.5,
    ),
  )

describe('scale-space keypoints', () => {
  it('finds a blob at its centre and scale, wherever it is', () => {
    fc.assert(
      fc.property(fc.integer({ min: 14, max: 26 }), fc.integer({ min: 14, max: 26 }), (cx, cy) => {
        const k = V.detectSift(blob(cx, cy, 3))[0]
        expect(k).toBeDefined()
        expect(Math.abs((k?.x ?? 0) - cx)).toBeLessThan(0.5)
        expect(Math.abs((k?.y ?? 0) - cy)).toBeLessThan(0.5)
        // A Gaussian blob of sigma s responds most at scale s sqrt(2): size = 2 sigma.
        expect(k?.size ?? 0).toBeGreaterThan(4)
        expect(k?.size ?? 0).toBeLessThan(14)
      }),
      { numRuns: 6 },
    )
    const ss = V.buildScaleSpace(blob(20, 20, 3))
    expect(ss.gaussians[0]?.length).toBe(6)
    expect(ss.dogs[1]?.[0]?.h).toBe(20)
  })

  it('describes keypoints with unit, clipped SIFT vectors and stable binary strings', () => {
    const img = texture()
    const kps: V.Keypoint[] = [
      { x: 24, y: 24, size: 6, angle: 0, response: 1, octave: 0 },
      { x: 20, y: 28, size: 4, angle: 120, response: 1, octave: 0 },
    ]
    for (const d of V.siftDescriptors(img, kps)) {
      expect(Math.hypot(...d)).toBeCloseTo(1, 10)
      expect(Math.max(...d)).toBeLessThanOrEqual(0.2 / 0.2 + 1e-12)
    }
    const a = V.briefDescriptors(img, kps, { key: key(3) })
    const b = V.briefDescriptors(img, kps, { key: key(3) })
    expect(a).toEqual(b)
    expect(
      V.descriptorDistance(
        a[0] as Uint8Array,
        V.briefDescriptors(img, kps, { key: key(4) })[0] as Uint8Array,
        'hamming',
      ),
    ).toBeGreaterThan(0)
    const o = V.orb(img, { nLevels: 2, nFeatures: 20 })
    expect(o.keypoints.length).toBe(o.descriptors.length)
    expect(o.keypoints.length).toBeGreaterThan(0)
    for (const d of o.descriptors) expect(d.length).toBe(32)
  })

  it('matches shifted features to themselves', () => {
    const a = texture(0)
    const b = texture(2)
    const kps = V.fast(a, { threshold: 30 / 255 }).filter((k) => k.x > 16 && k.x < 30 && k.y > 16 && k.y < 30)
    const moved = kps.map((k) => ({ ...k, x: k.x + 2 }))
    const da = V.siftDescriptors(
      a,
      kps.map((k) => ({ ...k, size: 6, angle: 0 })),
    )
    const db = V.siftDescriptors(
      b,
      moved.map((k) => ({ ...k, size: 6, angle: 0 })),
    )
    const m = V.matchDescriptors(da, db, { crossCheck: true })
    expect(m.length).toBeGreaterThan(0)
    for (const x of m) expect(x.trainIdx).toBe(x.queryIdx)
    expect(V.ratioTestMatch(da, db, { ratio: 0.9 }).every((x) => x.trainIdx === x.queryIdx)).toBe(true)
  })
})

describe('drawing', () => {
  it('draws symmetric circles and fills polygons by their area', () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 7 }), (r) => {
        const img = Array.from({ length: 17 }, () => new Array<number>(17).fill(0))
        const c = toFlat(V.circle(img, [8, 8], r, 1)) as number[]
        for (let y = 0; y < 17; y++)
          for (let x = 0; x < 17; x++) {
            expect(c[y * 17 + x]).toBe(c[y * 17 + (16 - x)])
            expect(c[y * 17 + x]).toBe(c[x * 17 + y])
          }
        const disc = toFlat(V.circle(img, [8, 8], r, 1, { thickness: -1 })).reduce((a, b) => a + b, 0)
        expect(Math.abs(disc - Math.PI * r * r)).toBeLessThan(2 * Math.PI * r + 2)
        const square = toFlat(
          V.fillPoly(
            img,
            [
              [
                [2, 2],
                [2 + r, 2],
                [2 + r, 2 + r],
                [2, 2 + r],
              ],
            ],
            1,
          ),
        ).reduce((a, b) => a + b, 0)
        expect(square).toBe((r + 1) ** 2)
      }),
      { numRuns: RUNS },
    )
  })

  it('anti-aliases, draws thick lines, ellipses and composite views in colour', () => {
    const img = Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => [0, 0, 0]))
    const aa = toFlat(V.line(img, [1, 1], [8, 5], [1, 0, 0], { lineType: 'aa' }))
    expect(aa.some((v) => v > 0 && v < 1)).toBe(true)
    const thick = toFlat(V.line(img, [1, 5], [8, 5], [0, 1, 0], { thickness: 3 })).reduce((a, b) => a + b, 0)
    expect(thick).toBeGreaterThan(8 * 2)
    expect(V.ellipse(img, [5, 5], [3, 2], 45, [0, 0, 1], { thickness: -1 }).shape).toEqual([10, 10, 3])
    const contours = V.findContours(
      Array.from({ length: 10 }, (_, i) =>
        Array.from({ length: 10 }, (_, j) => (i > 2 && i < 7 && j > 2 && j < 7 ? 1 : 0)),
      ),
    ).contours
    expect(toFlat(V.drawContours(img, contours, -1, [1, 1, 1], { thickness: -1 })).reduce((a, b) => a + b, 0)).toBe(48)
    const k = [{ x: 5, y: 5, size: 4, angle: 90, response: 1, octave: 0 }]
    expect(V.drawKeypoints(img, k).shape).toEqual([10, 10, 3])
    expect(
      V.drawMatches(
        img,
        k,
        Array.from({ length: 8 }, () => new Array<number>(6).fill(0)),
        k,
        [{ queryIdx: 0, trainIdx: 0, distance: 0 }],
      ).shape,
    ).toEqual([10, 16, 3])
  })
})
