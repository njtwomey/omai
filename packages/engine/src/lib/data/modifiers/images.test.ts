import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { asGenerated, type Generated } from '../generator.ts'
import { type ImageData, images } from '../schema/image.ts'
import { withCutmix, withCutout, withMixup, withRandomCrop, withRandomFlip } from './images.ts'

/** `n` random single-channel images of side `s` with labels and (optionally) boxes and masks. */
function sample(n: number, s: number, withAnnotations = false): Generated<ImageData> {
  const pixels = toFlat(normal(key(1), [n * s * s]))
  const labels = Array.from({ length: n }, (_, i) => i % 3)
  const boxes = withAnnotations ? Array.from({ length: n }, () => [[1, 1, s / 2, s / 2]]) : undefined
  const mask = withAnnotations
    ? fromStorage(
        Int32Array.from({ length: n * s * s }, (_, p) => (p % s < s / 2 ? 1 : 0)),
        [n, s, s],
        'int32',
      )
    : undefined
  return asGenerated(
    images(fromStorage(Float64Array.from(pixels), [n, s, s, 1], 'float64'), {
      labels,
      classes: ['a', 'b', 'c'],
      ...(boxes === undefined ? {} : { boxes, boxLabels: boxes.map(() => [2]) }),
      ...(mask === undefined ? {} : { masks: mask }),
    }),
  ) as Generated<ImageData>
}

const px = (g: Generated<ImageData>, i: number, y: number, x: number): number => {
  const [, h = 0, w = 0] = g.data.images.shape
  return g.data.images.data[(i * h + y) * w + x] ?? 0
}

describe('random crop', () => {
  it('is the identity with no padding at the full size, and shrinks with a smaller window', () => {
    const g = sample(4, 6)
    const same = withRandomCrop(g, { key: key(0) })
    expect(Array.from(same.data.images.data)).toEqual(Array.from(g.data.images.data))
    const small = withRandomCrop(g, { size: 4, key: key(0) })
    expect(small.data.images.shape).toEqual([4, 4, 4, 1])
  })

  it('cuts exactly the window its recorded offsets name, zero outside the original', () => {
    const g = sample(8, 6)
    const out = withRandomCrop(g, { size: 6, padding: 2, key: key(3) })
    const offsets = toFlat(out.truth.latents?.['cropOffsets'] as never)
    for (let i = 0; i < 8; i++)
      for (let y = 0; y < 6; y++)
        for (let x = 0; x < 6; x++) {
          const sy = (offsets[i * 2] ?? 0) + y
          const sx = (offsets[i * 2 + 1] ?? 0) + x
          const inside = sy >= 0 && sy < 6 && sx >= 0 && sx < 6
          expect(px(out, i, y, x)).toBe(inside ? px(g, i, sy, sx) : 0)
        }
    // Offsets range over the padded border, and over several images they are not all equal.
    expect(
      new Set(Array.from({ length: 8 }, (_, i) => `${offsets[i * 2]},${offsets[i * 2 + 1]}`)).size,
    ).toBeGreaterThan(1)
  })

  it('moves boxes and masks with the pixels', () => {
    const g = sample(6, 8, true)
    const out = withRandomCrop(g, { size: 6, padding: 1, key: key(5) })
    const offsets = toFlat(out.truth.latents?.['cropOffsets'] as never)
    out.data.boxes?.forEach((b, i) => {
      for (let k = 0; k < (b.shape[0] ?? 0); k++) {
        const x0 = b.data[k * 4] ?? 0
        expect(x0).toBeGreaterThanOrEqual(0)
        expect(b.data[k * 4 + 2] ?? 0).toBeLessThanOrEqual(6)
        // The box was [1, 1, 4, 4] in the original: its image in the crop is shifted by the offset.
        expect(x0).toBeCloseTo(Math.max(0, 1 - (offsets[i * 2 + 1] ?? 0)), 12)
      }
    })
    expect(out.data.masks?.shape).toEqual([6, 6, 6])
  })

  it('is deterministic in the key and rejects a window larger than the padded image', () => {
    const g = sample(3, 5)
    const a = withRandomCrop(g, { size: 3, padding: 1, key: key(9) })
    const b = withRandomCrop(g, { size: 3, padding: 1, key: key(9) })
    expect(Array.from(a.data.images.data)).toEqual(Array.from(b.data.images.data))
    expect(() => withRandomCrop(g, { size: 8, key: key(0) })).toThrow(/fit/)
  })
})

describe('random flip', () => {
  it('mirrors an image when its recorded flag is set and leaves it otherwise', () => {
    const g = sample(10, 5)
    const out = withRandomFlip(g, { horizontal: 0.5, vertical: 0.5, key: key(2) })
    const fh = toFlat(out.truth.latents?.['flippedHorizontal'] as never)
    const fv = toFlat(out.truth.latents?.['flippedVertical'] as never)
    for (let i = 0; i < 10; i++)
      for (let y = 0; y < 5; y++)
        for (let x = 0; x < 5; x++) expect(px(out, i, y, x)).toBe(px(g, i, fv[i] ? 4 - y : y, fh[i] ? 4 - x : x))
    expect(fh.some((v) => v === 1) && fh.some((v) => v === 0)).toBe(true)
  })

  it('is an involution under the same key and keeps box areas', () => {
    const g = sample(6, 8, true)
    const once = withRandomFlip(g, { horizontal: 1, vertical: 1, key: key(1) })
    const twice = withRandomFlip(once, { horizontal: 1, vertical: 1, key: key(1) })
    expect(Array.from(twice.data.images.data)).toEqual(Array.from(g.data.images.data))
    expect(Array.from(twice.data.masks?.data ?? [])).toEqual(Array.from(g.data.masks?.data ?? []))
    once.data.boxes?.forEach((b, i) => {
      const o = g.data.boxes?.[i]
      expect((b.data[2] ?? 0) - (b.data[0] ?? 0)).toBeCloseTo((o?.data[2] ?? 0) - (o?.data[0] ?? 0), 12)
      expect(b.data[0]).toBe(8 - (o?.data[2] ?? 0))
    })
  })
})

describe('cutout', () => {
  it('blanks exactly the recorded square and nothing else', () => {
    const g = sample(6, 8)
    const out = withCutout(g, { size: 3, fill: 7, key: key(4) })
    const holes = toFlat(out.truth.latents?.['cutoutBoxes'] as never)
    for (let i = 0; i < 6; i++) {
      const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = holes.slice(i * 4, i * 4 + 4)
      expect((x1 - x0) * (y1 - y0)).toBeLessThanOrEqual(9)
      for (let y = 0; y < 8; y++)
        for (let x = 0; x < 8; x++) {
          const inside = y >= y0 && y < y1 && x >= x0 && x < x1
          expect(px(out, i, y, x)).toBe(inside ? 7 : px(g, i, y, x))
        }
    }
    expect(Array.from(out.data.labels?.data ?? [])).toEqual(Array.from(g.data.labels?.data ?? []))
  })
})

describe('mixup and cutmix', () => {
  it('mixup is the convex blend its recorded weights and partners say, with matching soft labels', () => {
    const g = sample(9, 4)
    const out = withMixup(g, { alpha: 0.4, key: key(6) })
    const lambda = toFlat(out.truth.latents?.['lambda'] as never)
    const partner = toFlat(out.truth.latents?.['partner'] as never)
    const soft = toFlat(out.truth.latents?.['softLabels'] as never)
    expect(new Set(partner).size).toBe(9)
    for (let i = 0; i < 9; i++) {
      for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++)
          expect(px(out, i, y, x)).toBeCloseTo(
            (lambda[i] ?? 0) * px(g, i, y, x) + (1 - (lambda[i] ?? 0)) * px(g, partner[i] ?? 0, y, x),
            12,
          )
      const row = soft.slice(i * 3, i * 3 + 3)
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      const own = i % 3
      const other = (partner[i] ?? 0) % 3
      expect(row[own] ?? 0).toBeGreaterThanOrEqual(own === other ? 1 - 1e-12 : (lambda[i] ?? 0) - 1e-12)
      expect(out.data.labels?.data[i]).toBe((lambda[i] ?? 0) >= 0.5 ? own : other)
    }
    expect(out.data.boxes).toBeUndefined()
  })

  it('cutmix takes the recorded rectangle from the partner and the rest from the image, with labels mixed by area', () => {
    const g = sample(9, 8)
    const out = withCutmix(g, { alpha: 1, key: key(7) })
    const lambda = toFlat(out.truth.latents?.['lambda'] as never)
    const partner = toFlat(out.truth.latents?.['partner'] as never)
    const cuts = toFlat(out.truth.latents?.['cutBoxes'] as never)
    const soft = toFlat(out.truth.latents?.['softLabels'] as never)
    for (let i = 0; i < 9; i++) {
      const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = cuts.slice(i * 4, i * 4 + 4)
      expect(lambda[i]).toBeCloseTo(1 - ((x1 - x0) * (y1 - y0)) / 64, 12)
      for (let y = 0; y < 8; y++)
        for (let x = 0; x < 8; x++) {
          const inside = y >= y0 && y < y1 && x >= x0 && x < x1
          expect(px(out, i, y, x)).toBe(inside ? px(g, partner[i] ?? 0, y, x) : px(g, i, y, x))
        }
      const own = i % 3
      const other = (partner[i] ?? 0) % 3
      if (own !== other) expect(soft[i * 3 + own]).toBeCloseTo(lambda[i] ?? 0, 12)
    }
  })

  it('keeps pixels within the range of the inputs, whatever the concentration', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 40 }), fc.double({ min: 0.05, max: 5, noNaN: true }), (seed, alpha) => {
        const g = sample(6, 5)
        const lo = Math.min(...g.data.images.data)
        const hi = Math.max(...g.data.images.data)
        for (const out of [withMixup(g, { alpha, key: key(seed) }), withCutmix(g, { alpha, key: key(seed) })]) {
          for (const v of out.data.images.data) {
            expect(v).toBeGreaterThanOrEqual(lo - 1e-12)
            expect(v).toBeLessThanOrEqual(hi + 1e-12)
          }
        }
      }),
      { numRuns: 25 },
    )
  })

  it('refuses data that is not images', () => {
    expect(() => withMixup({ data: { kind: 'table' } as never, truth: {} }, { key: key(0) })).toThrow(/image/)
  })
})
