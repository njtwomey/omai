import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { boxIou } from '../metrics/index.ts'
import * as V from './index.ts'

const box = fc
  .tuple(
    fc.double({ min: 0, max: 40, noNaN: true }),
    fc.double({ min: 0, max: 40, noNaN: true }),
    fc.double({ min: 1, max: 20, noNaN: true }),
    fc.double({ min: 1, max: 20, noNaN: true }),
  )
  .map(([x, y, w, h]): V.Box => [x, y, x + w, y + h])

const scored = fc.array(fc.tuple(box, fc.double({ min: 0, max: 1, noNaN: true })), { minLength: 1, maxLength: 20 })

describe('boxes', () => {
  it('NMS keeps the best box and no two kept boxes overlap beyond the threshold', () => {
    fc.assert(
      fc.property(scored, fc.double({ min: 0.1, max: 0.9, noNaN: true }), (items, thr) => {
        const boxes = items.map(([b]) => b)
        const scores = items.map(([, s]) => s)
        const keep = V.nms(boxes, scores, thr)
        expect(scores[keep[0] ?? -1]).toBe(Math.max(...scores))
        const overlap = boxIou(boxes, boxes)
        const iou = (i: number, j: number): number => overlap.data[i * boxes.length + j] ?? 0
        // The suppression and the matrix compute the overlap with different arithmetic, so a pair whose overlap is the
        // threshold to within rounding may fall on either side: compare with a tolerance (fast-check found such pairs).
        const eps = 1e-9
        for (const i of keep)
          for (const j of keep) if (i !== j) expect(iou(i, j)).toBeLessThanOrEqual(thr + eps)
        for (let k = 1; k < keep.length; k++)
          expect(scores[keep[k] ?? 0] ?? 0).toBeLessThanOrEqual(scores[keep[k - 1] ?? 0] ?? 0)
        // Every box is either kept or overlaps a kept box with a score at least its own.
        for (const [i] of boxes.entries())
          if (!keep.includes(i))
            expect(keep.some((k) => iou(i, k) > thr - eps && (scores[k] ?? 0) >= (scores[i] ?? 0))).toBe(true)
        const classes = boxes.map((_, i) => i % 2)
        // Per class: the best box of each class is kept, and no two kept boxes of a class overlap beyond the threshold.
        // (The count is not comparable with the plain suppression's: a box that a box of another class suppressed there
        // survives here, and may suppress one that survived there.)
        const batched = V.batchedNms(boxes, scores, classes, thr)
        for (const c of [0, 1]) {
          const inClass = boxes.flatMap((_, i) => (classes[i] === c ? [i] : []))
          if (inClass.length === 0) continue
          const best = Math.max(...inClass.map((i) => scores[i] ?? 0))
          expect(batched.some((i) => classes[i] === c && scores[i] === best)).toBe(true)
          for (const i of batched)
            for (const j of batched)
              if (i !== j && classes[i] === c && classes[j] === c)
                expect(iou(i, j)).toBeLessThanOrEqual(thr + eps)
        }
        const soft = V.softNms(boxes, scores, { scoreThreshold: 0 })
        expect(soft.indices.length).toBe(boxes.length)
      }),
      { numRuns: RUNS },
    )
  })

  it('IoU is symmetric and in [0, 1]', () => {
    fc.assert(
      fc.property(box, box, (a, b) => {
        const ab = boxIou([a], [b]).data[0] ?? -1
        const ba = boxIou([b], [a]).data[0] ?? -1
        expect(ab).toBeCloseTo(ba, 12)
        expect(ab).toBeGreaterThanOrEqual(0)
        expect(ab).toBeLessThanOrEqual(1)
      }),
      { numRuns: RUNS },
    )
  })

  it('round-trips formats and the box coder', () => {
    fc.assert(
      fc.property(box, box, (b, ref) => {
        for (const f of ['xywh', 'cxcywh'] as const) {
          const back = V.boxConvert(V.boxConvert([b], 'xyxy', f), f, 'xyxy')[0] as V.Box
          for (let k = 0; k < 4; k++) expect(back[k]).toBeCloseTo(b[k] ?? 0, 9)
        }
        const w = { weights: [10, 10, 5, 5] as [number, number, number, number] }
        const dec = V.decodeBoxes(V.encodeBoxes([b], [ref], w), [ref], w)[0] as V.Box
        for (let k = 0; k < 4; k++) expect(dec[k]).toBeCloseTo(b[k] ?? 0, 8)
      }),
      { numRuns: RUNS },
    )
  })

  it('builds anchors and matches them', () => {
    const base = V.generateAnchors([8, 16], [1])
    expect(V.gridAnchors(base, [2, 3], [4, 4]).length).toBe(12)
    expect(V.matchAnchors([[0, 0, 4, 4]], [[0, 0, 4, 5]], { allowLowQuality: true })).toEqual([0])
    expect(() => V.boxConvert([[0, 0, 1, 1]], 'xyxy', 'yxyx' as V.BoxFormat)).toThrow()
    expect(V.scaleBoxes(V.clipBoxes([[-1, -1, 5, 5]], [4, 4]), 2)).toEqual([[0, 0, 8, 8]])
  })
})
