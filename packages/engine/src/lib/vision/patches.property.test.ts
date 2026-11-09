import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray } from '../tensor/index.ts'
import { extractPatches2d, imgToGraph, reconstructFromPatches2d } from './patches.ts'

const picture = fc
  .tuple(fc.integer({ min: 1, max: 9 }), fc.integer({ min: 1, max: 9 }), fc.integer({ min: 0, max: 1000 }))
  .map(([h, w, seed]) =>
    Array.from({ length: h }, (_, i) => Array.from({ length: w }, (_, j) => Math.sin(seed + i * 3.1 + j * 1.7))),
  )

describe('patch properties', () => {
  it('reconstruction undoes extraction for every patch size, and there are (H - h + 1)(W - w + 1) windows', () => {
    fc.assert(
      fc.property(picture, fc.integer({ min: 1, max: 9 }), fc.integer({ min: 1, max: 9 }), (image, a, b) => {
        const h = image.length
        const w = image[0]?.length ?? 1
        const size: [number, number] = [Math.min(a, h), Math.min(b, w)]
        const patches = extractPatches2d(image, size)
        expect(patches.shape[0]).toBe((h - size[0] + 1) * (w - size[1] + 1))
        const back = toArray(reconstructFromPatches2d(patches, [h, w])) as number[][]
        for (const [i, row] of back.entries())
          for (const [j, v] of row.entries()) expect(v).toBeCloseTo(image[i]?.[j] ?? 0, 12)
      }),
      { numRuns: 60 },
    )
  })

  it('the pixel graph has the grid edges, symmetric non-negative weights, and no weight above the range of the image', () => {
    fc.assert(
      fc.property(picture, (image) => {
        const h = image.length
        const w = image[0]?.length ?? 1
        const { graph, values } = imgToGraph(image)
        expect(graph.edges.length).toBe(h * (w - 1) + w * (h - 1))
        expect(values).toEqual(image.flat())
        const range = Math.max(...values) - Math.min(...values)
        for (const e of graph.edges) {
          expect(e.weight).toBeGreaterThanOrEqual(0)
          expect(e.weight).toBeLessThanOrEqual(range + 1e-12)
        }
      }),
      { numRuns: 50 },
    )
  })
})
