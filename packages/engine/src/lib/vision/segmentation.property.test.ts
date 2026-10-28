import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder } from '../trace/index.ts'
import * as V from './index.ts'

/** A dark square on a light ground, with a little noise. */
const square = (seed: number, n = 20, a = 6, b = 14): number[][] => {
  const e = toFlat(normal(key(seed), [n * n]))
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i >= a && i < b && j >= a && j < b ? 0.2 : 0.8) + 0.02 * (e[i * n + j] ?? 0)),
  )
}

/** Each label is one 4-connected region. */
const connected = (labels: readonly number[], h: number, w: number): boolean => {
  const seen = new Set<number>()
  const visited = new Uint8Array(h * w)
  for (let s = 0; s < h * w; s++) {
    if (visited[s]) continue
    const lab = labels[s] ?? 0
    if (seen.has(lab)) return false
    seen.add(lab)
    const queue = [s]
    visited[s] = 1
    for (let q = 0; q < queue.length; q++) {
      const p = queue[q] as number
      const y = Math.floor(p / w)
      const x = p % w
      for (const [yy, xx] of [
        [y - 1, x],
        [y + 1, x],
        [y, x - 1],
        [y, x + 1],
      ] as const) {
        const n = yy * w + xx
        if (yy < 0 || yy >= h || xx < 0 || xx >= w || visited[n] || labels[n] !== lab) continue
        visited[n] = 1
        queue.push(n)
      }
    }
  }
  return true
}

describe('superpixels and regions', () => {
  it('SLIC labels are connected, numbered from 0 and close to the requested count', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), fc.integer({ min: 4, max: 16 }), (seed, k) => {
        const x = square(seed)
        const labels = toFlat(V.slic(x, { nSegments: k, compactness: 5 }))
        const ids = new Set(labels)
        expect([...ids].sort((a, b) => a - b)).toEqual([...ids.keys()].map((_, i) => i))
        expect(ids.size).toBeGreaterThanOrEqual(2)
        expect(ids.size).toBeLessThanOrEqual(2 * k)
        expect(connected(labels, 20, 20)).toBe(true)
      }),
      { numRuns: 5 },
    )
    const tracer = recorder()
    V.slic(square(1), { nSegments: 4, tracer, policy: policies.minimal })
    expect(tracer.events.some((e) => e.type === 'metric' && e.name === 'moved')).toBe(true)
  })

  it('grows regions from seeds, segments by mean shift and quantises by k-means', () => {
    const x = square(3)
    const grown = toFlat(
      V.regionGrowing(
        x,
        [
          [0, 0],
          [10, 10],
        ],
        { threshold: 0.15 },
      ),
    )
    expect(grown[0]).toBe(1)
    expect(grown[10 * 20 + 10]).toBe(2)
    expect(new Set(grown)).toEqual(new Set([1, 2]))
    const ms = V.meanShiftSegmentation(x, { spatialRadius: 3, colourRadius: 0.15 })
    const labels = toFlat(ms.labels)
    expect(labels[10 * 20 + 10]).not.toBe(labels[0])
    expect(labels[19 * 20 + 19]).toBe(labels[0])
    const q = V.kmeansQuantise(x, 2, { key: key(4) })
    expect(new Set(toFlat(q.image)).size).toBe(2)
    expect(q.palette.map((c) => Math.round((c[0] ?? 0) * 10) / 10).sort()).toEqual([0.2, 0.8])
    expect(() => V.kmeansQuantise(x, 0)).toThrow()
  })
})

describe('contours, cuts and seams', () => {
  it('an active contour shrinks onto a dark square', () => {
    const x = square(5, 30, 10, 20)
    const init = Array.from({ length: 24 }, (_, t): [number, number] => [
      15 + 11 * Math.cos((2 * Math.PI * t) / 24),
      15 + 11 * Math.sin((2 * Math.PI * t) / 24),
    ])
    const snake = V.activeContour(x, init, { alpha: 0.015, beta: 10, gamma: 0.001, maxIter: 300 })
    for (const [px, py] of snake) expect(Math.max(Math.abs(px - 14.5), Math.abs(py - 14.5))).toBeLessThan(9)
  })

  it('GrabCut separates an object inside its rectangle and stops when the mask settles', () => {
    const img = square(7).map((r) => r.map((v) => [v, v, 1 - v]))
    const mask = toFlat(V.grabCut(img, [3, 3, 14, 14], { key: key(2), iterations: 5 }))
    expect(mask[10 * 20 + 10]).toBe(1)
    expect(mask[4 * 20 + 4]).toBe(0)
    expect(mask[0]).toBe(0)
    const tracer = recorder()
    V.grabCut(img, [3, 3, 14, 14], { key: key(2), iterations: 10, tracer, policy: policies.minimal })
    const energies = tracer.events.filter((e) => e.type === 'metric' && e.name === 'energy')
    expect(energies.length).toBeLessThan(10)
    expect(() => V.grabCut(img, [0, 0, 20, 20])).toThrow()
  })

  it('seams are connected minimum paths; carving keeps the structure', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), (seed) => {
        const e = toFlat(normal(key(seed), [8 * 10])).map(Math.abs)
        const energy = Array.from({ length: 8 }, (_, i) => e.slice(i * 10, (i + 1) * 10))
        const seam = V.findVerticalSeam(energy)
        expect(seam.length).toBe(8)
        for (let y = 1; y < 8; y++) expect(Math.abs((seam[y] ?? 0) - (seam[y - 1] ?? 0))).toBeLessThanOrEqual(1)
        // No straight column is cheaper than the optimal seam.
        const cost = seam.reduce((s, xx, y) => s + (energy[y]?.[xx] ?? 0), 0)
        for (let xx = 0; xx < 10; xx++)
          expect(cost).toBeLessThanOrEqual(energy.reduce((s, r) => s + (r[xx] ?? 0), 0) + 1e-12)
      }),
      { numRuns: 10 },
    )
    // A flat band beside a textured one: carving removes the flat columns first.
    const img = Array.from({ length: 6 }, (_, i) => Array.from({ length: 12 }, (_, j) => (j < 6 ? 0.5 : (i + j) % 2)))
    const out = V.seamCarve(img, [5, 9])
    expect(out.shape).toEqual([5, 9])
    expect(toFlat(V.removeVerticalSeam(img, [0, 0, 0, 0, 0, 0])).length).toBe(66)
    expect(() => V.removeVerticalSeam(img, [0])).toThrow()
  })
})
