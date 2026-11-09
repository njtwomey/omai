import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { extractPatches2d, imgToGraph, patchExtractor, reconstructFromPatches2d } from './patches.ts'

const image = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
]

describe('extractPatches2d', () => {
  it('takes every window in raster order of its corner', () => {
    const p = extractPatches2d(image, [2, 2])
    expect(p.shape).toEqual([6, 2, 2])
    expect((toArray(p) as number[][][])[0]).toEqual([
      [1, 2],
      [5, 6],
    ])
    expect((toArray(p) as number[][][])[1]).toEqual([
      [2, 3],
      [6, 7],
    ])
    expect((toArray(p) as number[][][])[3]).toEqual([
      [5, 6],
      [9, 10],
    ])
  })

  it('keeps the channels of a colour image', () => {
    const rgb = image.map((r) => r.map((v) => [v, -v]))
    const p = extractPatches2d(rgb, [2, 3])
    expect(p.shape).toEqual([4, 2, 3, 2])
    expect((toArray(p) as number[][][][])[0]?.[1]?.[2]).toEqual([7, -7])
  })

  it('a random sample takes windows of the image at corners drawn from the key, a fraction or a count', () => {
    const some = extractPatches2d(image, [2, 2], { maxPatches: 4, key: key(1) })
    expect(some.shape).toEqual([4, 2, 2])
    expect(toArray(extractPatches2d(image, [2, 2], { maxPatches: 4, key: key(1) }))).toEqual(toArray(some))
    expect(toArray(extractPatches2d(image, [2, 2], { maxPatches: 4, key: key(2) }))).not.toEqual(toArray(some))
    expect(extractPatches2d(image, [2, 2], { maxPatches: 0.5, key: key(1) }).shape).toEqual([3, 2, 2])
    const all = toArray(extractPatches2d(image, [2, 2])) as number[][][]
    for (const w of toArray(some) as number[][][])
      expect(all.some((a) => JSON.stringify(a) === JSON.stringify(w))).toBe(true)
  })

  it('refuses a patch bigger than the image, a bad count, and a sample without a key', () => {
    expect(() => extractPatches2d(image, [4, 2])).toThrow()
    expect(() => extractPatches2d(image, [0, 2])).toThrow()
    expect(() => extractPatches2d(image, [2, 2], { maxPatches: 7, key: key(0) })).toThrow()
    expect(() => extractPatches2d(image, [2, 2], { maxPatches: 2 })).toThrow()
    expect(() => extractPatches2d(image, [2, 2], { maxPatches: -1, key: key(0) })).toThrow()
  })
})

describe('reconstructFromPatches2d', () => {
  it('is the inverse of extraction, for any patch size, including one larger than half the image', () => {
    for (const size of [
      [1, 1],
      [2, 2],
      [2, 4],
      [3, 3],
      [3, 4],
    ] as const) {
      const back = reconstructFromPatches2d(extractPatches2d(image, size), [3, 4])
      expect(toArray(back)).toEqual(image)
    }
  })

  it('averages the patches that cover a pixel', () => {
    const p = extractPatches2d([[0, 0, 0]], [1, 2])
    const bumped = toArray(p) as number[][][]
    bumped[0] = [[2, 2]]
    bumped[1] = [[4, 4]]
    expect(toArray(reconstructFromPatches2d(bumped, [1, 3]))).toEqual([[2, 3, 4]])
  })

  it('rebuilds colour images and refuses a patch count that does not tile', () => {
    const rgb = image.map((r) => r.map((v) => [v, 2 * v, 3 * v]))
    expect(toArray(reconstructFromPatches2d(extractPatches2d(rgb, [2, 2]), [3, 4]))).toEqual(rgb)
    expect(() => reconstructFromPatches2d(extractPatches2d(image, [2, 2]), [3, 5])).toThrow()
  })
})

describe('patchExtractor', () => {
  const batch = [image, image.map((r) => r.map((v) => -v))]

  it('stacks the patches of each image, and takes a tenth of the image by default', () => {
    expect(
      patchExtractor({ patchSize: [2, 2] })
        .fit({ x: batch })
        .transform(batch).shape,
    ).toEqual([12, 2, 2])
    const big = [Array.from({ length: 20 }, (_, i) => Array.from({ length: 30 }, (_, j) => i + j))]
    expect(patchExtractor().fit({ x: big }).transform(big).shape).toEqual([19 * 28, 2, 3])
  })

  it('random corners come from the key of the fit and are the same on every call', () => {
    const est = patchExtractor({ patchSize: [2, 2], maxPatches: 3 })
    const f = est.fit({ x: batch }, { key: key(5) })
    expect(f.transform(batch).shape).toEqual([6, 2, 2])
    expect(toArray(f.transform(batch))).toEqual(toArray(f.transform(batch)))
    expect(toArray(est.revive(f.params).transform(batch))).toEqual(toArray(f.transform(batch)))
  })

  it('refuses a patch bigger than the image and rank-2 input', () => {
    expect(() =>
      patchExtractor({ patchSize: [5, 5] })
        .fit({ x: batch })
        .transform(batch),
    ).toThrow()
    expect(() => patchExtractor().fit({ x: image }).transform(image)).toThrow()
  })
})

describe('imgToGraph', () => {
  it('joins neighbours by their absolute differences and keeps the values', () => {
    const { graph, values } = imgToGraph([
      [1, 3],
      [2, 7],
    ])
    expect(graph.n).toBe(4)
    expect(graph.edges.map((e) => [e.source, e.target, e.weight])).toEqual([
      [0, 1, 2],
      [0, 2, 1],
      [1, 3, 4],
      [2, 3, 5],
    ])
    expect(values).toEqual([1, 3, 2, 7])
  })

  it('a mask keeps some pixels and the edges among them, renumbered', () => {
    const { graph, values } = imgToGraph(
      [
        [1, 3, 5],
        [2, 7, 11],
      ],
      {
        mask: [
          [1, 0, 1],
          [1, 1, 1],
        ],
      },
    )
    expect(graph.n).toBe(5)
    expect(values).toEqual([1, 5, 2, 7, 11])
    expect(graph.edges.map((e) => [e.source, e.target])).toEqual([
      [0, 2],
      [1, 4],
      [2, 3],
      [3, 4],
    ])
    expect(() => imgToGraph([[1, 2]], { mask: [[1, 1, 1]] })).toThrow()
  })

  it('a volume has edges along its third axis too', () => {
    const { graph } = imgToGraph([[[1, 2]], [[4, 8]]])
    expect(graph.edges.length).toBe(4)
  })
})
