import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { extractPatches2d, imgToGraph, patchExtractor, reconstructFromPatches2d } from './patches.ts'

learnGolden('image patches and pixel graphs against scikit-learn', 'vision/patches', {
  extract: ([image], p) => {
    const patches = extractPatches2d(image as Tensor, [num(p, 'patchHeight'), num(p, 'patchWidth')])
    return {
      patches,
      rebuilt: reconstructFromPatches2d(patches, [(image as Tensor).shape[0] ?? 0, (image as Tensor).shape[1] ?? 0]),
    }
  },
  reconstruct: ([patches], p) => ({
    image: reconstructFromPatches2d(patches as Tensor, [num(p, 'height'), num(p, 'width')]),
  }),
  extractor: ([images], p) => {
    const h = num(p, 'patchHeight')
    const f = patchExtractor(h === 0 ? {} : { patchSize: [h, num(p, 'patchWidth')] }).fit({ x: images as Tensor })
    return { patches: f.transform(images as Tensor) }
  },
  graph: ([image, mask]) => {
    const { graph, values } = imgToGraph(image as Tensor, mask === undefined ? {} : { mask })
    const dense = Array.from({ length: graph.n }, () => new Array<number>(graph.n).fill(0))
    for (const e of graph.edges) {
      ;(dense[e.source] as number[])[e.target] = e.weight
      ;(dense[e.target] as number[])[e.source] = e.weight
    }
    for (const [i, v] of values.entries()) (dense[i] as number[])[i] = v
    return { matrix: dense }
  },
})
