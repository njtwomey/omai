import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('boxes against torchvision and OpenCV', 'vision/boxes', {
  boxConvert: ([b], o) => tensor(V.boxConvert(b, o['source'] as V.BoxFormat, o['target'] as V.BoxFormat)),
  boxArea: ([b]) => tensor(V.boxArea(b)),
  clipBoxes: ([b], o) => tensor(V.clipBoxes(b, o['size'] as [number, number])),
  removeSmallBoxes: ([b], o) => tensor(V.removeSmallBoxes(b, o['minSize'] as number)),
  nms: ([b, s], o) => tensor(V.nms(b, s, o['iouThreshold'] as number)),
  batchedNms: ([b, s, c], o) => tensor(V.batchedNms(b, s, toFlat(c as never), o['iouThreshold'] as number)),
  softNms: ([b, s], o) => {
    const r = V.softNms(b, s, {
      method: o['method'] as 'gaussian' | 'linear',
      sigma: o['sigma'] as number,
      iouThreshold: o['iouThreshold'] as number,
      scoreThreshold: o['scoreThreshold'] as number,
    })
    return tensor([...r.indices, ...r.scores])
  },
  generateAnchors: (_, o) => tensor(V.generateAnchors(o['sizes'] as number[], o['ratios'] as number[])),
  gridAnchors: ([b], o) => tensor(V.gridAnchors(b, o['gridSize'] as [number, number], o['stride'] as [number, number])),
  encodeBoxes: ([g, r], o) =>
    tensor(V.encodeBoxes(g, r, { weights: o['weights'] as [number, number, number, number] })),
  decodeBoxes: ([c, r], o) =>
    tensor(V.decodeBoxes(c, r, { weights: o['weights'] as [number, number, number, number] })),
  matchAnchors: ([t, a], o) => tensor(V.matchAnchors(t, a, { allowLowQuality: o['allowLowQuality'] as boolean })),
})
