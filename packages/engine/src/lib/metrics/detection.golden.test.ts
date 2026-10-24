import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as M from './index.ts'

interface Image {
  readonly gt_boxes: number[][]
  readonly gt_labels: number[]
  readonly crowd: number[]
  readonly difficult: number[]
  readonly dt_boxes: number[][]
  readonly dt_labels: number[]
  readonly scores: number[]
}

const split = (images: readonly Image[], flag: 'iscrowd' | 'difficult') => ({
  preds: images.map((im) => ({ boxes: im.dt_boxes, scores: im.scores, labels: im.dt_labels })),
  targets: images.map((im) => ({
    boxes: im.gt_boxes,
    labels: im.gt_labels,
    ...(flag === 'iscrowd' ? { iscrowd: im.crowd } : { difficult: im.difficult }),
  })),
})

goldenSuite('box overlaps and detection mAP against torchvision, pycocotools and voc_eval', 'metrics/detection', {
  boxIou: ([a, b]) => tensor(M.boxIou(a, b)),
  generalizedBoxIou: ([a, b]) => tensor(M.generalizedBoxIou(a, b)),
  distanceBoxIou: ([a, b]) => tensor(M.distanceBoxIou(a, b)),
  completeBoxIou: ([a, b]) => tensor(M.completeBoxIou(a, b)),
  cocoStat: (_inputs, p) => {
    const { preds, targets } = split(p['images'] as Image[], 'iscrowd')
    return M.detectionMeanAveragePrecision(preds, targets).breakdown?.[p['stat'] as string] ?? Number.NaN
  },
  vocMap: (_inputs, p) => {
    const { preds, targets } = split(p['images'] as Image[], 'difficult')
    return M.detectionMeanAveragePrecision(preds, targets, { protocol: p['protocol'] as 'voc' | 'voc2007' }).estimate
      .value
  },
})
