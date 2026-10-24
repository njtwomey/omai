import { goldenSuite } from '../../testing/suite.ts'
import * as M from './index.ts'

goldenSuite('segmentation metrics against MONAI, sklearn and torchmetrics', 'metrics/segmentation', {
  diceCoefficient: ([p, t]) => M.diceCoefficient(p, t).estimate.value,
  maskIou: ([p, t]) => M.maskIou(p, t).estimate.value,
  maskHausdorffDistance: ([p, t], o) => M.maskHausdorffDistance(p, t, o).estimate.value,
  meanIou: ([p, t]) => M.meanIou(p, t).estimate.value,
  boundaryF1: ([p, t], o) => M.boundaryF1(p, t, o).estimate.value,
  panopticQuality: ([p, t], o) => M.panopticQuality(p, t, o as unknown as M.PanopticOptions).estimate.value,
})
