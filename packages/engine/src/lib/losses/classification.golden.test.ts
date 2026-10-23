import { goldenSuite } from '../../testing/suite.ts'
import * as L from './index.ts'

goldenSuite('classification losses against torch and torchvision', 'losses/classification', {
  binaryCrossEntropyWithLogits: ([x, y], p) => L.binaryCrossEntropyWithLogits(x, y, p),
  binaryCrossEntropy: ([x, y], p) => L.binaryCrossEntropy(x, y, p),
  crossEntropy: ([z, y, w], p) => L.crossEntropy(z, y, { ...p, ...(w === undefined ? {} : { classWeight: w }) }),
  nllLoss: ([lp, y, w], p) => L.nllLoss(lp, y, { ...p, ...(w === undefined ? {} : { classWeight: w }) }),
  focal: ([x, y], p) => L.focal(x, y, p),
  categoricalFocal: ([z, y, w], p) => L.categoricalFocal(z, y, { ...p, ...(w === undefined ? {} : { alpha: w }) }),
  multiLabelSoftMargin: ([x, y, w], p) =>
    L.multiLabelSoftMargin(x, y, { ...p, ...(w === undefined ? {} : { classWeight: w }) }),
})
