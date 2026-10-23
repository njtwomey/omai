import { goldenSuite } from '../../testing/suite.ts'
import * as L from './index.ts'

goldenSuite('segmentation losses against MONAI and the Lovasz reference', 'losses/segmentation', {
  dice: ([p, y, w], o) => L.dice(p, y, { ...o, ...(w === undefined ? {} : { classWeight: w }) }),
  tversky: ([p, y], o) => L.tversky(p, y, o),
  focalTversky: ([p, y], o) => L.focalTversky(p, y, o),
  lovaszHinge: ([x, y], o) => L.lovaszHinge(x, y, o),
  lovaszSoftmax: ([p, y], o) => L.lovaszSoftmax(p, y, o),
})
