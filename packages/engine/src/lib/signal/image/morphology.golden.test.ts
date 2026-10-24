import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { BorderMode } from '../../numerics/convolution/index.ts'
import { type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import * as I from './index.ts'

const binary = (o: Params, s: Tensor | undefined): I.BinaryOptions => ({
  ...(s === undefined ? {} : { structure: s }),
  ...(o['iterations'] === undefined ? {} : { iterations: o['iterations'] as number }),
  ...(o['borderValue'] === undefined ? {} : { borderValue: o['borderValue'] as number }),
  ...(o['origin'] === undefined ? {} : { origin: o['origin'] as number[] }),
})
const grey = (o: Params, fp: Tensor | undefined): I.GreyOptions => ({
  ...(fp === undefined ? {} : { footprint: fp }),
  ...(o['size'] === undefined ? {} : { size: o['size'] as number[] }),
  ...(o['mode'] === undefined ? {} : { mode: o['mode'] as BorderMode }),
  ...(o['cval'] === undefined ? {} : { cval: o['cval'] as number }),
  ...(o['origin'] === undefined ? {} : { origin: o['origin'] as number }),
})

goldenSuite('morphology against scipy.ndimage and scikit-image', 'signal/image/morphology', {
  generateBinaryStructure: (_x, o) => I.generateBinaryStructure(o['rank'] as number, o['connectivity'] as number),
  binaryErosion: ([x, s], o) => I.binaryErosion(x, binary(o, s)),
  binaryDilation: ([x, s], o) => I.binaryDilation(x, binary(o, s)),
  binaryOpening: ([x, s], o) => I.binaryOpening(x, binary(o, s)),
  binaryClosing: ([x, s], o) => I.binaryClosing(x, binary(o, s)),
  greyErosion: ([x, fp], o) => I.greyErosion(x, grey(o, fp)),
  greyDilation: ([x, fp], o) => I.greyDilation(x, grey(o, fp)),
  greyOpening: ([x, fp], o) => I.greyOpening(x, grey(o, fp)),
  greyClosing: ([x, fp], o) => I.greyClosing(x, grey(o, fp)),
  morphologicalGradient: ([x, fp], o) => I.morphologicalGradient(x, grey(o, fp)),
  whiteTophat: ([x, fp], o) => I.whiteTophat(x, grey(o, fp)),
  blackTophat: ([x, fp], o) => I.blackTophat(x, grey(o, fp)),
  label: ([x, s]) => {
    const r = I.label(x, s)
    return tensor([...toFlat(r.labels), r.count])
  },
  distanceTransformEdt: ([x], o) => I.distanceTransformEdt(x, (o['sampling'] as number[] | undefined) ?? 1),
  thin: ([x], o) => I.thin(x, (o['maxIter'] as number | undefined) ?? Number.POSITIVE_INFINITY),
})
