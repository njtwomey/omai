import { goldenSuite } from '../../../testing/suite.ts'
import type { BorderMode } from '../../numerics/convolution/index.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import * as I from './index.ts'

goldenSuite('image features against scikit-image', 'signal/image/features', {
  structureTensor: ([x], o) =>
    tensor(
      I.structureTensor(x, (o['sigma'] as number | undefined) ?? 1, {
        ...(o['mode'] === undefined ? {} : { mode: o['mode'] as BorderMode }),
      }).flatMap((t) => toFlat(t)),
    ),
  cornerHarris: ([x], o) =>
    I.cornerHarris(x, {
      ...(o['method'] === undefined ? {} : { method: o['method'] as 'k' | 'eps' }),
      ...(o['sigma'] === undefined ? {} : { sigma: o['sigma'] as number }),
    }),
  cornerShiTomasi: ([x]) => I.cornerShiTomasi(x),
  peakLocalMax: ([x], o) =>
    tensor(
      I.peakLocalMax(x, {
        minDistance: o['minDistance'] as number,
        ...(o['thresholdRel'] === undefined ? {} : { thresholdRel: o['thresholdRel'] as number }),
        ...(o['excludeBorder'] === undefined ? {} : { excludeBorder: o['excludeBorder'] as boolean }),
        ...(o['numPeaks'] === undefined ? {} : { numPeaks: o['numPeaks'] as number }),
      }).flat(),
    ),
  canny: ([x], o) =>
    I.canny(x, {
      sigma: o['sigma'] as number,
      ...(o['lowThreshold'] === undefined ? {} : { lowThreshold: o['lowThreshold'] as number }),
      ...(o['highThreshold'] === undefined ? {} : { highThreshold: o['highThreshold'] as number }),
      useQuantiles: o['useQuantiles'] === true,
    }),
  localBinaryPattern: ([x], o) =>
    I.localBinaryPattern(x, o['P'] as number, o['R'] as number, o['method'] as 'default' | 'ror' | 'uniform'),
  hog: ([x], o) =>
    I.hog(x, {
      orientations: o['orientations'] as number,
      pixelsPerCell: o['pixelsPerCell'] as [number, number],
      cellsPerBlock: o['cellsPerBlock'] as [number, number],
      blockNorm: o['blockNorm'] as 'L1' | 'L1-sqrt' | 'L2' | 'L2-Hys',
    }),
  houghLine: ([x], o) => {
    const r = I.houghLine(x, o['thetas'] as number[] | undefined)
    return tensor([...toFlat(r.accumulator), ...Array.from(r.thetas), ...Array.from(r.distances)])
  },
  houghCircle: ([x], o) => I.houghCircle(x, o['radii'] as number[], o['normalize'] as boolean),
})
