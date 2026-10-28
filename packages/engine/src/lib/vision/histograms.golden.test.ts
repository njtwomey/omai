import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('histograms against OpenCV and scikit-image', 'vision/histograms', {
  calcHist: ([x, mask], o) =>
    V.calcHist(x, {
      bins: o['bins'] as number | number[],
      ...(o['channels'] === undefined ? {} : { channels: o['channels'] as number[] }),
      ...(mask === undefined ? {} : { mask }),
    }),
  calcBackProject: ([x, h], o) =>
    V.calcBackProject(x, h, { channels: o['channels'] as number[], scale: o['scale'] as number }),
  equalizeHist: ([x]) => V.equalizeHist(x),
  clahe: ([x], o) =>
    V.clahe(x, { clipLimit: o['clipLimit'] as number, tileGridSize: o['tileGridSize'] as [number, number] }),
  matchHistograms: ([x, ref]) => V.matchHistograms(x, ref),
  rescaleIntensity: ([x], o) => V.rescaleIntensity(x, { inRange: o['inRange'] as [number, number] }),
  contrastStretch: ([x], o) => V.contrastStretch(x, { low: o['low'] as number, high: o['high'] as number }),
  normalize: ([x], o) =>
    V.normalize(x, { normType: o['normType'] as V.NormType, alpha: o['alpha'] as number, beta: o['beta'] as number }),
  compareHist: ([a, b], o) => tensor([V.compareHist(a, b, o['method'] as V.HistogramComparison)]),
})
