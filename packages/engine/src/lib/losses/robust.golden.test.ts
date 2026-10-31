import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('noise-robust classification losses against their reference formulas', 'losses/robust', {
  generalisedCrossEntropy: ([z, y], p) => L.generalisedCrossEntropy(z, y, p),
  truncatedCrossEntropy: ([z, y], p) => L.truncatedCrossEntropy(z, y, p),
  symmetricCrossEntropy: ([z, y], p) => L.symmetricCrossEntropy(z, y, p),
  bootstrapped: ([z, y], p) => L.bootstrapped(z, y, p as { beta: number; mode: 'soft' | 'hard' }),
  forwardCorrected: ([z, y, t]) => L.forwardCorrected(z, y, t as Tensor),
  backwardCorrected: ([z, y, t]) => L.backwardCorrected(z, y, t as Tensor),
  biTemperedLogistic: ([z, y], p) => L.biTemperedLogistic(z, y, p),
})
