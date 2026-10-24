import { goldenSuite } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as I from './index.ts'

goldenSuite('matching and registration against scikit-image', 'signal/image/motion', {
  matchTemplate: ([x, t], o) => I.matchTemplate(x, t as never, o['padInput'] === true),
  phaseCrossCorrelation: ([a, b]) => {
    const r = I.phaseCrossCorrelation(a, b as never)
    return tensor([...r.shift, r.error, r.phasediff])
  },
})
