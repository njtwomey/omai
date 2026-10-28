import { goldenSuite } from '../../testing/suite.ts'
import * as V from './index.ts'

goldenSuite('the running average against OpenCV', 'vision/background', {
  accumulateWeighted: ([x, acc], o) => V.accumulateWeighted(x, acc, o['alpha'] as number),
})
