import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('regression losses against torch and sklearn', 'losses/regression', {
  mse: ([x, y, w], p) => L.mse(x, y, { ...p, ...(w === undefined ? {} : { weight: w }) }),
  mae: ([x, y], p) => L.mae(x, y, p),
  huber: ([x, y], p) => L.huber(x, y, p),
  smoothL1: ([x, y], p) => L.smoothL1(x, y, p),
  logCosh: ([x, y], p) => L.logCosh(x, y, p),
  pinball: ([x, y], p) => L.pinball(x, y, p),
  expectile: ([x, y], p) => L.expectile(x, y, p),
  poissonNll: ([x, y], p) => L.poissonNll(x, y, p),
  gaussianNll: ([x, y, v], p) => L.gaussianNll(x, y, v as Tensor, p),
  tweedieDeviance: ([x, y], p) => L.tweedieDeviance(x, y, p),
})
