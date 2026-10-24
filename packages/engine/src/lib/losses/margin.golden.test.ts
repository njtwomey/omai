import { goldenSuite } from '../../testing/suite.ts'
import * as L from './index.ts'

goldenSuite('margin losses against optax, torch and sklearn', 'losses/margin', {
  hinge: ([f, y], p) => L.hinge(f, y, p),
  squaredHinge: ([f, y], p) => L.squaredHinge(f, y, p),
  logistic: ([f, y], p) => L.logistic(f, y, p),
  exponential: ([f, y], p) => L.exponential(f, y, p),
  modifiedHuber: ([f, y], p) => L.modifiedHuber(f, y, p),
  perceptron: ([f, y], p) => L.perceptron(f, y, p),
  crammerSinger: ([s, y], p) => L.crammerSinger(s, y, p),
  westonWatkins: ([s, y, w], p) => L.westonWatkins(s, y, { ...p, ...(w === undefined ? {} : { classWeight: w }) }),
})
