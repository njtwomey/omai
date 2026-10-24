import { goldenSuite } from '../../testing/suite.ts'
import * as L from './index.ts'

goldenSuite('divergence losses against torch, scipy, sklearn and POT', 'losses/divergence', {
  klDiv: ([x, t], p) => L.klDiv(x, t, p),
  klDivergence: ([a, b], p) => L.klDivergence(a, b, p),
  jensenShannon: ([a, b], p) => L.jensenShannon(a, b, p),
  distillation: ([s, t, y], p) => L.distillation(s, t, { ...p, ...(y === undefined ? {} : { target: y }) }),
  mmd: ([x, y], p) => L.mmd(x, y, p),
  sinkhorn: ([x, y, a, b], p) =>
    L.sinkhorn(x, y, { ...p, ...(a === undefined ? {} : { a }), ...(b === undefined ? {} : { b }) }),
})
