import { goldenSuite } from '../../testing/suite.ts'
import { normal } from '../dists/index.ts'
import { exp, matmul, mul, type Tensor, tanh, type Value, zerosLike } from '../tensor/index.ts'
import * as L from './index.ts'

const critic =
  (w: Tensor, v: Tensor): L.Critic =>
  (x: Value) =>
    matmul(tanh(matmul(x, w)), v)

goldenSuite('generative-model losses against torch', 'losses/generative', {
  discriminatorLoss: ([r, f], p) => L.discriminatorLoss(r, f, p),
  generatorLoss: ([f], p) => L.generatorLoss(f, p),
  gradientPenalty: ([xr, xf, w, v, a]) =>
    L.gradientPenalty(critic(w as Tensor, v as Tensor), xr, xf, { alpha: a as Tensor }),
  r1Penalty: ([x, w, v]) => L.r1Penalty(critic(w, v as Tensor), x),
  gaussianKl: ([m, s], p) => L.gaussianKl(m, s, p),
  sampledKl: ([m, s, z]) =>
    L.sampledKl(normal(m, exp(mul(s, 0.5))), normal(zerosLike(m), exp(zerosLike(m))), z as Tensor),
  bernoulliLogLikelihood: ([l, x]) => L.bernoulliLogLikelihood(l, x),
  gaussianLogLikelihood: ([m, x, s]) => L.gaussianLogLikelihood(m, x, s === undefined ? {} : { logVariance: s }),
  negativeElbo: ([ll, kl], p) => L.negativeElbo(ll, kl, p),
  importanceWeightedBound: ([w]) => L.importanceWeightedBound(w),
  denoisingScoreMatching: ([s, e, sg], p) => L.denoisingScoreMatching(s, e, sg as Tensor, p),
  diffusionLoss: ([pr, x0, e, a, s], p) => L.diffusionLoss(pr, x0, e as Tensor, a as Tensor, s as Tensor, p),
  flowMatchingPath: ([x0, x1, t], p) => L.flowMatchingPath(x0, x1, t as Tensor, p).xt,
  flowMatching: ([v, x0, x1], p) => L.flowMatching(v, x0, x1 as Tensor, p),
})
