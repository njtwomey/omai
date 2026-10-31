import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { kernelPlsRegression } from './kernel-multivariate.ts'

learnGolden('kernel PLS with a linear kernel against sklearn PLSRegression', 'kernel_methods/pls', {
  kernelPls: ([x, y, xt], p) => ({
    predict: kernelPlsRegression({
      kernel: 'linear',
      gamma: 1,
      nComponents: num(p, 'nComponents'),
      tol: 1e-15,
      maxIter: 20000,
    })
      .fit({ x: x as Tensor, y: y as Tensor })
      .predict(xt as Tensor),
  }),
})
