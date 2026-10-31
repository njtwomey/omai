import { learnGolden, num, str } from '../../testing/learn.ts'
import type { PairwiseKernelMetric } from '../kernels/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { kernelRidge } from './kernel-ridge.ts'

const hyperOf = (p: Readonly<Record<string, unknown>>) => ({
  kernel: str(p, 'kernel', 'rbf') as PairwiseKernelMetric,
  alpha: num(p, 'alpha'),
  gamma: num(p, 'gamma', 1),
  degree: num(p, 'degree', 3),
  coef0: num(p, 'coef0', 1),
})

learnGolden('kernel ridge regression against sklearn.kernel_ridge', 'kernel_methods/kernel_ridge', {
  kernelRidge: ([x, y, xt], p) => {
    const f = kernelRidge.with(hyperOf(p)).fit({ x: x as Tensor, y: y as Tensor })
    return { dualCoef: f.dualCoef, predict: f.predict(xt as Tensor) }
  },
  looRidge: ([x, y], p) => ({
    predict: kernelRidge
      .with({ ...hyperOf({ ...p, kernel: 'rbf' }) })
      .fit({ x: x as Tensor, y: y as Tensor })
      .looPredictions(),
  }),
})
