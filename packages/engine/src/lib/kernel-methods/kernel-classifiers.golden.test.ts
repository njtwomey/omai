import { learnGolden, num, str } from '../../testing/learn.ts'
import type { PairwiseKernelMetric } from '../kernels/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { kernelLogisticRegression, kernelRidgeClassifier, lssvm } from './kernel-classifiers.ts'

const options = (p: Readonly<Record<string, unknown>>) => ({
  kernel: str(p, 'kernel', 'rbf') as PairwiseKernelMetric,
  gamma: num(p, 'gamma', 1),
})

learnGolden(
  'kernel classifiers against scikit-learn (feature-space logistic regression, RidgeClassifier, KernelRidge)',
  'kernel_methods/classifiers',
  {
    klr: ([x, y, xt], p) => ({
      decision: kernelLogisticRegression({ ...options(p), lambda: num(p, 'lam'), tol: 1e-12 })
        .fit({ x: x as Tensor, y: y as Tensor })
        .decisionFunction(xt as Tensor),
    }),
    lssvm: ([x, y, xt], p) => {
      const fit = lssvm({ ...options(p), regularisation: num(p, 'c') }).fit({ x: x as Tensor, y: y as Tensor })
      return {
        dual: fit.params.dualCoef.map((r) => r[0] ?? 0),
        bias: fit.params.intercept,
        decision: fit.decisionFunction(xt as Tensor),
      }
    },
    ridgeClassifier: ([x, y, xt], p) => ({
      decision: kernelRidgeClassifier({ ...options(p), alpha: num(p, 'alpha') })
        .fit({ x: x as Tensor, y: y as Tensor })
        .decisionFunction(xt as Tensor),
    }),
  },
)
