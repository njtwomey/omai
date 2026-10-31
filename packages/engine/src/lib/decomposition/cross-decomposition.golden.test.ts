import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { cca, plsCanonical, plsRegression, plsSvd } from './cross-decomposition.ts'

const est = {
  regression: plsRegression,
  canonical: plsCanonical,
  cca,
} as const

goldenSuite('cross-decomposition against scikit-learn', 'methods/decomposition/cross_decomposition', {
  rotations: ([x, y]) => tensor(plsRegression.with({ nComponents: 2 }).fit({ x, y }).params.xRotations),
  predict: ([x, y]) => plsRegression.with({ nComponents: 2 }).fit({ x, y }).predict?.(x) ?? tensor([]),
  scores: ([x, y], o) => {
    const variant = o['variant'] as string
    return variant === 'svd'
      ? plsSvd({ nComponents: 2 }).fit({ x, y }).transform(x)
      : est[variant as keyof typeof est].with({ nComponents: 2 }).fit({ x, y }).transform(x)
  },
})
