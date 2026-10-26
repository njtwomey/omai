import { tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { cca, plsCanonical, plsRegression, plsSvd } from './cross-decomposition.ts'

const est = { regression: plsRegression, canonical: plsCanonical, cca, svd: plsSvd } as const

goldenSuite('cross-decomposition against scikit-learn', 'methods/decomposition/cross_decomposition', {
  rotations: ([x, y]) => tensor(plsRegression({ nComponents: 2 }).fit({ x, y }).params.xRotations),
  predict: ([x, y]) => plsRegression({ nComponents: 2 }).fit({ x, y }).predict?.(x) ?? tensor([]),
  scores: ([x, y], o) => est[o['variant'] as keyof typeof est]({ nComponents: 2 }).fit({ x, y }).transform(x),
})
