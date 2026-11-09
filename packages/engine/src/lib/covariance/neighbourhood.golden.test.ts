import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { climeAlgorithm, neighbourhoodSelectionAlgorithm } from './neighbourhood.ts'

goldenSuite('neighbourhood selection and CLIME against scikit-learn and scipy', 'covariance/neighbourhood', {
  coefficients: ([x], o) => {
    const p = (toArray(x) as number[][])[0]?.length ?? 0
    const run = runAlgorithm(
      neighbourhoodSelectionAlgorithm(x, { penalty: o['penalty'] as number, tol: 1e-14, maxIter: 100_000 }),
      {
        maxSteps: p,
      },
    )
    return tensor(run.state.coefficients)
  },
  climeColumns: ([s], o) => {
    const rows = toArray(s) as number[][]
    return tensor(
      runAlgorithm(climeAlgorithm(rows, { lambda: o['lambda_'] as number }), { maxSteps: rows.length }).state.columns,
    )
  },
})
