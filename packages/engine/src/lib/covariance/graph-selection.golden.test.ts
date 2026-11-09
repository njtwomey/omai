import { goldenSuite } from '../../testing/suite.ts'
import { kFold } from '../learn/model-selection/index.ts'
import { tensor } from '../tensor/index.ts'
import { type PenaltySelectionOptions, selectGraphicalLassoPenalty } from './graph-selection.ts'

const fit = (x: Parameters<typeof selectGraphicalLassoPenalty>[0], o: Readonly<Record<string, unknown>>) => {
  const alphas = o['alphas']
  const options: PenaltySelectionOptions = {
    method: 'cv',
    alphas: typeof alphas === 'number' ? alphas : (alphas as number[]),
    nRefinements: o['nRefinements'] as number,
    cv: kFold(o['folds'] as number),
    tol: 1e-10,
    enetTol: 1e-12,
    maxIter: 1000,
  }
  return selectGraphicalLassoPenalty(x, options)
}

goldenSuite('GraphicalLassoCV against scikit-learn on a fixed k-fold split', 'covariance/graph_selection', {
  alpha: ([x], o) => tensor([fit(x, o).alpha]),
  alphas: ([x], o) => tensor(fit(x, o).alphas),
  scores: ([x], o) => {
    const r = fit(x, o)
    return tensor([...r.scores, r.unpenalisedScores])
  },
  precision: ([x], o) => tensor(fit(x, o).precision),
})
