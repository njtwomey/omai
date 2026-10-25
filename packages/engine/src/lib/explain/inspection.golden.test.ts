import { gaussianNB, learnGolden, leastSquares, num, rows } from '../../testing/learn.ts'
import { pipeline } from '../learn/compose.ts'
import type { Fitted } from '../learn/estimator.ts'
import { polynomialFeatures } from '../learn/preprocessing/index.ts'
import type { Tensor } from '../tensor/index.ts'
import {
  accumulatedLocalEffects,
  functionalAnova,
  hStatistic,
  partialDependence,
  permutationImportance,
} from './inspection.ts'

const polyModel = (x: Tensor, y: Tensor): Fitted =>
  pipeline([
    ['poly', polynomialFeatures()],
    ['lr', leastSquares()],
  ]).fit({ x, y })

const features = (p: Readonly<Record<string, unknown>>): number[] => p['features'] as number[]

learnGolden('inspection against sklearn.inspection', 'explain/inspection', {
  permutation: ([x, y, perms]) => {
    const r = permutationImportance(
      polyModel(x as Tensor, y as Tensor),
      { x: x as Tensor, y: y as Tensor },
      {
        scoring: 'r2',
        permutations: rows(perms as Tensor),
      },
    )
    return { importances: r.importances, mean: r.mean, std: r.std }
  },
  pdRegression: ([x, y], p) => {
    const r = partialDependence(polyModel(x as Tensor, y as Tensor), x as Tensor, features(p), {
      gridResolution: num(p, 'resolution'),
    })
    return {
      average: r.average,
      grid: r.values.flat(),
      ...(r.individual[0] === undefined ? {} : { individual: r.individual[0] }),
    }
  },
  pdClassifier: ([x, y], p) => {
    const f = gaussianNB().fit({ x: x as Tensor, y: y as Tensor })
    const r = partialDependence(f, x as Tensor, features(p), { gridResolution: num(p, 'resolution') })
    return { average: r.average }
  },
  ale: ([x, y], p) => {
    const r = accumulatedLocalEffects(polyModel(x as Tensor, y as Tensor), x as Tensor, num(p, 'feature'), {
      bins: num(p, 'bins'),
    })
    return { edges: r.edges, effects: r.effects[0] ?? [] }
  },
  hStatistic: ([x, y, xs]) => {
    const f = polyModel(x as Tensor, y as Tensor)
    return { pair: hStatistic(f, xs as Tensor, 0, 1), total: hStatistic(f, xs as Tensor, 0) }
  },
  anova: ([x, y, xs]) => {
    const r = functionalAnova(polyModel(x as Tensor, y as Tensor), xs as Tensor)
    return { shares: r.components.map((c) => c.share) }
  },
})
