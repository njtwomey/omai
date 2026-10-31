import { learnGolden, num, str } from '../../testing/learn.ts'
import type { DistanceName } from '../distances/index.ts'
import type { Tensor } from '../tensor/index.ts'
import {
  kNeighboursClassifier,
  kNeighboursRegressor,
  type NeighbourAlgorithm,
  nearestCentroid,
  radiusNeighboursClassifier,
  radiusNeighboursRegressor,
} from './neighbours.ts'

const NAMES: Readonly<Record<string, DistanceName>> = {
  euclidean: 'euclidean',
  manhattan: 'manhattan',
  chebyshev: 'chebyshev',
  cosine: 'cosine',
  minkowski: 'minkowski',
}

const search = (p: Readonly<Record<string, unknown>>) => ({
  metric: NAMES[str(p, 'metric', 'euclidean')] ?? 'euclidean',
  metricOptions: typeof p['p'] === 'number' ? { p: p['p'] } : {},
  algorithm: str(p, 'algorithm', 'brute') as NeighbourAlgorithm,
  leafSize: num(p, 'leafSize', 30),
})

learnGolden('neighbour learners against sklearn.neighbors', 'learn/neighbours', {
  kClassifier: ([x, y, t], p) => {
    const f = kNeighboursClassifier({
      ...search(p),
      nNeighbors: num(p, 'k'),
      weights: str(p, 'weights') as 'uniform' | 'distance',
    }).fit({ x: x as Tensor, y: y as Tensor })
    return { predict: f.predict(t as Tensor), proba: f.predictProba(t as Tensor) }
  },
  kRegressor: ([x, y, t], p) => {
    const f = kNeighboursRegressor({
      ...search(p),
      nNeighbors: num(p, 'k'),
      weights: str(p, 'weights') as 'uniform' | 'distance',
    }).fit({ x: x as Tensor, y: y as Tensor })
    return { predict: f.predict(t as Tensor) }
  },
  radiusClassifier: ([x, y, t], p) => {
    const outlier = p['outlier']
    const f = radiusNeighboursClassifier({
      radius: num(p, 'radius'),
      weights: str(p, 'weights') as 'uniform' | 'distance',
      outlierLabel: outlier === 'most-frequent' ? 'most-frequent' : (outlier as number),
    }).fit({ x: x as Tensor, y: y as Tensor })
    return { predict: f.predict(t as Tensor), proba: f.predictProba(t as Tensor) }
  },
  radiusRegressor: ([x, y, t], p) => {
    const f = radiusNeighboursRegressor({
      radius: num(p, 'radius'),
      weights: str(p, 'weights') as 'uniform' | 'distance',
    }).fit({ x: x as Tensor, y: y as Tensor })
    return { predict: f.predict(t as Tensor) }
  },
  centroid: ([x, y, t], p) => {
    const f = nearestCentroid({
      metric: str(p, 'metric') as 'euclidean' | 'manhattan',
      priors: str(p, 'priors') as 'uniform' | 'empirical',
      ...(typeof p['shrink'] === 'number' ? { shrinkThreshold: p['shrink'] } : {}),
    }).fit({ x: x as Tensor, y: y as Tensor })
    const out: Record<string, Tensor | readonly (readonly number[])[]> = {
      centroids: f.params.centroids,
      predict: f.predict(t as Tensor),
      withinStd: f.params.withinStd as unknown as Tensor,
    }
    if (p['metric'] === 'euclidean') {
      out['decision'] = f.decisionFunction(t as Tensor)
      out['proba'] = f.predictProba(t as Tensor)
    }
    return out as never
  },
})
