import { learnGolden, num, str } from '../../testing/learn.ts'
import type { DistanceName } from '../distances/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { localOutlierFactor } from './neighbours.ts'

const metric = (p: Readonly<Record<string, unknown>>): DistanceName => str(p, 'metric', 'euclidean') as DistanceName

learnGolden('the local outlier factor against scikit-learn', 'anomaly/lof', {
  train: (inputs, p) => {
    const [x] = inputs as [Tensor]
    return {
      scores: localOutlierFactor({ nNeighbors: num(p, 'k'), metric: metric(p), algorithm: 'brute' }).fit({ x }).params
        .scores,
    }
  },
  novel: (inputs, p) => {
    const [x, q] = inputs as [Tensor, Tensor]
    const f = localOutlierFactor({ nNeighbors: num(p, 'k'), metric: metric(p) }).fit({ x })
    return { scores: Array.from(toFlat(f.scoreSamples(q))) }
  },
  contamination: (inputs, p) => {
    const [x, q] = inputs as [Tensor, Tensor]
    const f = localOutlierFactor({ nNeighbors: 10, contamination: num(p, 'contamination') }).fit({ x })
    return {
      decision: Array.from(toFlat(f.decisionFunction(q))),
      predict: Array.from(toFlat(f.predict(q))),
      threshold: f.params.threshold,
    }
  },
  auto: (inputs) => {
    const [x, q] = inputs as [Tensor, Tensor]
    const f = localOutlierFactor({ nNeighbors: 10 }).fit({ x })
    return { decision: Array.from(toFlat(f.decisionFunction(q))), predict: Array.from(toFlat(f.predict(q))) }
  },
})
