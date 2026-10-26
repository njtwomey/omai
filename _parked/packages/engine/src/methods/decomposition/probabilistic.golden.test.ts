import { type Tensor, tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { factorAnalysis, latentCovariance, probabilisticPca } from './probabilistic.ts'

const fa = (x: Tensor, rotation?: 'varimax' | 'quartimax') =>
  factorAnalysis({ nComponents: 2, ...(rotation === undefined ? {} : { rotation }) }).fit({ x })

goldenSuite('factor analysis against scikit-learn', 'methods/decomposition/probabilistic', {
  covariance: ([x]) => tensor(latentCovariance(fa(x).params)),
  noise: ([x]) => tensor(fa(x).params.noiseVariance),
  score: ([x]) => fa(x).score({ x }),
  rotated: ([x], o) =>
    tensor(fa(x, o['rotation'] as 'varimax' | 'quartimax').params.components.map((r) => r.map(Math.abs))),
  ppcaScore: ([x]) => probabilisticPca({ nComponents: 2 }).fit({ x }).score({ x }),
})
