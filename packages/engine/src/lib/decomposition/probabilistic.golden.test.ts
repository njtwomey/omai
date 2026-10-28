import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { factorAnalysis, latentCovariance, probabilisticPca } from './probabilistic.ts'

const fa = (x: Tensor, rotation?: 'varimax' | 'quartimax') =>
  factorAnalysis.with({ nComponents: 2, rotation: rotation ?? 'none' }).fit({ x })

goldenSuite('factor analysis against scikit-learn', 'methods/decomposition/probabilistic', {
  covariance: ([x]) => tensor(latentCovariance(fa(x).params)),
  noise: ([x]) => tensor(fa(x).params.noiseVariance),
  score: ([x]) => fa(x).score({ x }),
  rotated: ([x], o) =>
    tensor(fa(x, o['rotation'] as 'varimax' | 'quartimax').params.components.map((r) => r.map(Math.abs))),
  ppcaScore: ([x]) => probabilisticPca.with({ nComponents: 2 }).fit({ x }).score({ x }),
})
