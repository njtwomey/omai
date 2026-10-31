import { learnGolden, num } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { ellipticEnvelope, minCovDet } from './robust.ts'

learnGolden('the minimum covariance determinant and elliptic envelope against scikit-learn', 'covariance/robust', {
  mcd: (inputs, p) => {
    const [x] = inputs as [Tensor]
    const fraction = num(p, 'fraction')
    const f = minCovDet({ ...(fraction > 0 ? { supportFraction: fraction } : {}), nTrials: 30 }).fit(
      { x },
      { key: key(0) },
    )
    return {
      location: f.params.location,
      covariance: f.params.covariance,
      rawLocation: f.params.rawLocation,
      rawCovariance: f.params.rawCovariance,
      support: f.params.support.map((s) => (s ? 1 : 0)),
      mahalanobis: Array.from(toFlat(f.mahalanobis(x))),
    }
  },
  envelope: (inputs, p) => {
    const [x] = inputs as [Tensor]
    const f = ellipticEnvelope({ contamination: num(p, 'contamination'), nTrials: 30 }).fit({ x }, { key: key(0) })
    return {
      decision: Array.from(toFlat(f.decisionFunction(x))),
      predict: Array.from(toFlat(f.predict(x))),
      offset: -f.params.threshold,
    }
  },
})
