import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import {
  type CalibratedCost,
  calibratedEqualisedOdds,
  optimiseThresholds,
  type RejectOptionMetric,
  rejectOptionClassification,
  type ThresholdConstraint,
  thresholdDecisions,
} from './post-processing.ts'

goldenSuite('threshold optimisation against fairlearn', 'fairness/post_processing', {
  decisions: ([scores, labels, groups], o) => {
    const s = Array.from((scores as { data: ArrayLike<number> }).data)
    const y = Array.from((labels as { data: ArrayLike<number> }).data)
    const g = Array.from((groups as { data: ArrayLike<number> }).data)
    const fitted = optimiseThresholds(s, y, g, { constraint: o['constraint'] as ThresholdConstraint })
    return tensor(thresholdDecisions(fitted.rules, s, g))
  },
  mixRates: ([scores, labels, groups], o) => {
    const fit = calibratedEqualisedOdds(
      Array.from(toFlat(scores as never)),
      Array.from(toFlat(labels as never)),
      Array.from(toFlat(groups as never)),
      { cost: o['cost'] as CalibratedCost },
    )
    return tensor([...fit.mixRates])
  },
  rejectOption: ([scores, labels, groups], o) => {
    const fit = rejectOptionClassification(
      Array.from(toFlat(scores as never)),
      Array.from(toFlat(labels as never)),
      Array.from(toFlat(groups as never)),
      { metric: o['metric'] as RejectOptionMetric, thresholds: [0.01, 0.99, 40], margins: 20 },
    )
    return tensor([fit.threshold, fit.margin])
  },
})
