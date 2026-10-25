import { tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { optimiseThresholds, type ThresholdConstraint, thresholdDecisions } from './post-processing.ts'

goldenSuite('threshold optimisation against fairlearn', 'methods/fairness/post_processing', {
  decisions: ([scores, labels, groups], o) => {
    const s = Array.from((scores as { data: ArrayLike<number> }).data)
    const y = Array.from((labels as { data: ArrayLike<number> }).data)
    const g = Array.from((groups as { data: ArrayLike<number> }).data)
    const fitted = optimiseThresholds(s, y, g, { constraint: o['constraint'] as ThresholdConstraint })
    return tensor(thresholdDecisions(fitted.rules, s, g))
  },
})
