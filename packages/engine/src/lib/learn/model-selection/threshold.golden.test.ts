import { learnGolden, num, str } from '../../../testing/learn.ts'
import type { Tensor } from '../../tensor/index.ts'
import { gaussianNaiveBayes } from '../naive-bayes.ts'
import { stratifiedKFold } from './splitters.ts'
import { fixedThresholdClassifier, tunedThresholdClassifierCV } from './threshold.ts'

learnGolden('threshold tuning against scikit-learn over GaussianNB', 'learn/model_selection/threshold', {
  tuned: ([x, y, held], p) => {
    const f = tunedThresholdClassifierCV({
      estimator: gaussianNaiveBayes(),
      scoring: str(p, 'scoring'),
      cv: stratifiedKFold(num(p, 'folds')),
      thresholds: num(p, 'thresholds'),
    }).fit({ x: x as Tensor, y: y as Tensor })
    return {
      threshold: f.params.threshold,
      score: f.params.bestScore ?? Number.NaN,
      predict: f.predict(held as Tensor),
    }
  },
  fixed: ([x, y, held], p) => {
    const f = fixedThresholdClassifier({ estimator: gaussianNaiveBayes(), threshold: num(p, 'threshold') }).fit({
      x: x as Tensor,
      y: y as Tensor,
    })
    return { predict: f.predict(held as Tensor) }
  },
})
