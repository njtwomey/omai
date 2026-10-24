import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import type { ClassificationCriterion, RegressionCriterion } from './criteria.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from './grow.ts'
import { costComplexityPruningPath } from './prune.ts'
import { nLeaves, type TreeParams } from './tree.ts'

type In = readonly [Tensor, Tensor]

const arrays = (t: TreeParams) => ({
  feature: t.feature,
  threshold: t.threshold,
  left: t.left,
  right: t.right,
  impurity: t.impurity,
  nSamples: t.nSamples,
  value:
    t.task === 'classification'
      ? t.value.map((v) => {
          const s = v.reduce((a, b) => a + b, 0)
          return v.map((c) => c / s)
        })
      : t.value,
})

learnGolden('CART trees against scikit-learn', 'methods/trees/grow', {
  classifier: (inputs, p) => {
    const [x, y] = inputs as In
    const hyper = { criterion: str(p, 'criterion') as ClassificationCriterion, maxDepth: 4, minSamplesLeaf: 5 }
    const f = decisionTreeClassifier(hyper).fit({ x, y })
    const path = costComplexityPruningPath(f.params)
    const pruned = decisionTreeClassifier({ ...hyper, ccpAlpha: num(p, 'ccpAlpha') }).fit({ x, y })
    return {
      ...arrays(f.params),
      alphas: path.alphas,
      pathImpurities: path.impurities,
      prunedLeaves: nLeaves(pruned.params),
      importances: f.featureImportances,
      proba: f.predictProba?.(sliceAxis(x, 0, 0, 6)) as Tensor,
    }
  },
  regressor: (inputs, p) => {
    const [x, y] = inputs as In
    const f = decisionTreeRegressor({
      criterion: str(p, 'criterion') as RegressionCriterion,
      maxDepth: 3,
      minSamplesLeaf: 5,
    }).fit({ x, y })
    return { ...arrays(f.params), predict: f.predict(sliceAxis(x, 0, 0, 6)) }
  },
})
