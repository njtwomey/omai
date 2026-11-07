import { learnGolden, num, str } from '../../testing/learn.ts'
import { sliceAxis, type Tensor } from '../tensor/index.ts'
import type { Criterion } from './criteria.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from './decision-tree.ts'
import { costComplexityPath } from './prune.ts'
import { type FittedTree, leafCount, treeFeatureImportances } from './tree.ts'

const arrays = (t: FittedTree, classifier: boolean) => ({
  feature: t.feature,
  threshold: t.threshold,
  left: t.left,
  right: t.right,
  impurity: t.impurity,
  nSamples: t.nSamples,
  weightedNSamples: t.weightedNSamples,
  value: classifier
    ? t.value.map((v) => {
        const s = v.reduce((a, b) => a + b, 0)
        return v.map((c) => c / s)
      })
    : t.value,
  missingLeft: t.missingLeft.map((b) => (b ? 1 : 0)),
})

const optional = (p: Readonly<Record<string, unknown>>, name: string): Record<string, number> =>
  p[name] === undefined ? {} : { [name]: num(p, name) }

const options = (p: Readonly<Record<string, unknown>>) => ({
  criterion: str(p, 'criterion') as Criterion,
  ...optional(p, 'maxDepth'),
  ...optional(p, 'minSamplesLeaf'),
  ...optional(p, 'maxLeafNodes'),
  ...optional(p, 'minImpurityDecrease'),
  ...optional(p, 'minWeightFractionLeaf'),
})

learnGolden('CART trees against scikit-learn', 'trees/grow', {
  classifier: (inputs, p) => {
    const [x, y, w] = inputs as [Tensor, Tensor, Tensor | undefined]
    const data = { x, y, ...(w === undefined ? {} : { weights: w }) }
    const f = decisionTreeClassifier(options(p) as never).fit(data)
    const out: Record<string, unknown> = {
      ...arrays(f.params, true),
      proba: f.predictProba?.(sliceAxis(x, 0, 0, 6)) as Tensor,
    }
    if (p['ccpAlpha'] !== undefined) {
      const path = costComplexityPath(f.params)
      const pruned = decisionTreeClassifier({ ...options(p), ccpAlpha: num(p, 'ccpAlpha') } as never).fit(data)
      Object.assign(out, {
        alphas: path.alphas,
        pathImpurities: path.impurities,
        prunedLeaves: leafCount(pruned.params),
        prunedFeature: pruned.params.feature,
        importances: treeFeatureImportances(f.params),
      })
    }
    return out as never
  },
  regressor: (inputs, p) => {
    const [x, y, w] = inputs as [Tensor, Tensor, Tensor | undefined]
    const f = decisionTreeRegressor(options(p) as never).fit({ x, y, ...(w === undefined ? {} : { weights: w }) })
    return { ...arrays(f.params, false), predict: f.predict(sliceAxis(x, 0, 0, 6)) } as never
  },
})
