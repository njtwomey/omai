import { learnGolden, num, str } from '../../../testing/learn.ts'
import type { Tensor } from '../../tensor/index.ts'
import { treePredict } from './shapley.ts'
import { type LimeFeatureSelection, limeFit, surrogateTree } from './surrogates.ts'

learnGolden('surrogates against lime and sklearn trees', 'learn/explain/surrogates', {
  lime: ([z, y, d], p) => {
    const r = limeFit(z as Tensor, y as Tensor, d as Tensor, {
      kernelWidth: num(p, 'width'),
      nFeatures: num(p, 'k'),
      featureSelection: str(p, 'selection') as LimeFeatureSelection,
    })
    return {
      intercept: r.intercept,
      features: r.weights.map(([j]) => j),
      weights: r.weights.map(([, w]) => w),
      score: r.score,
      local: [r.localPrediction],
    }
  },
  treeRegression: ([x, y, z], p) => {
    const t = surrogateTree(x as Tensor, y as Tensor, { maxDepth: num(p, 'depth') })
    return { predict: treePredict(t, z as Tensor), train: treePredict(t, x as Tensor) }
  },
  treeClassification: ([x, y, z], p) => {
    const t = surrogateTree(x as Tensor, y as Tensor, {
      task: 'classification',
      maxDepth: num(p, 'depth'),
      minSamplesLeaf: num(p, 'minLeaf'),
    })
    return { predict: treePredict(t, z as Tensor), train: treePredict(t, x as Tensor) }
  },
})
