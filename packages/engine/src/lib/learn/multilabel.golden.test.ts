import { gaussianNB, learnGolden, logisticRegression, str } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { binaryRelevance, classifierChain, labelPowerset } from './multilabel.ts'
import type { ChainMethod } from './multioutput.ts'

learnGolden('multilabel wrappers against scikit-learn', 'learn/multilabel', {
  relevance: (inputs) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = binaryRelevance(logisticRegression()).fit({ x, y })
    return {
      predict: f.predict(q),
      proba: (f.predictProba as NonNullable<typeof f.predictProba>)(q),
      decision: (f.decisionFunction as NonNullable<typeof f.decisionFunction>)(q),
    }
  },
  chain: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const order = Array.from(toFlat(p['order'] as Tensor))
    const f = classifierChain(logisticRegression(), { order, chainMethod: str(p, 'method') as ChainMethod }).fit({
      x,
      y,
    })
    return { predict: f.predict(q), proba: (f.predictProba as NonNullable<typeof f.predictProba>)(q) }
  },
  powerset: (inputs) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = labelPowerset(gaussianNB()).fit({ x, y })
    return { predict: f.predict(q), proba: f.predictProba(q), combinations: f.predictCombinations(q) }
  },
})
