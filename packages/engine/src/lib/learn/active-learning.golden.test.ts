import { gaussianNB, learnGolden, num, rows, str } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  classMargin,
  consensusEntropySampling,
  entropySampling,
  leastConfidence,
  marginSampling,
  maxDisagreementSampling,
  predictiveEntropy,
  type QueryContext,
  type QueryStrategy,
  rankedBatchSampling,
  uncertaintySampling,
  voteEntropySampling,
} from './active-learning.ts'
import { requireVerb } from './estimator.ts'
import { matrixView } from './validation.ts'

const strategies: Readonly<Record<string, () => QueryStrategy>> = {
  uncertainty: uncertaintySampling,
  margin: marginSampling,
  entropy: entropySampling,
  'ranked-batch': rankedBatchSampling,
  'vote-entropy': voteEntropySampling,
  'consensus-entropy': consensusEntropySampling,
  'max-disagreement': maxDisagreementSampling,
}

function context(x: Tensor, y: Tensor, pool: Tensor, committee: QueryContext['committee'] = []): QueryContext {
  const estimator = gaussianNB()
  return {
    fitted: estimator.fit({ x, y }),
    committee,
    labeled: { x, y },
    pool: matrixView(pool, 'test'),
    estimator,
    key: key(0),
    fitOptions: {},
  }
}

learnGolden('active-learning queries against modAL', 'learn/active_learning', {
  scores: ([x, y, pool]) => {
    const p = requireVerb(gaussianNB().fit({ x: x as Tensor, y: y as Tensor }), 'predictProba', 'test')(pool as Tensor)
    return { uncertainty: leastConfidence(p), margin: classMargin(p), entropy: predictiveEntropy(p) }
  },
  query: ([x, y, pool], p) => {
    const strategy = strategies[str(p, 'strategy')]?.()
    if (strategy === undefined) throw new Error('unknown strategy')
    const picked = strategy.select(context(x as Tensor, y as Tensor, pool as Tensor), num(p, 'n'))
    return { indices: str(p, 'strategy') === 'ranked-batch' ? picked : [...picked].sort((a, b) => a - b) }
  },
  committee: ([x, y, pool, masks], p) => {
    const xs = rows(x as Tensor)
    const ys = toFlat(y as Tensor)
    const committee = rows(masks as Tensor).map((mask) => {
      const idx = mask.flatMap((m, i) => (m === 1 ? [i] : []))
      return gaussianNB().fit({ x: idx.map((i) => xs[i] ?? []), y: idx.map((i) => ys[i] ?? 0) })
    })
    const strategy = strategies[str(p, 'strategy')]?.()
    if (strategy === undefined) throw new Error('unknown strategy')
    const picked = strategy.select(context(x as Tensor, y as Tensor, pool as Tensor, committee), num(p, 'n'))
    return { indices: [...picked].sort((a, b) => a - b) }
  },
})
