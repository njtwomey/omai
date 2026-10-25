import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { biasedCredit } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { delayedImpactStudy } from './dynamics.ts'
import { withoutColumns } from './groups.ts'
import { individualFairnessAudit } from './individual.ts'

describe('individual and dynamics studies (smoke)', () => {
  it('audits individual fairness of a credit model', () => {
    const { data } = generate(biasedCredit, { key: key(9), knobs: { n: 200 } })
    const model = logisticRegression().fit(withoutColumns(data, ['group']))
    const r = individualFairnessAudit(model, data, { pairs: 500 })
    expect(r.consistency).toBeGreaterThan(0.5)
    expect(recordsOf(r.exposition, 'coefficientTable')[0]?.names).toHaveLength(3)
    expect(recordsOf(r.exposition, 'curves')).toHaveLength(1)
  })

  it('studies the delayed impact of lending policies', () => {
    const r = delayedImpactStudy(
      { scores: [300, 450, 600, 750], repay: [0.1, 0.5, 0.8, 0.95] },
      [
        { name: 'a', share: 0.6, distribution: [0.1, 0.2, 0.3, 0.4] },
        { name: 'b', share: 0.4, distribution: [0.4, 0.3, 0.2, 0.1] },
      ],
      { rounds: 3 },
    )
    expect(r.policies).toHaveLength(3)
    expect(recordsOf(r.exposition, 'curves')).toHaveLength(2)
    expect(r.meanScore).toHaveLength(4)
  })
})
