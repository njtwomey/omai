import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { biasedCredit, germanCredit } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { dataBiasAudit } from './data-bias.ts'
import { intersectionalAudit } from './intersectional.ts'

describe('intersectional and data-bias audits', () => {
  it('audits a credit model over sex, age band and foreign worker as a subgroup report', () => {
    const data = germanCredit()
    const x = toFlat(data.x)
    const age = Array.from({ length: 1000 }, (_, i) => ((x[i * 12 + 6] ?? 0) < 26 ? 'under 26' : '26 and over'))
    const model = logisticRegression({ maxIter: 200 }).fit(data)
    const r = intersectionalAudit(model, data, { attributes: ['female', age], names: ['sex', 'age'] })
    const [rec] = recordsOf(r.exposition, 'subgroupReport')
    expect(rec?.attributes).toEqual(['sex', 'age'])
    expect(rec?.subgroups).toHaveLength(4)
    expect(rec?.counts.reduce((a, b) => a + b, 0)).toBe(1000)
    expect(rec?.entropy?.total).toBeGreaterThan(0)
    expect(rec?.audit?.features).toEqual(['sex=female', 'age=under 26'])
    expect(r.selectionGap).toBeGreaterThan(0)
  })

  it('audits the bias in a dataset, with clean labels and a model', () => {
    const { data, truth } = generate(biasedCredit, { key: key(2), knobs: { n: 800, labelBias: 0.4 } })
    const model = logisticRegression().fit(data)
    const r = dataBiasAudit(data, {
      cleanLabels: truth?.latents?.['cleanLabels'] as number[],
      model,
      modelWithoutSensitive: false,
      populationShares: { majority: 0.5, minority: 0.5 },
    })
    const [rec] = recordsOf(r.exposition, 'dataBias')
    expect(rec?.groups).toEqual(['majority', 'minority'])
    expect(rec?.proxies?.features).toEqual(['income', 'repayment', 'postcode'])
    expect(r.strongestProxy?.feature).toBe('postcode')
    expect(rec?.labelFlips?.positiveToNegative[1]).toBeGreaterThan(0.2)
    expect(rec?.amplification?.selectionRates).toHaveLength(2)
    expect(r.statisticalParityDifference).toBeLessThan(0)
  })
})
