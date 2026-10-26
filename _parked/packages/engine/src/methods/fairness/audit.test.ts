import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { biasedCredit } from '../../lib/datasets/index.ts'
import { groupReportAt, recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { fairnessAudit } from './audit.ts'
import { sensitiveGroups, withoutColumns } from './groups.ts'

const data = generate(biasedCredit, { key: key(1), knobs: { n: 600, labelBias: 0.4 } }).data

describe('fairnessAudit', () => {
  const model = logisticRegression().fit(data)
  const audit = fairnessAudit(model, data)

  it('finds the protected column from the dataset and reports each group', () => {
    expect(audit.attribute).toBe('group')
    expect(audit.groups).toEqual(['majority', 'minority'])
    expect(audit.thresholds).toEqual([0.5, 0.5])
    // Biased labels: the model selects the minority less often.
    expect(audit.rates['minority']?.selectionRate).toBeLessThan(audit.rates['majority']?.selectionRate ?? 0)
    expect(audit.disparities.demographicParityDifference).toBeGreaterThan(0.1)
    expect(audit.disparities.disparateImpact).toBeLessThan(0.8)
  })

  it('returns a group report agreeing with the core metrics, which moves with the thresholds', () => {
    const [r] = recordsOf(audit.exposition, 'groupReport')
    if (r === undefined) throw new Error('no report')
    expect(r.groups).toEqual(audit.groups)
    expect(r.reference).toBe(0)
    expect(r.rates[1]?.selectionRate).toBeCloseTo(audit.rates['minority']?.selectionRate ?? -1, 12)
    expect(r.rates[0]?.truePositiveRate).toBeCloseTo(audit.rates['majority']?.truePositiveRate ?? -1, 12)
    const dp = r.disparities.find((d) => d.name === 'demographic parity difference')
    expect(dp?.value).toBeCloseTo(audit.disparities.demographicParityDifference, 12)
    const eo = r.disparities.find((d) => d.name === 'equalised odds difference')
    expect(eo?.value).toBeCloseTo(audit.disparities.equalizedOddsDifference, 12)
    expect(r.roc?.[0]?.fpr.length).toBeGreaterThan(5)
    expect(r.sweep?.selectionRate[1]).toHaveLength(51)
    expect(groupReportAt(r, r.thresholds).rates).toEqual(r.rates)
    // Lowering the minority's threshold selects more of it.
    const lower = groupReportAt(r, [0.5, 0.3])
    expect(lower.rates[1]?.selectionRate).toBeGreaterThan(r.rates[1]?.selectionRate ?? 1)
    expect(lower.rates[0]).toEqual(r.rates[0])
  })

  it('audits a model fitted without the protected column, and takes labels for the groups', () => {
    const unaware = withoutColumns(data, ['group'])
    expect(unaware.featureNames).toEqual(['income', 'repayment', 'postcode'])
    const fitted = logisticRegression().fit(unaware)
    const a = fairnessAudit(fitted, data, { dropSensitive: true, threshold: [0.5, 0.45] })
    expect(a.thresholds).toEqual([0.5, 0.45])
    const groups = sensitiveGroups(data).index.map((g) => (g === 1 ? 'b' : 'a'))
    const b = fairnessAudit(fitted, unaware, { sensitive: groups, reference: 'b' })
    expect(b.groups).toEqual(['a', 'b'])
    expect(recordsOf(b.exposition, 'groupReport')[0]?.reference).toBe(1)
  })

  it('gives intervals around every rate and disparity, and p-values that find the gap', () => {
    const [r] = recordsOf(audit.exposition, 'groupReport')
    const iv = r?.intervals
    expect(iv?.rates).toHaveLength(2)
    const [lo, hi] = iv?.rates[1]?.selectionRate ?? [0, 0]
    expect(lo).toBeLessThanOrEqual(r?.rates[1]?.selectionRate ?? -1)
    expect(hi).toBeGreaterThanOrEqual(r?.rates[1]?.selectionRate ?? 2)
    expect(iv?.disparities).toHaveLength(r?.disparities.length ?? -1)
    expect(iv?.pValues?.[0]).toBeLessThan(0.05)
    expect(groupReportAt(r ?? (undefined as never), [0.4, 0.4]).intervals).toBeUndefined()
    const none = fairnessAudit(model, data, { intervals: false })
    expect(recordsOf(none.exposition, 'groupReport')[0]?.intervals).toBeUndefined()
  })
})
