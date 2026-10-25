/**
 * Hub entry for an intersectional audit of a classifier: its rates over the combinations of two protected attributes,
 * the worst subgroup, the entropy of benefits between and within subgroups, and a GerryFair rich-subgroup audit.
 */

import { type FairnessData, intersectionalAudit, sensitiveGroups } from 'whitebox-engine/methods/fairness'
import { choice, dict, int } from 'whitebox-engine/space'
import { asTensor, toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { awareParam, fairnessDataOf, fitClassifier } from './_shared.ts'

/**
 * The second attribute of the audit: an age band when the data has an age (German credit), else the postcode cut at
 * its median (a neighbourhood, for the generated credit data).
 */
function secondAttribute(data: FairnessData): { readonly name: string; readonly labels: string[] } {
  const names = data.featureNames ?? []
  const x = toFlat(asTensor(data.x))
  const d = names.length
  const n = d === 0 ? 0 : x.length / d
  const column = (j: number): number[] => Array.from({ length: n }, (_, i) => x[i * d + j] ?? 0)
  const age = names.indexOf('age')
  if (age >= 0) return { name: 'age', labels: column(age).map((v) => (v < 26 ? 'under 26' : '26 and over')) }
  const at = Math.max(0, names.indexOf('postcode'))
  const values = column(at)
  const median = [...values].sort((a, b) => a - b)[Math.floor(n / 2)] ?? 0
  return { name: names[at] ?? 'x0', labels: values.map((v) => (v >= median ? 'high' : 'low')) }
}

/** An intersectional audit: a classifier's rates over the combinations of two protected attributes. */
export const intersectionalAuditEntry: HubEntry = defineEntry({
  id: 'fairness/intersectional',
  title: 'Intersectional audit',
  summary:
    'Rates in every combination of two attributes, the worst subgroup, entropy between and within, and a GerryFair audit.',
  collection: 'fairness',
  tags: ['fairness', 'intersectionality', 'subgroups', 'GerryFair', 'Theil index'],
  datasets: ['fairnessTabular'],
  params: dict({
    aware: awareParam,
    rate: choice(['falsePositive', 'falseNegative'], { default: 'falsePositive', label: 'audited error rate' }),
    minSize: int(1, 100, { default: 20, label: 'smallest subgroup counted' }),
  }),
  run: (input) => {
    const data = fairnessDataOf(input)
    const aware = input.params['aware'] as boolean
    const second = secondAttribute(data)
    return intersectionalAudit(fitClassifier(data, aware, input), data, {
      attributes: [sensitiveGroups(data).column ?? 0, second.labels],
      names: [sensitiveGroups(data).attribute, second.name],
      dropSensitive: !aware,
      minSize: input.params['minSize'] as number,
      audit: input.params['rate'] as 'falsePositive' | 'falseNegative',
    })
  },
  views: [{ record: 'subgroupReport', title: 'Subgroups' }],
  methods: [
    { file: 'methods/fairness/intersectional.ts', name: 'intersectionalAudit' },
    { file: 'lib/fairness/intersectional.ts', name: 'richSubgroupAudit' },
  ],
  references: [
    'Buolamwini and Gebru (2018), Gender shades',
    'Kearns, Neel, Roth and Wu (2018), Preventing fairness gerrymandering',
    'Speicher et al. (2018), A unified approach to quantifying algorithmic unfairness',
  ],
})
