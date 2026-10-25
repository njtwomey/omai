/**
 * Hub entry for a fairness audit of a classifier: its rates, gaps to a reference group and ROC curves across the groups
 * of a protected attribute, on tabular data with a protected attribute.
 */

import { fairnessAudit as audit } from 'whitebox-engine/methods/fairness'
import { dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { awareParam, fairnessDataOf, fitClassifier } from './_shared.ts'

/** A fairness audit: a classifier's rates, gaps and ROC curves across the groups of a protected attribute. */
export const fairnessAudit: HubEntry = defineEntry({
  id: 'fairness/audit',
  title: 'Fairness audit',
  summary: 'Rates side by side per group, gaps to the reference group, ROC curves and per-group contingency tables.',
  collection: 'fairness',
  tags: ['fairness', 'audit', 'demographic parity', 'equalised odds', 'classification'],
  datasets: ['fairnessTabular'],
  params: dict({
    aware: awareParam,
    threshold: real(0.05, 0.95, { default: 0.5, label: 'decision threshold' }),
  }),
  run: (input) => {
    const data = fairnessDataOf(input)
    const aware = input.params['aware'] as boolean
    return audit(fitClassifier(data, aware, input), data, {
      dropSensitive: !aware,
      threshold: input.params['threshold'] as number,
    })
  },
  views: [{ record: 'groupReport', title: 'Group report' }],
  methods: [
    { file: 'methods/fairness/audit.ts', name: 'fairnessAudit' },
    { file: 'lib/metrics/fairness.ts', name: 'groupRates' },
  ],
  references: ['Barocas, Hardt and Narayanan (2023), Fairness and Machine Learning', 'fairlearn MetricFrame'],
})
