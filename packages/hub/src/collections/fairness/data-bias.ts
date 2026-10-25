/**
 * Hub entry for the bias in a dataset with a protected attribute: representation, base-rate gaps in the labels, the
 * features that act as proxies for the group, and how much a fitted model amplifies the gap.
 */

import { dataBiasAudit } from 'mdml-engine/methods/fairness'
import { dict } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { awareParam, fairnessDataOf, fitClassifier } from './_shared.ts'

/** The bias in a dataset: representation, base rates, proxies and a model's amplification. */
export const dataBiasEntry: HubEntry = defineEntry({
  id: 'fairness/data-bias',
  title: 'Bias in data',
  summary: 'Before any model: who is represented, base-rate gaps, which features reveal the group, and amplification.',
  collection: 'fairness',
  tags: ['fairness', 'data bias', 'proxies', 'representation', 'bias amplification'],
  datasets: ['fairnessTabular'],
  params: dict({ aware: awareParam }),
  run: (input) => {
    const data = fairnessDataOf(input)
    const aware = input.params['aware'] as boolean
    return dataBiasAudit(data, {
      model: fitClassifier(data, aware, input),
      modelWithoutSensitive: !aware,
    })
  },
  views: [{ record: 'dataBias', title: 'Bias in the data' }],
  methods: [
    { file: 'methods/fairness/data-bias.ts', name: 'dataBiasAudit' },
    { file: 'lib/fairness/data-bias.ts', name: 'proxyScores' },
  ],
  references: [
    'Mehrabi et al. (2021), A survey on bias and fairness in machine learning',
    'Wang and Russakovsky (2021), Directional bias amplification',
  ],
})
