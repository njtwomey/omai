import { reweighedFit } from 'whitebox-engine/methods/fairness'
import { dict } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { awareParam, fairnessDataOf } from './_shared.ts'

/** Preprocessing: weight rows so the label is independent of the group. */
export const reweighing: HubEntry = defineEntry({
  id: 'fairness/reweighing',
  title: 'Reweighing',
  summary: 'Weight each row by P(group) P(label) / P(group, label) and refit: the weighted labels ignore the group.',
  collection: 'fairness',
  tags: ['fairness', 'preprocessing', 'sample weights'],
  datasets: ['fairnessTabular'],
  params: dict({ aware: awareParam }),
  run: (input) =>
    reweighedFit(fairnessDataOf(input), { dropSensitive: !(input.params['aware'] as boolean), key: input.key }),
  views: [
    { record: 'groupReport', index: 1, title: 'Reweighed' },
    { record: 'coefficientTable', title: 'Weight per group and label', area: 'side' },
    { record: 'groupReport', index: 0, title: 'Unweighted', area: 'below' },
  ],
  methods: [{ file: 'methods/fairness/preprocessing.ts', name: 'reweighing' }],
  references: ['Kamiran and Calders (2012), Data preprocessing techniques for classification without discrimination'],
})
