/**
 * Hub entry: the accuracy–fairness frontier, in training and after it.
 */

import type { ReductionConstraint } from 'whitebox-engine/fairness'
import { fairnessFrontier } from 'whitebox-engine/methods/fairness'
import { choice, dict } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fairnessDataOf } from './_shared.ts'

/** The accuracy–fairness frontier, in training and after it. */
export const frontier: HubEntry = defineEntry({
  id: 'fairness/frontier',
  title: 'Accuracy–fairness frontier',
  summary: 'The best accuracy at each level of unfairness: a grid of prices in training, a moved threshold after.',
  collection: 'fairness',
  tags: ['fairness', 'trade-off', 'Pareto front', 'study'],
  datasets: ['fairnessTabular'],
  maxRows: 1500,
  params: dict({ constraint: choice(['demographicParity', 'equalisedOdds'], { default: 'demographicParity' }) }),
  run: (input) =>
    fairnessFrontier(fairnessDataOf(input), {
      constraint: input.params['constraint'] as ReductionConstraint,
      key: input.key,
    }),
  views: [{ record: 'curves', title: 'Accuracy against violation' }],
  methods: [{ file: 'methods/fairness/studies.ts', name: 'fairnessFrontier' }],
})
