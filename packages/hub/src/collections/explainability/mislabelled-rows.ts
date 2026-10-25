/**
 * The data valuation entry: flipping some training labels and asking k-NN Shapley, data Banzhaf and Beta Shapley to
 * find them.
 */

import { mislabelledRowStudy, type ValuationMethod } from 'whitebox-engine/methods/explain'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Mislabelled-row detection by data valuation. */
export const mislabelledRows: HubEntry = defineEntry({
  id: 'explainability/mislabelled-rows',
  title: 'Finding mislabelled rows by data value',
  summary: 'Flip some labels, value every training row, inspect the least valuable first: how many flips are found.',
  collection: 'explainability',
  tags: ['data valuation', 'data Shapley', 'Beta Shapley', 'data Banzhaf', 'KNN Shapley', 'label noise'],
  datasets: ['classificationTabular'],
  dataset: 'breastCancer',
  params: dict({
    flip: real(0.02, 0.4, { default: 0.15, label: 'share of labels flipped' }),
    rows: int(30, 200, { default: 80, label: 'training rows' }),
    samples: int(10, 300, { default: 60, log: true, label: 'Banzhaf subsets' }),
    permutations: int(2, 60, { default: 10, label: 'Beta Shapley permutations' }),
  }),
  run: ({ data, params, key }) =>
    mislabelledRowStudy(logisticRegression({ C: 1 }), learnDataOf(data), {
      flipFraction: Number(params['flip']),
      trainRows: Number(params['rows']),
      nSamples: Number(params['samples']),
      nPermutations: Number(params['permutations']),
      methods: ['knn shapley', 'data banzhaf', 'beta shapley'] as ValuationMethod[],
      key,
    }),
  views: [{ record: 'curves', title: 'Flipped labels found as rows are inspected' }],
  methods: [{ file: 'methods/explain/valuation.ts', name: 'mislabelledRowStudy' }],
  references: [
    'Ghorbani and Zou (2019), Data Shapley: equitable valuation of data for machine learning.',
    'Kwon and Zou (2022), Beta Shapley: a unified and noise-reduced data valuation framework for machine learning.',
    'Wang and Jia (2023), Data Banzhaf: a robust data valuation framework for machine learning.',
    'Jia et al. (2019), Efficient task-specific data valuation for nearest neighbor algorithms.',
  ],
})
