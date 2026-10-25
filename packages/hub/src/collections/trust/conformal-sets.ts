/**
 * Hub entry for conformal prediction sets for a classifier, sets of classes holding the true one at a chosen rate, on
 * two-dimensional classification data and tabular classification data.
 */

import type { ClassificationScore } from 'whitebox-engine/learn'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { conformalClassification } from 'whitebox-engine/methods/trust'
import { bool, choice, dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Conformal prediction sets for a classifier. */
export const conformalSets: HubEntry = defineEntry({
  id: 'trust/conformal-sets',
  title: 'Conformal prediction sets',
  summary: 'Sets of classes holding the true one for 1 − α of new rows: larger where the classes overlap.',
  collection: 'trust',
  tags: ['conformal prediction', 'prediction sets', 'coverage', 'classification'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  knobs: { n: 600 },
  params: dict({
    alpha: real(0.01, 0.5, { default: 0.1, label: 'α (miscoverage)' }),
    score: choice(['lac', 'aps'], { default: 'lac' }),
    classConditional: bool({ default: false, label: 'calibrate per class' }),
  }),
  run: (input) =>
    conformalClassification(logisticRegression(), learnDataOf(input.data), {
      alpha: input.params['alpha'] as number,
      score: input.params['score'] as ClassificationScore,
      classConditional: input.params['classConditional'] as boolean,
      key: input.key,
    }),
  views: [{ record: 'conformal', title: 'Set sizes and calibration scores' }],
  methods: [{ file: 'methods/trust/conformal.ts', name: 'conformalClassification' }],
  references: ['Romano, Sesia and Candès (2020), Classification with valid and adaptive coverage', 'MAPIE'],
})
