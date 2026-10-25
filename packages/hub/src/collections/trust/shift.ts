/**
 * Hub entry for covariate and concept shift and the tests that detect them (Kolmogorov-Smirnov tests and a domain
 * classifier), on tabular classification data and two-dimensional classification data.
 */

import { logisticRegression } from 'mdml-engine/methods/linear'
import { type ShiftKind, shiftStudy } from 'mdml-engine/methods/trust'
import { choice, dict } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Covariate and concept shift, and what detects them. */
export const shift: HubEntry = defineEntry({
  id: 'trust/shift',
  title: 'Dataset shift',
  summary: 'Shift the current rows: KS tests and a domain classifier catch covariate shift, not concept shift.',
  collection: 'trust',
  tags: ['dataset shift', 'drift detection', 'two-sample tests', 'domain classifier'],
  datasets: ['classificationTabular', 'classification2d'],
  dataset: 'makeClassification',
  knobs: { n: 1000, nFeatures: 5, nInformative: 3, nRedundant: 0 },
  params: dict({ kind: choice(['covariate', 'concept'], { default: 'covariate' }) }),
  run: (input) =>
    shiftStudy(logisticRegression(), learnDataOf(input.data), {
      kind: input.params['kind'] as ShiftKind,
      key: input.key,
    }),
  views: [{ record: 'curves', title: 'Detection and accuracy against the shift' }],
  methods: [{ file: 'methods/trust/shift.ts', name: 'shiftStudy' }],
  references: ['Rabanser, Günnemann and Lipton (2019), Failing loudly'],
})
