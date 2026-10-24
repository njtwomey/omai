import { passiveAggressiveClassifier } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fitEstimator } from '../../entry/run.ts'

/** The passive-aggressive classifier (Crammer et al. 2006): fix each margin violation with the smallest step. */
export const passiveAggressive: HubEntry = defineEntry({
  id: 'online/passive-aggressive',
  title: 'Passive-aggressive classifier',
  summary: 'Leave the weights alone when a row is classified with margin; otherwise move just enough to fix it.',
  collection: 'online',
  tags: ['online learning', 'classification', 'linear', 'margin'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'blobs',
  ...fitEstimator(passiveAggressiveClassifier({ maxIter: 20 })),
  views: [
    { record: 'learningCurve', title: 'Progressive and training error' },
    { record: 'decisionSurface', title: 'Decision surface', area: 'side' },
    { record: 'curves', title: 'Coefficients as rows arrive', area: 'below' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'below' },
  ],
  methods: [{ file: 'methods/linear/online.ts', name: 'passiveAggressiveClassifier' }],
  references: ['Crammer, Dekel, Keshet, Shalev-Shwartz and Singer (2006), Online passive-aggressive algorithms'],
})
