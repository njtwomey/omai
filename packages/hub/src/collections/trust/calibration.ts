/**
 * Hub entry for recalibrating an overconfident naive Bayes classifier by Platt, isotonic and temperature scaling, with
 * reliability diagrams before and after, on tabular classification data.
 */

import { gaussianNB } from 'whitebox-engine/methods/naive-bayes'
import { calibrateClassifier } from 'whitebox-engine/methods/trust'
import { dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Recalibration: Platt, isotonic and temperature scaling of an overconfident classifier. */
export const calibration: HubEntry = defineEntry({
  id: 'trust/calibration',
  title: 'Probability calibration',
  summary: 'Reliability diagrams before and after Platt, isotonic and temperature scaling of naive Bayes.',
  collection: 'trust',
  tags: ['calibration', 'reliability diagram', 'ECE', 'Platt scaling', 'isotonic regression'],
  datasets: ['classificationTabular'],
  dataset: 'makeClassification',
  knobs: { n: 2000, nFeatures: 12, nInformative: 3, nRedundant: 6, nClasses: 2 },
  params: dict({ nBins: int(3, 30, { default: 10, label: 'bins' }) }),
  run: (input) => {
    const data = learnDataOf(input.data)
    const model = gaussianNB().fit(data, { key: input.key })
    return calibrateClassifier(model, data, { nBins: input.params['nBins'] as number, key: input.key })
  },
  views: [{ record: 'calibration', title: 'Reliability on held-out rows' }],
  methods: [{ file: 'methods/trust/calibration.ts', name: 'calibrateClassifier' }],
  references: ['Niculescu-Mizil and Caruana (2005)', 'Guo et al. (2017), On calibration of modern neural networks'],
})
