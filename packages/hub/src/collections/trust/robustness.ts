import { logisticRegression } from 'whitebox-engine/methods/linear'
import { robustnessStudy } from 'whitebox-engine/methods/trust'
import { dict } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Accuracy under input noise and missing values. */
export const robustness: HubEntry = defineEntry({
  id: 'trust/robustness',
  title: 'Robustness to noise and missing values',
  summary: 'Accuracy as Gaussian noise grows, in every feature or one at a time, and as values go missing.',
  collection: 'trust',
  tags: ['robustness', 'noise', 'missing values', 'sensitivity'],
  datasets: ['classificationTabular', 'classification2d'],
  params: dict({}),
  run: (input) => {
    const data = learnDataOf(input.data)
    return robustnessStudy(logisticRegression().fit(data, { key: input.key }), data, { key: input.key })
  },
  views: [
    { record: 'curves', index: 0, title: 'Accuracy against noise' },
    { record: 'curves', index: 1, title: 'Accuracy against missing values', area: 'side' },
  ],
  methods: [{ file: 'methods/trust/robustness.ts', name: 'robustnessStudy' }],
})
