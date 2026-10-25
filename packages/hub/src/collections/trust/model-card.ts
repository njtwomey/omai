import { logisticRegression } from 'whitebox-engine/methods/linear'
import { modelCard as card } from 'whitebox-engine/methods/trust'
import { dict } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** The evidence of a model card: performance by group, calibration and robustness. */
export const modelCard: HubEntry = defineEntry({
  id: 'trust/model-card',
  title: 'Model card',
  summary: 'The evidence behind a model card: performance by group, calibration and robustness of a fitted model.',
  collection: 'trust',
  tags: ['model cards', 'documentation', 'fairness', 'calibration', 'robustness'],
  datasets: ['fairnessTabular'],
  params: dict({}),
  run: (input) => {
    const data = learnDataOf(input.data)
    return card(logisticRegression().fit(data, { key: input.key }), data, { key: input.key })
  },
  views: [
    { record: 'groupReport', title: 'Performance by group' },
    { record: 'calibration', title: 'Calibration', area: 'side' },
    { record: 'curves', index: 0, title: 'Robustness to noise', area: 'below' },
    { record: 'curves', index: 1, title: 'Robustness to missing values', area: 'below' },
  ],
  methods: [{ file: 'methods/trust/model-cards.ts', name: 'modelCard' }],
  references: ['Mitchell et al. (2019), Model cards for model reporting'],
})
