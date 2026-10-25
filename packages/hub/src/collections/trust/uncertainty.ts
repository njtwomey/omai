/**
 * Hub entry for epistemic and aleatoric uncertainty from a bootstrap ensemble, its members disagreeing away from the
 * data, on one-dimensional regression curves.
 */

import { decisionTreeRegressor } from 'whitebox-engine/methods/trees'
import { ensembleUncertainty } from 'whitebox-engine/methods/trust'
import { dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Epistemic and aleatoric uncertainty from a bootstrap ensemble. */
export const uncertainty: HubEntry = defineEntry({
  id: 'trust/uncertainty',
  title: 'Epistemic and aleatoric uncertainty',
  summary: 'A bootstrap ensemble: its members disagree away from the data (epistemic); the noise stays (aleatoric).',
  collection: 'trust',
  tags: ['uncertainty', 'ensembles', 'bootstrap', 'epistemic', 'aleatoric'],
  datasets: ['regression1d'],
  params: dict({
    members: int(2, 60, { default: 20 }),
    maxDepth: int(1, 12, { default: 5, label: 'tree depth' }),
  }),
  run: (input) =>
    ensembleUncertainty(
      decisionTreeRegressor({ maxDepth: input.params['maxDepth'] as number }),
      learnDataOf(input.data),
      {
        members: input.params['members'] as number,
        key: input.key,
      },
    ),
  views: [{ record: 'curves', title: 'Mean with epistemic and total bands' }],
  methods: [{ file: 'methods/trust/uncertainty.ts', name: 'ensembleUncertainty' }],
  references: ['Lakshminarayanan, Pritzel and Blundell (2017), Simple and scalable predictive uncertainty'],
})
