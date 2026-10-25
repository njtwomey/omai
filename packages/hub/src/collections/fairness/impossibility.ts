/**
 * Hub entry showing why calibration and equalised odds cannot both hold when groups have different base rates, on
 * synthetic scores.
 */

import { impossibilityStudy } from 'whitebox-engine/methods/fairness'
import { dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

/** Calibration against equalised odds: why both cannot hold when base rates differ. */
export const impossibility: HubEntry = defineEntry({
  id: 'fairness/impossibility',
  title: 'Calibration against equalised odds',
  summary:
    'With unequal base rates, calibrated scores have unequal error rates; equalising them breaks predictive parity.',
  collection: 'fairness',
  tags: ['fairness', 'impossibility', 'calibration', 'equalised odds', 'study'],
  datasets: [],
  params: dict({ n: int(500, 10_000, { default: 3000, label: 'applicants per dataset' }) }),
  run: ({ params, key }) => impossibilityStudy({ n: params['n'] as number, key }),
  views: [{ record: 'curves', title: 'Disparities against the base-rate gap' }],
  methods: [{ file: 'methods/fairness/studies.ts', name: 'impossibilityStudy' }],
  references: [
    'Kleinberg, Mullainathan and Raghavan (2016), Inherent trade-offs in the fair determination of risk scores',
    'Chouldechova (2017), Fair prediction with disparate impact',
  ],
})
