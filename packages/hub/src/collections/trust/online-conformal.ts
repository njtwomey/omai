import { kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { onlineConformal as online } from 'whitebox-engine/methods/trust'
import { dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Adaptive conformal inference on a stream whose noise drifts. */
export const onlineConformal: HubEntry = defineEntry({
  id: 'trust/online-conformal',
  title: 'Coverage as data arrive',
  summary:
    'Adaptive conformal inference keeps its running coverage near target as the noise drifts; a fixed width does not.',
  collection: 'trust',
  tags: ['conformal prediction', 'online', 'adaptive conformal inference', 'drift'],
  datasets: ['regression1d'],
  dataset: 'heteroscedastic',
  knobs: { n: 1000 },
  params: dict({
    alpha: real(0.02, 0.4, { default: 0.1, label: 'α (miscoverage)' }),
    gamma: real(0.001, 0.2, { default: 0.02, label: 'γ (step)', log: true }),
    window: int(10, 400, { default: 60 }),
  }),
  run: (input) =>
    online(kNeighborsRegressor({ nNeighbors: 15 }), learnDataOf(input.data), {
      alpha: input.params['alpha'] as number,
      gamma: input.params['gamma'] as number,
      window: input.params['window'] as number,
      key: input.key,
    }),
  views: [{ record: 'conformal', title: 'Intervals and running coverage along the stream' }],
  methods: [{ file: 'methods/trust/conformal.ts', name: 'onlineConformal' }],
  references: ['Gibbs and Candès (2021), Adaptive conformal inference under distribution shift'],
})
