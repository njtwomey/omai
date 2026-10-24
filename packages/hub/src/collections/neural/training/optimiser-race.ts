import { optimiserRace as race } from 'whitebox-engine/methods/neural/training'
import { dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type ViewSlot } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

const OPTIMISERS = ['sgd', 'momentum', 'rmsprop', 'adam'] as const
const TITLES = ['SGD', 'SGD with momentum', 'RMSprop', 'Adam']
const num = (v: unknown, fallback: number): number => (typeof v === 'number' ? v : fallback)

/** Four optimisers from the same initial weights on the same minibatches. */
export const optimiserRace: HubEntry = defineEntry({
  id: 'neural/training/optimiser-race',
  title: 'Optimiser race',
  summary: 'SGD, momentum, RMSprop and Adam train the same network from the same start: who gets there first?',
  collection: 'neural/training',
  tags: ['neural networks', 'optimisation', 'adam', 'momentum'],
  datasets: ['classification2d', 'regression1d'],
  params: dict({
    epochs: int(5, 300, { default: 60 }),
    batchSize: int(4, 256, { log: true, default: 32, label: 'batch size' }),
    sgd: real(1e-3, 1, { log: true, default: 0.1, label: 'SGD rate' }),
    momentum: real(1e-3, 1, { log: true, default: 0.05, label: 'momentum rate' }),
    rmsprop: real(1e-4, 0.1, { log: true, default: 0.005, label: 'RMSprop rate' }),
    adam: real(1e-4, 0.1, { log: true, default: 0.01, label: 'Adam rate' }),
  }),
  run: ({ data, params, key, progress }) =>
    race(
      learnDataOf(data),
      {
        epochs: num(params['epochs'], 60),
        batchSize: num(params['batchSize'], 32),
        learningRates: Object.fromEntries(OPTIMISERS.map((o) => [o, num(params[o], 0.01)])),
        ...(progress === undefined ? {} : { onProgress: progress }),
      },
      { key },
    ),
  views: [
    { record: 'curves', title: 'The race: training loss per epoch, every optimiser from the same start' },
    { record: 'searchTrials', title: 'Final loss per optimiser', area: 'side' },
    ...OPTIMISERS.map(
      (_, i): ViewSlot => ({ record: 'decisionSurface', index: i, title: `${TITLES[i]}: the fit`, area: 'below' }),
    ),
  ],
  methods: [{ file: 'methods/neural/training/optimisers.ts', name: 'optimiserRace' }],
  references: ['Kingma and Ba (2015), Adam: a method for stochastic optimization'],
})
