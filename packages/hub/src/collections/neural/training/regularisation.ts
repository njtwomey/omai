import { regularisationStudy as study } from 'whitebox-engine/methods/neural/training'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type ViewSlot } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

const STRENGTHS = { dropout: [0, 0.2, 0.5], weightDecay: [0, 1e-3, 1e-2, 1e-1] } as const

/** Dropout or weight decay against overfitting: training and validation scores per strength. */
export const regularisation: HubEntry = defineEntry({
  id: 'neural/training/regularisation',
  title: 'Dropout and weight decay',
  summary: 'An over-sized network overfits noisy data; dropout or weight decay narrow the train–validation gap.',
  collection: 'neural/training',
  tags: ['neural networks', 'regularisation', 'dropout', 'overfitting'],
  datasets: ['classification2d', 'regression1d'],
  params: dict({
    kind: choice(['dropout', 'weightDecay'] as const, { default: 'dropout', label: 'regulariser' }),
    epochs: int(10, 500, { log: true, default: 100 }),
  }),
  run: ({ data, params, key, progress }) =>
    study(
      learnDataOf(data),
      {
        kind: (params['kind'] ?? 'dropout') as 'dropout' | 'weightDecay',
        epochs: typeof params['epochs'] === 'number' ? params['epochs'] : 100,
        ...(progress === undefined ? {} : { onProgress: progress }),
      },
      { key },
    ),
  views: [
    { record: 'learningCurve', title: 'Training and validation score against the strength' },
    // Slots show the dropout rates' records (weight decay has one more strength).
    ...STRENGTHS.dropout.map(
      (v, i): ViewSlot => ({ record: 'decisionSurface', index: i, title: `strength ${v}: the fit`, area: 'side' }),
    ),
    ...STRENGTHS.dropout.map(
      (v, i): ViewSlot => ({
        record: 'learningCurve',
        index: i + 1,
        title: `strength ${v}: training against validation loss`,
        area: 'below',
      }),
    ),
  ],
  methods: [{ file: 'methods/neural/training/regularisation.ts', name: 'regularisationStudy' }],
  references: ['Srivastava et al. (2014), Dropout: a simple way to prevent neural networks from overfitting'],
})
