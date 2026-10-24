import { universalApproximation as study } from 'whitebox-engine/methods/neural/foundations'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type ViewSlot } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

const WIDTHS = [1, 2, 4, 8, 16, 32] as const

/** One hidden layer of growing width fitting the same curve. */
export const universalApproximation: HubEntry = defineEntry({
  id: 'neural/foundations/universal-approximation',
  title: 'Universal approximation in one dimension',
  summary: 'One hidden layer fits any curve given enough units: watch the fit sharpen as the layer widens.',
  collection: 'neural/foundations',
  tags: ['neural networks', 'regression', 'universal approximation'],
  datasets: ['regression1d'],
  dataset: 'dopplerCurve',
  params: dict({
    activation: choice(['tanh', 'relu', 'sigmoid'] as const, { default: 'tanh' }),
    epochs: int(20, 2000, { log: true, default: 300 }),
  }),
  run: ({ data, params, key }) =>
    study(
      learnDataOf(data),
      {
        widths: WIDTHS,
        activation: (params['activation'] ?? 'tanh') as 'tanh' | 'relu' | 'sigmoid',
        epochs: typeof params['epochs'] === 'number' ? params['epochs'] : 300,
      },
      { key },
    ),
  views: [
    { record: 'learningCurve', title: 'R² against the number of hidden units' },
    {
      record: 'decisionSurface',
      index: WIDTHS.length,
      title: `The ${WIDTHS.at(-1)} units of the widest network: output weight × activation`,
      area: 'side',
    },
    ...WIDTHS.map(
      (w, i): ViewSlot => ({
        record: 'decisionSurface',
        index: i,
        title: `${w} hidden unit${w === 1 ? '' : 's'}`,
        area: 'below',
      }),
    ),
  ],
  methods: [{ file: 'methods/neural/foundations/approximation.ts', name: 'universalApproximation' }],
  references: [
    'Cybenko (1989), Approximation by superpositions of a sigmoidal function',
    'Hornik (1991), Approximation capabilities of multilayer feedforward networks',
  ],
})
