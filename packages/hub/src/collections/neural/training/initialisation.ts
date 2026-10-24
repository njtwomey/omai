import { initialisationStudy as study } from 'whitebox-engine/methods/neural/training'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type ViewSlot } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

const SCHEMES = ['small', 'xavier', 'kaiming', 'large'] as const
const TITLES = ['N(0, 0.01²)', 'Xavier', 'Kaiming', 'N(0, 1)']

/** How the scale of the initial weights makes a deep network's signals vanish, survive or explode. */
export const initialisation: HubEntry = defineEntry({
  id: 'neural/training/initialisation',
  title: 'Initialisation and depth',
  summary:
    'Too small and the signal vanishes layer by layer, too large and it saturates: Xavier and Kaiming keep it alive.',
  collection: 'neural/training',
  tags: ['neural networks', 'initialisation', 'vanishing gradients'],
  datasets: ['classification2d'],
  params: dict({
    depth: int(1, 12, { default: 6, label: 'hidden layers' }),
    width: int(4, 64, { default: 32 }),
    activation: choice(['tanh', 'relu', 'sigmoid'] as const, { default: 'tanh' }),
    epochs: int(1, 100, { default: 20 }),
  }),
  run: ({ data, params, key }) =>
    study(
      learnDataOf(data),
      {
        schemes: SCHEMES,
        depth: typeof params['depth'] === 'number' ? params['depth'] : 6,
        width: typeof params['width'] === 'number' ? params['width'] : 32,
        activation: (params['activation'] ?? 'tanh') as 'tanh' | 'relu' | 'sigmoid',
        epochs: typeof params['epochs'] === 'number' ? params['epochs'] : 20,
      },
      { key },
    ),
  views: [
    ...SCHEMES.map(
      (_, i): ViewSlot => ({
        record: 'learningCurve',
        index: i,
        title: `${TITLES[i]}: activation spread and gradient size by layer`,
      }),
    ),
    { record: 'searchTrials', title: 'Final loss per initialisation', area: 'side' },
    ...SCHEMES.map(
      (_, i): ViewSlot => ({
        record: 'learningCurve',
        index: SCHEMES.length + i,
        title: `${TITLES[i]}: loss per epoch`,
        area: 'below',
      }),
    ),
  ],
  methods: [{ file: 'methods/neural/training/initialisation.ts', name: 'initialisationStudy' }],
  references: [
    'Glorot and Bengio (2010), Understanding the difficulty of training deep feedforward neural networks',
    'He, Zhang, Ren and Sun (2015), Delving deep into rectifiers',
  ],
})
