import { normalisationStudy as study } from 'whitebox-engine/methods/neural/training'
import { dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type ViewSlot } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

const TITLES = ['No normalisation', 'Batch normalisation', 'Layer normalisation']

/** Batch and layer normalisation against none on the same deep ReLU network. */
export const normalisation: HubEntry = defineEntry({
  id: 'neural/training/normalisation',
  title: 'Batch and layer normalisation',
  summary: 'The same deep ReLU network with no normalisation, batch normalisation and layer normalisation.',
  collection: 'neural/training',
  tags: ['neural networks', 'normalisation', 'batch norm'],
  datasets: ['classification2d'],
  params: dict({
    depth: int(1, 10, { default: 4, label: 'hidden layers' }),
    epochs: int(1, 100, { default: 20 }),
  }),
  run: ({ data, params, key }) =>
    study(
      learnDataOf(data),
      {
        depth: typeof params['depth'] === 'number' ? params['depth'] : 4,
        epochs: typeof params['epochs'] === 'number' ? params['epochs'] : 20,
      },
      { key },
    ),
  views: [
    ...TITLES.map((t, i): ViewSlot => ({ record: 'learningCurve', index: i, title: `${t}: loss per epoch` })),
    { record: 'searchTrials', title: 'Final loss', area: 'side' },
    ...TITLES.map((t, i): ViewSlot => ({ record: 'decisionSurface', index: i, title: t, area: 'below' })),
  ],
  methods: [{ file: 'methods/neural/training/initialisation.ts', name: 'normalisationStudy' }],
  references: ['Ioffe and Szegedy (2015), Batch normalization', 'Ba, Kiros and Hinton (2016), Layer normalization'],
})
