import { learningRateFinder as finder } from 'whitebox-engine/methods/neural/training'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

/** Leslie Smith's learning-rate range test. */
export const learningRateFinder: HubEntry = defineEntry({
  id: 'neural/training/learning-rate-finder',
  title: 'Learning-rate finder',
  summary: 'Raise the learning rate step by step until the loss blows up; pick a rate where it falls fastest.',
  collection: 'neural/training',
  tags: ['neural networks', 'optimisation', 'learning rate'],
  datasets: ['classification2d', 'regression1d'],
  params: dict({
    steps: int(20, 300, { default: 100 }),
    optimiser: choice(['sgd', 'momentum', 'adam'] as const, { default: 'sgd' }),
  }),
  run: ({ data, params, key }) =>
    finder(
      learnDataOf(data),
      {
        steps: typeof params['steps'] === 'number' ? params['steps'] : 100,
        optimiser: (params['optimiser'] ?? 'sgd') as 'sgd' | 'momentum' | 'adam',
      },
      { key },
    ),
  views: [{ record: 'learningCurve', title: 'Loss against the learning rate' }],
  methods: [{ file: 'methods/neural/training/optimisers.ts', name: 'learningRateFinder' }],
  references: ['Smith (2017), Cyclical learning rates for training neural networks'],
})
