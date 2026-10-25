/**
 * Hub entry for a LeNet convolutional network learning the 8 by 8 handwritten digits, its feature maps tapped as it
 * trains.
 */

import { take } from 'mdml-engine/data'
import { digits } from 'mdml-engine/datasets'
import type { Exposition } from 'mdml-engine/exposition'
import type { LearnData } from 'mdml-engine/learn'
import { lenet as lenetMethod } from 'mdml-engine/methods/neural/vision'
import { choice, dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

const num = (v: SpaceValue | undefined, fallback: number): number => (typeof v === 'number' ? v : fallback)

/** A LeNet on the 8 by 8 handwritten digits, its feature maps tapped as it trains. */
export const lenet: HubEntry = defineEntry({
  id: 'neural/vision/lenet',
  title: 'LeNet on handwritten digits',
  summary: 'Two convolutions and a classifier learn the 8 by 8 digits in seconds; watch the feature maps form.',
  collection: 'neural/vision',
  tags: ['neural networks', 'convolution', 'images', 'classification'],
  datasets: [],
  params: dict({
    rows: int(100, 1797, { default: 800, label: 'digits trained on' }),
    epochs: int(1, 30, { default: 5 }),
    learningRate: real(1e-4, 0.1, { log: true, default: 0.02, label: 'learning rate' }),
    activation: choice(['relu', 'tanh'] as const, { default: 'relu' }),
    pool: choice(['max', 'avg'] as const, { default: 'max', label: 'pooling' }),
  }),
  run: ({ params, key, progress }) => {
    const all = digits()
    const n = Math.min(num(params['rows'], 800), 1797)
    const data = take(
      all,
      Array.from({ length: n }, (_, i) => i),
    ) as unknown as LearnData
    const fitted = lenetMethod({
      epochs: num(params['epochs'], 5),
      learningRate: num(params['learningRate'], 0.02),
      activation: (params['activation'] ?? 'relu') as 'relu' | 'tanh',
      pool: (params['pool'] ?? 'max') as 'max' | 'avg',
      ...(progress === undefined ? {} : { onProgress: progress }),
    }).fit(data, { key })
    return { ...fitted, exposition: fitted.exposition as readonly Exposition[] }
  },
  views: [
    {
      record: 'valueGrids',
      index: 1,
      title: 'The first convolution’s six feature maps of one digit, step by step',
    },
    { record: 'valueGrids', index: 0, title: 'The digit tapped', area: 'side' },
    { record: 'valueGrids', index: 3, title: 'The first convolution’s filters', area: 'side' },
    { record: 'valueGrids', index: 2, title: 'The second convolution’s twelve maps', area: 'side' },
    { record: 'learningCurve', title: 'Loss per epoch', area: 'below' },
    { record: 'learningCurve', index: 1, title: 'Accuracy per epoch', area: 'below' },
    { record: 'confusionMatrix', title: 'Validation digits: true against predicted', area: 'below' },
    { record: 'coefficientTable', title: 'Validation accuracy per digit', area: 'below' },
    { record: 'diagram', title: 'The architecture', area: 'below' },
  ],
  methods: [{ file: 'methods/neural/vision/lenet.ts', name: 'lenet' }],
  references: ['LeCun, Bottou, Bengio and Haffner (1998), Gradient-based learning applied to document recognition'],
})
