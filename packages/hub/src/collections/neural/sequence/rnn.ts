import { generate } from 'whitebox-engine/data'
import { parityTask } from 'whitebox-engine/datasets'
import { type RecurrentCell, recurrentTagger } from 'whitebox-engine/methods/neural/sequence'
import { choice, dict, int, type SpaceValue } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

const num = (v: SpaceValue | undefined, fallback: number): number => (typeof v === 'number' ? v : fallback)

/** A recurrent network learning the parity of every prefix of a bit string. */
export const rnn: HubEntry = defineEntry({
  id: 'neural/sequence/rnn',
  title: 'A recurrent network learns parity',
  summary: 'An RNN, GRU or LSTM reads bits one at a time and must carry their running parity in its hidden state.',
  collection: 'neural/sequence',
  tags: ['neural networks', 'recurrent', 'sequences', 'GRU', 'LSTM'],
  datasets: [],
  params: dict({
    cell: choice(['rnn', 'gru', 'lstm'] as const, { default: 'rnn' }),
    task: choice(['running parity', 'parity'] as const, { default: 'running parity' }),
    length: int(2, 16, { default: 8, label: 'bits per sequence' }),
    sequences: int(40, 1000, { log: true, default: 160 }),
    hidden: int(2, 64, { log: true, default: 16, label: 'hidden units' }),
    epochs: int(5, 400, { log: true, default: 80 }),
  }),
  run: ({ params, key, progress }) => {
    const length = num(params['length'], 8)
    const { data } = generate(parityTask, {
      key,
      knobs: {
        n: num(params['sequences'], 160),
        minLength: length,
        maxLength: length,
        running: params['task'] !== 'parity',
      },
    })
    return recurrentTagger(
      data,
      {
        cell: (params['cell'] ?? 'rnn') as RecurrentCell,
        hidden: num(params['hidden'], 16),
        epochs: num(params['epochs'], 80),
        ...(progress === undefined ? {} : { onProgress: progress }),
      },
      { key },
    )
  },
  views: [
    { record: 'decisionSurface', title: 'Hidden states of one sequence (units by position), step by step' },
    { record: 'decisionSurface', index: 1, title: 'Its input bits, target parities and predictions', area: 'side' },
    { record: 'learningCurve', index: 2, title: 'Accuracy by position', area: 'side' },
    { record: 'learningCurve', title: 'Loss per epoch', area: 'below' },
    { record: 'learningCurve', index: 1, title: 'Token accuracy per epoch', area: 'below' },
  ],
  methods: [{ file: 'methods/neural/sequence/rnn.ts', name: 'recurrentTagger' }],
  references: [
    'Elman (1990), Finding structure in time',
    'Cho et al. (2014), Learning phrase representations using RNN encoder-decoder',
    'Hochreiter and Schmidhuber (1997), Long short-term memory',
  ],
})
