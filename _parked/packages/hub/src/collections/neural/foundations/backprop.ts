/**
 * Hub entry for backpropagation worked by hand on a tiny 2-2-1 network learning XOR, checked against reverse-mode
 * autodiff at every step.
 */

import { type BackpropActivation, backpropByHand } from 'mdml-engine/methods/neural/foundations'
import { choice, dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

const options = (p: Readonly<Record<string, SpaceValue>>): Parameters<typeof backpropByHand>[0] => ({
  sizes: [2, typeof p['hidden'] === 'number' ? p['hidden'] : 2, 1],
  activation: (p['activation'] ?? 'logistic') as BackpropActivation,
  steps: typeof p['steps'] === 'number' ? p['steps'] : 8,
  learningRate: typeof p['learningRate'] === 'number' ? p['learningRate'] : 0.5,
})

/** Backpropagation on a tiny network, worked by hand and checked against autodiff at every step. */
export const backprop: HubEntry = defineEntry({
  id: 'neural/foundations/backprop',
  title: 'Backpropagation by hand',
  summary: 'The chain rule unit by unit on a 2-2-1 network learning XOR, beside reverse-mode autodiff.',
  collection: 'neural/foundations',
  tags: ['neural networks', 'backpropagation', 'autodiff', 'chain rule'],
  datasets: [],
  params: dict({
    hidden: int(1, 4, { default: 2, label: 'hidden units' }),
    activation: choice(['logistic', 'tanh', 'relu'] as const, { default: 'logistic' }),
    steps: int(1, 40, { default: 8 }),
    learningRate: real(0.01, 5, { log: true, default: 0.5, label: 'learning rate' }),
  }),
  run: ({ params }) => backpropByHand(options(params)),
  views: [
    { record: 'diagram', title: 'The network, pass by pass: forward values (z, a), the loss, then the deltas (δ)' },
    {
      record: 'coefficientTable',
      title: 'Gradients at the first step: by hand, ± the autodiff disagreement',
      area: 'side',
    },
    { record: 'learningCurve', title: 'Loss per step, and the largest disagreement', area: 'below' },
  ],
  methods: [{ file: 'methods/neural/foundations/backprop.ts', name: 'backpropByHand' }],
})
