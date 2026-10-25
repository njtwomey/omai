/**
 * Hub entry for a variational autoencoder with a two-dimensional latent space trained on the 8 by 8 handwritten digits,
 * showing the latent space, reconstructions and samples as it trains.
 */

import { digits } from 'whitebox-engine/datasets'
import type { LearnData } from 'whitebox-engine/learn'
import { vae as vaeMethod } from 'whitebox-engine/methods/generative'
import { dict, int, real } from 'whitebox-engine/space'
import { div } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { num } from '../_shared.ts'

// The 8 by 8 digits with grey levels scaled to [0, 1], for a Bernoulli decoder.
let cached: LearnData | undefined
function scaledDigits(): LearnData {
  if (cached === undefined) {
    const d = digits()
    cached = { x: div(d.x, 16), y: d.y }
  }
  return cached
}

/** A variational autoencoder on the 8 by 8 digits: a two-dimensional latent space, reconstructions and samples. */
export const vaeDigits: HubEntry = defineEntry({
  id: 'generative/autoencoders/vae-digits',
  title: 'Variational autoencoder on digits',
  summary:
    'A VAE with a two-dimensional latent space learns the 8 by 8 digits: the classes separate in the latent space as reconstructions and samples sharpen.',
  collection: 'generative/autoencoders',
  tags: ['generative models', 'variational autoencoders', 'images', 'digits'],
  datasets: [],
  params: dict({
    hidden: int(8, 128, { log: true, default: 32, label: 'hidden width' }),
    beta: real(0.05, 10, { log: true, default: 1, label: 'β (KL weight)' }),
    steps: int(50, 3000, { log: true, default: 300 }),
    learningRate: real(1e-4, 0.05, { log: true, default: 0.01, label: 'learning rate' }),
  }),
  run: ({ params, key, progress }) =>
    vaeMethod(
      scaledDigits(),
      {
        likelihood: 'bernoulli',
        latent: 2,
        hidden: num(params['hidden'], 32),
        activation: 'relu',
        beta: num(params['beta'], 1),
        steps: num(params['steps'], 300),
        learningRate: num(params['learningRate'], 0.01),
        imageShape: [8, 8],
        images: 10,
        samples: 10,
        batchSize: 32,
        frames: 25,
        ...(progress === undefined ? {} : { onProgress: progress }),
      },
      { key },
    ),
  views: [
    { record: 'embeddingSteps', view: 'EmbeddingPlayer', title: 'Latent space of the digits, coloured by digit' },
    { record: 'imageSteps', title: 'Digits, their reconstructions and samples from the prior', area: 'below' },
    { record: 'learningCurve', title: 'Reconstruction against KL, per step', area: 'below' },
    { record: 'learningCurve', index: 1, title: 'Negative ELBO per step', area: 'below' },
  ],
  methods: [{ file: 'methods/generative/autoencoders.ts', name: 'vae' }],
  references: ['Kingma and Welling (2014), Auto-encoding variational Bayes'],
})
