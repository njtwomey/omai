/**
 * Hub entry for a variational autoencoder on two-dimensional toy data, showing its latent space, reconstructions and
 * samples from the prior step by step.
 */

import { vae as vaeMethod } from 'mdml-engine/methods/generative'
import { dict, int, real } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'
import { GENERATIVE_2D, num } from '../_shared.ts'

/** A variational autoencoder on a two-dimensional dataset: its latent space, reconstructions and samples per step. */
export const vae: HubEntry = defineEntry({
  id: 'generative/autoencoders/vae',
  title: 'Variational autoencoder',
  summary:
    'An encoder and decoder trained on the ELBO: watch the latent space organise, reconstructions tighten and samples from the prior fill the data.',
  collection: 'generative/autoencoders',
  tags: ['generative models', 'variational autoencoders', 'ELBO', 'latent variables'],
  ...GENERATIVE_2D,
  params: dict({
    latent: int(1, 4, { default: 2, label: 'latent dimensions' }),
    hidden: int(4, 64, { default: 32, label: 'hidden width' }),
    beta: real(0.05, 10, { log: true, default: 1, label: 'β (KL weight)' }),
    steps: int(50, 3000, { log: true, default: 600 }),
    learningRate: real(1e-4, 0.05, { log: true, default: 0.01, label: 'learning rate' }),
  }),
  run: ({ data, params, key, progress }) =>
    vaeMethod(
      learnDataOf(data),
      {
        latent: num(params['latent'], 2),
        hidden: num(params['hidden'], 32),
        beta: num(params['beta'], 1),
        steps: num(params['steps'], 600),
        learningRate: num(params['learningRate'], 0.01),
        ...(progress === undefined ? {} : { onProgress: progress }),
      },
      { key },
    ),
  views: [
    { record: 'embeddingSteps', view: 'EmbeddingPlayer', title: 'Latent space: the encoder’s means of the data' },
    {
      record: 'embeddingSteps',
      index: 2,
      view: 'EmbeddingPlayer',
      title: 'Samples from the prior, decoded',
      area: 'side',
    },
    {
      record: 'embeddingSteps',
      index: 1,
      view: 'EmbeddingPlayer',
      title: 'Reconstructions of the data',
      area: 'below',
    },
    { record: 'learningCurve', title: 'Reconstruction against KL, per step', area: 'below' },
    { record: 'learningCurve', index: 1, title: 'Negative ELBO per step', area: 'below' },
  ],
  methods: [{ file: 'methods/generative/autoencoders.ts', name: 'vae' }],
  references: ['Kingma and Welling (2014), Auto-encoding variational Bayes'],
})
