import { realNvp as realNvpMethod } from 'whitebox-engine/methods/generative'
import { dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'
import { GENERATIVE_2D, num } from '../_shared.ts'

/** RealNVP on a two-dimensional dataset: its density, samples and the base grid warped into the data, step by step. */
export const realNvp: HubEntry = defineEntry({
  id: 'generative/flows/real-nvp',
  title: 'RealNVP normalising flow',
  summary:
    'Affine coupling layers bend a standard normal into the data: watch the density, the samples and the warped grid sharpen step by step.',
  collection: 'generative/flows',
  tags: ['generative models', 'normalising flows', 'density estimation', 'maximum likelihood'],
  ...GENERATIVE_2D,
  params: dict({
    layers: int(2, 12, { default: 6, label: 'coupling layers' }),
    hidden: int(4, 64, { default: 24, label: 'conditioner width' }),
    steps: int(50, 2000, { log: true, default: 400 }),
    learningRate: real(1e-4, 0.05, { log: true, default: 0.01, label: 'learning rate' }),
  }),
  run: ({ data, params, key, progress }) =>
    realNvpMethod(
      learnDataOf(data),
      {
        layers: num(params['layers'], 6),
        hidden: num(params['hidden'], 24),
        steps: num(params['steps'], 400),
        learningRate: num(params['learningRate'], 0.01),
        ...(progress === undefined ? {} : { onProgress: progress }),
      },
      { key },
    ),
  views: [
    { record: 'decisionSurface', view: 'DecisionSurfaceView', title: 'Learned density, step by step' },
    { record: 'embeddingSteps', view: 'EmbeddingPlayer', title: 'Samples from fixed base noise', area: 'side' },
    {
      record: 'embeddingSteps',
      index: 1,
      view: 'EmbeddingPlayer',
      props: () => ({ graph: true }),
      title: 'The base space’s grid, pushed into the data space',
      area: 'below',
    },
    {
      record: 'embeddingSteps',
      index: 2,
      view: 'EmbeddingPlayer',
      title: 'The data carried layer by layer to the Gaussian',
      area: 'below',
    },
    { record: 'learningCurve', title: 'Negative log likelihood per step', area: 'below' },
  ],
  methods: [{ file: 'methods/generative/flows.ts', name: 'realNvp' }],
  references: ['Dinh, Sohl-Dickstein and Bengio (2017), Density estimation using Real NVP'],
})
