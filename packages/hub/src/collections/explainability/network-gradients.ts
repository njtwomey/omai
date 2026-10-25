/**
 * The network gradient entry: saliency, integrated gradients, noise tunnels, GradientSHAP, guided backpropagation and
 * Grad-CAM maps of a LeNet trained on the digits, and its last convolution's conductance per channel.
 */
import { WhiteboxError } from 'whitebox-engine/base'
import { matrixView } from 'whitebox-engine/learn'
import { explainLayer, explainNetwork, type NetworkMethod } from 'whitebox-engine/methods/explain'
import { lenet } from 'whitebox-engine/methods/neural/vision'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Gradient attribution maps of a convolutional network. */
export const networkGradients: HubEntry = defineEntry({
  id: 'explainability/network-gradients',
  title: 'Saliency, integrated gradients and Grad-CAM',
  summary: 'Where a LeNet looks when it reads a digit: gradient maps, their smoothed and guided forms, and Grad-CAM.',
  collection: 'explainability',
  tags: [
    'saliency',
    'integrated gradients',
    'SmoothGrad',
    'VarGrad',
    'GradientSHAP',
    'guided backpropagation',
    'Grad-CAM',
    'layer conductance',
    'neural networks',
  ],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  maxRows: 800,
  params: dict({
    method: choice(
      [
        'saliency',
        'inputXGradient',
        'integratedGradients',
        'smoothGrad',
        'varGrad',
        'gradientShap',
        'guidedBackprop',
        'deconvolution',
        'gradCam',
        'guidedGradCam',
      ],
      { default: 'integratedGradients', label: 'explainer' },
    ),
    row: int(0, 799, { default: 4, label: 'image' }),
    epochs: int(1, 20, { default: 3, label: 'training epochs' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const n = matrixView(learn.x, 'network-gradients').n
    if (n === 0) throw new WhiteboxError('value', 'network-gradients: no images')
    const fitted = lenet({ epochs: Number(params['epochs']), snapshots: 2 }).fit(learn, { key, explain: false })
    const row = Math.min(Number(params['row']), n - 1)
    const map = explainNetwork(fitted, learn, row, { method: params['method'] as NetworkMethod, key })
    const layer = explainLayer(fitted, learn, row)
    return { ...map, exposition: [...map.exposition, ...layer.exposition] }
  },
  views: [
    { record: 'featureAttributions', view: 'AttributionImageView', title: 'The digit and its attribution map' },
    { record: 'valueGrids', title: 'Last convolution: conductance per channel', area: 'side' },
  ],
  methods: [
    { file: 'methods/explain/gradients.ts', name: 'explainNetwork' },
    { file: 'methods/explain/gradients.ts', name: 'explainLayer' },
  ],
  references: [
    'Simonyan, Vedaldi and Zisserman (2014), Deep inside convolutional networks.',
    'Sundararajan, Taly and Yan (2017), Axiomatic attribution for deep networks.',
    'Springenberg et al. (2015), Striving for simplicity: the all convolutional net.',
    'Selvaraju et al. (2017), Grad-CAM: visual explanations from deep networks via gradient-based localization.',
    'Dhamdhere, Sundararajan and Yan (2018), How important is a neuron?',
  ],
})
