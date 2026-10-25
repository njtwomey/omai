/**
 * The sanity-check entry: a LeNet's gradient maps as its layers are randomised from the top down, and after training
 * on shuffled labels.
 */

import { WhiteboxError } from 'whitebox-engine/base'
import { matrixView } from 'whitebox-engine/learn'
import { sanityChecks as check, type SanityMethod } from 'whitebox-engine/methods/explain'
import { lenet } from 'whitebox-engine/methods/neural/vision'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Sanity checks for saliency maps. */
export const sanityChecks: HubEntry = defineEntry({
  id: 'explainability/sanity-checks',
  title: 'Sanity checks for saliency maps',
  summary: 'Does the map depend on the model? Randomise its layers top-down, or its labels, and watch the map change.',
  collection: 'explainability',
  tags: [
    'sanity checks',
    'model randomisation',
    'label randomisation',
    'saliency',
    'Grad-CAM',
    'evaluation of explanations',
  ],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  maxRows: 600,
  params: dict({
    method: choice(['saliency', 'integratedGradients', 'guidedBackprop', 'gradCam'], {
      default: 'guidedBackprop',
      label: 'map',
    }),
    row: int(0, 599, { default: 3, label: 'image' }),
    epochs: int(1, 10, { default: 3, label: 'training epochs' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const n = matrixView(learn.x, 'sanity-checks').n
    if (n === 0) throw new WhiteboxError('value', 'sanity-checks: no images')
    const epochs = Number(params['epochs'])
    const fitted = lenet({ epochs, snapshots: 2 }).fit(learn, { key, explain: false })
    return check(fitted, learn, Math.min(Number(params['row']), n - 1), {
      method: params['method'] as SanityMethod,
      hyper: { epochs },
      key,
    })
  },
  views: [
    { record: 'valueGrids', title: 'The map as the network is randomised' },
    { record: 'curves', title: 'Similarity to the original map', area: 'side' },
  ],
  methods: [{ file: 'methods/explain/evaluation.ts', name: 'sanityChecks' }],
  references: ['Adebayo et al. (2018), Sanity checks for saliency maps.'],
})
