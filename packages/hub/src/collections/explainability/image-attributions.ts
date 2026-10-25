/**
 * The image attribution entry: occlusion, superpixel ablation and SHAP, RISE and LIME maps of a digit classifier.
 */
import { WhiteboxError } from 'whitebox-engine/base'
import { type Fitted, matrixView } from 'whitebox-engine/learn'
import { randomForestClassifier } from 'whitebox-engine/methods/ensemble'
import { explainImage, type ImageMethod } from 'whitebox-engine/methods/explain'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Perturbation attribution maps over an image. */
export const imageAttributions: HubEntry = defineEntry({
  id: 'explainability/image-attributions',
  title: 'Occlusion, RISE and superpixel SHAP',
  summary:
    'Which pixels a digit classifier relies on, found by hiding parts of the image and watching its probability.',
  collection: 'explainability',
  tags: ['occlusion', 'RISE', 'feature ablation', 'superpixels', 'SHAP', 'LIME', 'images', 'perturbation'],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  maxRows: 1000,
  params: dict({
    method: choice(['occlusion', 'ablation', 'shap', 'rise', 'lime'], { default: 'occlusion', label: 'explainer' }),
    row: int(0, 999, { default: 7, label: 'image' }),
    window: int(1, 4, { default: 2, label: 'window / superpixel side' }),
    segments: choice(['grid', 'slic'], { default: 'grid', label: 'superpixels' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const n = matrixView(learn.x, 'image-attributions').n
    if (n === 0) throw new WhiteboxError('value', 'image-attributions: no images')
    const fitted: Fitted = randomForestClassifier({ nEstimators: 40, maxDepth: 8 }).fit(learn, { key, explain: false })
    const w = Number(params['window'])
    return explainImage(fitted, learn, Math.min(Number(params['row']), n - 1), {
      method: params['method'] as ImageMethod,
      window: w,
      cell: w,
      segments: params['segments'] as 'grid' | 'slic',
      nMasks: 500,
      key,
    })
  },
  views: [{ record: 'featureAttributions', view: 'AttributionImageView', title: 'The input and its attribution map' }],
  methods: [{ file: 'methods/explain/perturbation.ts', name: 'explainImage' }],
  references: [
    'Zeiler and Fergus (2014), Visualizing and understanding convolutional networks.',
    'Petsiuk, Das and Saenko (2018), RISE: randomized input sampling for explanation of black-box models.',
    'Achanta et al. (2012), SLIC superpixels compared to state-of-the-art superpixel methods.',
  ],
})
