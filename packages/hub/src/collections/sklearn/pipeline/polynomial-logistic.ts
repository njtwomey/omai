import { type NamedStep, pipeline } from 'whitebox-engine/learn'
import { polynomialFeatures, standardScaler } from 'whitebox-engine/learn/preprocessing'
import { pca } from 'whitebox-engine/methods/decomposition'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { bool, dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

/** The number of polynomial features of d inputs up to a degree, without the bias column. */
function featuresAt(d: number, degree: number): number {
  let count = 1
  for (let k = 1; k <= degree; k++) count = (count * (d + k)) / k
  return Math.round(count) - 1
}

/**
 * A pipeline (sklearn.pipeline.Pipeline) of polynomial features, standardisation, PCA and a logistic regression on two
 * features: each stage's output in the form that suits it, and the whole pipeline's probability over the raw inputs.
 */
export const polynomialLogistic: HubEntry = defineEntry({
  id: 'sklearn/pipeline/polynomial-logistic',
  title: 'Polynomial features, scaling, PCA and logistic regression',
  summary: 'A pipeline stage by stage: features made, rescaled, projected, then classified over the raw inputs.',
  collection: 'sklearn/pipeline',
  tags: ['pipeline', 'preprocessing', 'classification', 'dimensionality reduction'],
  datasets: ['classification2d'],
  params: dict({
    degree: int(1, 6, { default: 3, label: 'polynomial degree' }),
    standardise: bool({ default: true, label: 'standardise' }),
    components: int(0, 12, { default: 4, label: 'PCA components (0 for none)' }),
    C: real(1e-3, 1e2, { log: true, default: 1, label: 'inverse penalty C' }),
  }),
  run: ({ data, params }) => {
    const learn = learnDataOf(data)
    const degree = Number(params['degree'] ?? 3)
    const d = (data as { readonly x?: { readonly shape?: readonly number[] } } | undefined)?.x?.shape?.[1] ?? 2
    const k = Math.min(Number(params['components'] ?? 4), featuresAt(d, degree))
    const steps: NamedStep[] = [
      ['poly', polynomialFeatures({ degree, includeBias: false })],
      ...(params['standardise'] === false ? [] : [['scale', standardScaler()] as const]),
      ...(k > 0 ? [['pca', pca({ nComponents: k })] as const] : []),
      ['logistic', logisticRegression({ C: Number(params['C'] ?? 1) })],
    ]
    return pipeline(steps).fit(learn)
  },
  views: [{ record: 'pipelineStages', title: 'Stages' }],
  methods: [
    { file: 'lib/learn/compose.ts', name: 'pipeline' },
    { file: 'methods/decomposition/pca.ts', name: 'pca' },
    { file: 'methods/linear/classification.ts', name: 'logisticRegression' },
  ],
})
