import { WhiteboxError } from 'whitebox-engine/base'
import type { TimeSeriesData } from 'whitebox-engine/data'
import { linearGaussianStateSpace, type StructuralModel } from 'whitebox-engine/methods/models'
import { bool, choice, dict, int } from 'whitebox-engine/space'
import { asTensor, toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** A local-level (or local-trend) model: a hidden random walk seen through noise, smoothed by Kalman and RTS. */
export const localLevel: HubEntry = defineEntry({
  id: 'probabilistic/latent-gaussian/local-level',
  title: 'Local level model',
  summary: 'A hidden level wanders as a random walk and is seen through noise; Kalman smoothing recovers it.',
  collection: 'probabilistic/latent-gaussian',
  tags: ['state space', 'Kalman filter', 'smoothing', 'EM', 'time series'],
  datasets: ['timeSeries'],
  params: dict({
    model: choice(['local-level', 'local-linear-trend'], { default: 'local-level' }),
    learn: bool({ default: true, label: 'learn the noise variances by EM' }),
    maxIter: int(1, 200, { default: 20, label: 'EM iterations' }),
  }),
  run: ({ data, params }) => {
    const series = data as TimeSeriesData | undefined
    if (series?.kind !== 'timeseries') throw new WhiteboxError('value', 'local-level: needs a time series')
    const values = asTensor(series.values)
    const d = values.shape[1] ?? 1
    const first = toFlat(values).filter((_, i) => i % d === 0)
    return linearGaussianStateSpace(first, {
      model: params['model'] as StructuralModel,
      learn: params['learn'] as boolean,
      maxIter: params['maxIter'] as number,
    })
  },
  views: [
    { record: 'posteriorGrid', title: 'Smoothed level (±1 and 2 sd) with the observations' },
    { record: 'posteriorGrid', index: 1, title: 'Filtered level', area: 'below' },
    { record: 'coefficientTable', title: 'Noise variances', area: 'side' },
    { record: 'learningCurve', title: 'EM log likelihood', area: 'side' },
  ],
  methods: [{ file: 'methods/models/latent-gaussian.ts', name: 'linearGaussianStateSpace' }],
  references: ['Durbin and Koopman (2012), Time series analysis by state space methods'],
})
