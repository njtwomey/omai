import { WhiteboxError } from 'whitebox-engine/base'
import type { TimeSeriesData } from 'whitebox-engine/data'
import { gaussianHMM as gaussianHmmEstimator } from 'whitebox-engine/methods/models'
import { asTensor, fromStorage, toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

const inner = fitEstimator(gaussianHmmEstimator({ nComponents: 3 }))

/** A Gaussian HMM (hmmlearn.hmm.GaussianHMM) segmenting a time series into regimes. */
export const gaussianHmm: HubEntry = defineEntry({
  id: 'probabilistic/hmm/gaussian-hmm',
  title: 'Gaussian hidden Markov model',
  summary: 'A series switches between hidden regimes, each with its own Gaussian; Baum-Welch learns them.',
  collection: 'probabilistic/hmm',
  tags: ['hidden Markov model', 'Baum-Welch', 'regimes', 'time series', 'segmentation'],
  datasets: ['timeSeries'],
  dataset: 'changepointSeries',
  params: inner.params,
  run: (input) => {
    const series = input.data as TimeSeriesData | undefined
    if (series?.kind !== 'timeseries') throw new WhiteboxError('value', 'gaussian-hmm: needs a time series')
    const values = asTensor(series.values)
    const [n = 0, d = 1] = values.shape
    // Missing steps carry the last observed value forward.
    const flat = Array.from(toFlat(values))
    for (let i = d; i < flat.length; i++) if (!Number.isFinite(flat[i])) flat[i] = flat[i - d] ?? 0
    const x = fromStorage(
      Float64Array.from(flat, (v) => (Number.isFinite(v) ? v : 0)),
      [n, d],
      'float64',
    )
    return inner.run({ ...input, data: { x } as never })
  },
  views: [
    { record: 'stateSequence', title: 'State posteriors over time, with the Viterbi path' },
    { record: 'mixtureSteps', title: 'Baum-Welch iterations', area: 'side' },
    { record: 'coefficientTable', title: 'Start and transition probabilities', area: 'below' },
  ],
  methods: [{ file: 'methods/models/hmm.ts', name: 'gaussianHMM' }],
})
