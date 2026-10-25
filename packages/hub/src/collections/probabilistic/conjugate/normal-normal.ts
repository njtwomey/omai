/**
 * Hub entry for the normal-normal model, a mean learned from noisy measurements of known spread with its exact normal
 * posterior.
 */

import { normalNormalModel } from 'whitebox-engine/methods/models'
import { normal } from 'whitebox-engine/random'
import { dict, int, real } from 'whitebox-engine/space'
import { toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** The normal-normal model: a mean from noisy measurements of known spread, with its exact posterior. */
export const normalNormal: HubEntry = defineEntry({
  id: 'probabilistic/conjugate/normal-normal',
  title: 'Normal-normal model',
  summary: 'Measure a quantity with known noise: the posterior mean is a precision-weighted blend of prior and data.',
  collection: 'probabilistic/conjugate',
  tags: ['Bayesian inference', 'conjugate prior', 'exact posterior', 'shrinkage'],
  datasets: [],
  params: dict({
    mean: real(-10, 10, { default: 3, label: 'the true mean' }),
    sigma: real(0.1, 10, { log: true, default: 2, label: 'measurement sd (known)' }),
    measurements: int(1, 500, {
      label: 'measurements',
      description: 'The number of noisy measurements observed.',
      default: 20,
    }),
    priorMean: real(-10, 10, { default: 0, label: 'prior mean' }),
    priorSd: real(0.1, 20, { log: true, default: 1, label: 'prior sd' }),
  }),
  run: ({ params, key }) => {
    const sigma = params['sigma'] as number
    const z = toFlat(normal(key, [params['measurements'] as number]))
    const xs = Array.from(z, (v) => (params['mean'] as number) + sigma * v)
    return normalNormalModel(xs, { mean: params['priorMean'] as number, sd: params['priorSd'] as number, sigma })
  },
  views: [
    { record: 'learningCurve', title: 'Posterior mean (± sd) and the sample mean' },
    { record: 'posteriorGrid', title: 'Prior (dashed) and posterior densities', area: 'side' },
    { record: 'coefficientTable', title: 'Prior and posterior', area: 'below' },
  ],
  methods: [{ file: 'methods/models/conjugate.ts', name: 'normalNormalModel' }],
})
