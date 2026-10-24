import { gammaPoissonModel } from 'whitebox-engine/methods/models'
import { poisson } from 'whitebox-engine/random'
import { dict, int, real } from 'whitebox-engine/space'
import { toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** The gamma-Poisson model: an event rate from counts, with its exact posterior. */
export const gammaPoisson: HubEntry = defineEntry({
  id: 'probabilistic/conjugate/gamma-poisson',
  title: 'Gamma-Poisson model',
  summary: 'Count events per interval: a Gamma prior on the rate becomes a Gamma posterior.',
  collection: 'probabilistic/conjugate',
  tags: ['Bayesian inference', 'conjugate prior', 'exact posterior', 'counts'],
  datasets: [],
  params: dict({
    rate: real(0.1, 30, { default: 4, label: 'the true rate' }),
    intervals: int(1, 500, {
      label: 'intervals',
      description: 'The number of intervals in which events are counted.',
      default: 30,
    }),
    shape: real(0.1, 50, { log: true, default: 2, label: 'prior shape' }),
    priorRate: real(0.01, 10, { log: true, default: 0.5, label: 'prior rate' }),
  }),
  run: ({ params, key }) => {
    const counts = Array.from(toFlat(poisson(key, params['rate'] as number, [params['intervals'] as number])))
    return gammaPoissonModel(counts, { shape: params['shape'] as number, rate: params['priorRate'] as number })
  },
  views: [
    { record: 'learningCurve', title: 'Posterior mean (± sd) and the sample mean' },
    { record: 'posteriorGrid', title: 'Posterior density (line: the prior)', area: 'side' },
    { record: 'coefficientTable', title: 'Prior and posterior', area: 'below' },
  ],
  methods: [{ file: 'methods/models/conjugate.ts', name: 'gammaPoissonModel' }],
})
