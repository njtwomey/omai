import { betaBinomialModel } from 'whitebox-engine/methods/models'
import { bernoulli } from 'whitebox-engine/random'
import { dict, int, real } from 'whitebox-engine/space'
import { toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** The beta-binomial model: a coin's bias, from flips, with its exact posterior. */
export const betaBinomial: HubEntry = defineEntry({
  id: 'probabilistic/conjugate/beta-binomial',
  title: 'Beta-binomial model',
  summary: 'Flip a coin of unknown bias: a Beta prior becomes a Beta posterior that narrows with every flip.',
  collection: 'probabilistic/conjugate',
  tags: ['Bayesian inference', 'conjugate prior', 'exact posterior', 'coin flips'],
  datasets: [],
  params: dict({
    bias: real(0, 1, { default: 0.7, label: 'the coin’s true bias' }),
    flips: int(1, 500, { default: 50 }),
    a: real(0.1, 50, { log: true, default: 2, label: 'prior a (pseudo-heads)' }),
    b: real(0.1, 50, { log: true, default: 2, label: 'prior b (pseudo-tails)' }),
  }),
  run: ({ params, key }) => {
    const flips = toFlat(bernoulli(key, params['bias'] as number, [params['flips'] as number])).map(Number)
    return betaBinomialModel(flips, { a: params['a'] as number, b: params['b'] as number })
  },
  views: [
    { record: 'learningCurve', title: 'Posterior mean (± sd) and the frequency as flips arrive' },
    { record: 'posteriorGrid', title: 'Posterior density (line: the prior)', area: 'side' },
    { record: 'coefficientTable', title: 'Prior and posterior', area: 'below' },
  ],
  methods: [{ file: 'methods/models/conjugate.ts', name: 'betaBinomialModel' }],
})
