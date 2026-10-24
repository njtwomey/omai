import { casinoStudy } from 'whitebox-engine/methods/models'
import { bool, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** The occasionally dishonest casino: which die was it? An HMM decodes the dice from the rolls alone. */
export const dishonestCasino: HubEntry = defineEntry({
  id: 'probabilistic/hmm/dishonest-casino',
  title: 'The dishonest casino',
  summary: 'A casino swaps in a loaded die now and then; a hidden Markov model finds when, from the rolls alone.',
  collection: 'probabilistic/hmm',
  tags: ['hidden Markov model', 'Baum-Welch', 'Viterbi', 'forward-backward', 'sequences'],
  datasets: [],
  params: dict({
    rolls: int(50, 2000, { default: 400, label: 'rolls watched' }),
    learn: bool({ default: true, label: 'learn the model by Baum-Welch' }),
    nIter: int(1, 200, { default: 100, label: 'Baum-Welch iterations' }),
  }),
  run: ({ params, key }) =>
    casinoStudy({
      rolls: params['rolls'] as number,
      learn: params['learn'] as boolean,
      nIter: params['nIter'] as number,
      key,
    }),
  views: [
    {
      record: 'stateSequence',
      title: 'The rolls, P(fair) and P(loaded) at each roll, the Viterbi path and the true dice',
    },
    { record: 'mixtureSteps', title: 'Baum-Welch iterations', area: 'side' },
    { record: 'coefficientTable', title: 'Start and switching probabilities', area: 'side' },
  ],
  methods: [
    { file: 'methods/models/hmm.ts', name: 'casinoStudy' },
    { file: 'methods/models/hmm.ts', name: 'categoricalHMM' },
  ],
  references: ['Durbin, Eddy, Krogh and Mitchison (1998), Biological sequence analysis, chapter 3'],
})
