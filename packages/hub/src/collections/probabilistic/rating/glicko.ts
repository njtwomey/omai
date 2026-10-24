import { glickoRatings } from 'whitebox-engine/methods/models'
import { dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { comparisonsOf } from './_rating.ts'

/** Glicko ratings: Elo with a rating deviation that shrinks as a player plays and grows while they sit out. */
export const glicko: HubEntry = defineEntry({
  id: 'probabilistic/rating/glicko',
  title: 'Glicko ratings',
  summary: 'Ratings with a deviation: new or idle players move a lot, established ones little.',
  collection: 'probabilistic/rating',
  tags: ['ratings', 'paired comparisons', 'uncertainty', 'Bayesian'],
  datasets: ['comparisons'],
  params: dict({
    deviation: real(30, 500, { default: 350, label: 'starting deviation' }),
    c: real(0, 100, { default: 0, label: 'deviation growth per game sat out' }),
  }),
  run: ({ data, params }) =>
    glickoRatings(comparisonsOf(data), { deviation: params['deviation'] as number, c: params['c'] as number }),
  views: [
    { record: 'curves', title: 'The top three players’ ratings with two deviations, game by game' },
    { record: 'coefficientTable', title: 'Final ratings with deviations', area: 'side' },
    { record: 'learningCurve', title: 'Prediction log loss and accuracy', area: 'below' },
  ],
  methods: [{ file: 'methods/models/rating.ts', name: 'glickoRatings' }],
  references: ['Glickman (1999), Parameter estimation in large dynamic paired comparison experiments'],
})
