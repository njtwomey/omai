import { eloRatings } from 'whitebox-engine/methods/models'
import { choice, dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { comparisonsOf } from './_rating.ts'

/** Elo ratings (with the chess conventions): each result moves both ratings by K times the surprise. */
export const elo: HubEntry = defineEntry({
  id: 'probabilistic/rating/elo',
  title: 'Elo ratings',
  summary: 'After each game both players move by K times how surprising the result was.',
  collection: 'probabilistic/rating',
  tags: ['ratings', 'paired comparisons', 'online learning', 'chess'],
  datasets: ['comparisons'],
  params: dict({
    k: choice(['16', '32', '64', 'fide'], { default: '32', label: 'K-factor' }),
    scale: real(100, 800, { default: 400, label: 'points for ten-to-one odds' }),
  }),
  run: ({ data, params }) => {
    const k = params['k'] === 'fide' ? 'fide' : Number(params['k'])
    return eloRatings(comparisonsOf(data), { k, scale: params['scale'] as number })
  },
  views: [
    { record: 'posteriorGrid', title: 'The leader’s rating, game by game' },
    { record: 'coefficientTable', title: 'Final ratings', area: 'side' },
    { record: 'learningCurve', title: 'Prediction log loss and accuracy', area: 'below' },
  ],
  methods: [{ file: 'methods/models/rating.ts', name: 'eloRatings' }],
  references: ['Elo (1978), The rating of chessplayers, past and present'],
})
