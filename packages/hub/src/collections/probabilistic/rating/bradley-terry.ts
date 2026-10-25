/**
 * Hub entry for Bradley-Terry strengths, maximum-likelihood ratings fitted to all the games at once by Hunter's MM
 * algorithm, on simulated tournaments.
 */

import { bradleyTerry as bradleyTerryRatings } from 'mdml-engine/methods/models'
import { dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { comparisonsOf } from './_rating.ts'

/** Bradley-Terry strengths: the maximum-likelihood ratings of all the games at once, by Hunter's MM algorithm. */
export const bradleyTerry: HubEntry = defineEntry({
  id: 'probabilistic/rating/bradley-terry',
  title: 'Bradley-Terry model',
  summary: 'P(i beats j) = s_i / (s_i + s_j): strengths fitted to every game at once, the static answer Elo tracks.',
  collection: 'probabilistic/rating',
  tags: ['ratings', 'paired comparisons', 'maximum likelihood', 'MM algorithm'],
  datasets: ['comparisons'],
  dataset: 'roundRobin',
  params: dict({ maxIter: int(1, 500, { default: 200, label: 'MM iterations' }) }),
  run: ({ data, params }) => bradleyTerryRatings(comparisonsOf(data), { maxIter: params['maxIter'] as number }),
  views: [
    { record: 'coefficientTable', title: 'Strengths on the Elo scale' },
    { record: 'learningCurve', title: 'Log likelihood per MM iteration', area: 'side' },
  ],
  methods: [{ file: 'methods/models/rating.ts', name: 'bradleyTerry' }],
  references: ['Hunter (2004), MM algorithms for generalized Bradley-Terry models'],
})
