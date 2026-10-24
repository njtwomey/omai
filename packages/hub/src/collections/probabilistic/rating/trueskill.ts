import { trueSkillRatings } from 'whitebox-engine/methods/models'
import { dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { comparisonsOf } from './_rating.ts'

/** TrueSkill: Gaussian beliefs over skill, updated by moment matching after each game. */
export const trueskill: HubEntry = defineEntry({
  id: 'probabilistic/rating/trueskill',
  title: 'TrueSkill',
  summary: 'Each skill is a Gaussian belief; a result truncates the performance gap and the beliefs are re-fitted.',
  collection: 'probabilistic/rating',
  tags: ['ratings', 'paired comparisons', 'expectation propagation', 'Bayesian'],
  datasets: ['comparisons'],
  params: dict({
    beta: real(0.5, 20, { default: 25 / 6, label: 'performance noise beta' }),
    tau: real(0, 2, { default: 25 / 300, label: 'skill drift tau' }),
    drawProbability: real(0, 0.5, { default: 0.1, label: 'draw probability' }),
  }),
  run: ({ data, params }) =>
    trueSkillRatings(comparisonsOf(data), {
      beta: params['beta'] as number,
      tau: params['tau'] as number,
      drawProbability: params['drawProbability'] as number,
    }),
  views: [
    { record: 'curves', title: 'The top three players’ skill beliefs (± 2σ), game by game' },
    { record: 'coefficientTable', title: 'Skill means with sigma', area: 'side' },
    { record: 'learningCurve', title: 'Prediction log loss and accuracy', area: 'below' },
  ],
  methods: [{ file: 'methods/models/rating.ts', name: 'trueSkillRatings' }],
  references: ['Herbrich, Minka and Graepel (2007), TrueSkill: a Bayesian skill rating system'],
})
