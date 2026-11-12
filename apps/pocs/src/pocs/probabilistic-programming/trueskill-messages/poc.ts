import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'TrueSkill: messages on the factor graph',
  summary:
    'Three players rated from the games they play. Expectation propagation passes a message along each edge of the factor graph, one game at a time, and every player’s belief narrows as games are added. Choose how many games have been played, play the messages back, and check the ratings against NUTS.',
  tags: [
    'trueskill',
    'rating',
    'ranking',
    'expectation propagation',
    'ep',
    'message passing',
    'factor graph',
    'probit',
    'nuts',
  ],
  added: '2026-11-10',
}
