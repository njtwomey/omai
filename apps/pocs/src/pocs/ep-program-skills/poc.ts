import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Program: who is better: a tournament with unknown skills',
  summary:
    'A round-robin tournament of players with latent skills, each game won by a probit on the difference of two skills (the TrueSkill performance model), with a hierarchical gamma prior on the spread of the skills. Run by EP, it recovers the skills and their ordering with bands, against the truth and a long NUTS run, with one pass of TrueSkill from the library beside it.',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'trueskill',
    'ranking',
    'tournament',
    'probit',
    'hierarchical',
    'gamma precision',
    'truth recovery',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
