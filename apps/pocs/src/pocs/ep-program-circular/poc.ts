import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Program: circular times without a circular distribution',
  summary:
    'Clock times are periodic: 23:50 and 00:10 are twenty minutes apart. A mixture of ordinary normals can fit them without a circular distribution in two ways that are the same arithmetic: the paper’s, which copies each component’s mean a day early and late (shift the means), and the post’s, which copies each reading (shift the data) and scores the copies together. The program writes both as models with a discrete copy index and runs expectation propagation, beside a model that ignores the circle and one that makes the copies extra rows (the wrong way: sd 21.7 hours for a truth of 3).',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'circular statistics',
    'periodic data',
    'wrapped normal',
    'mixture',
    'gates',
    'clock time',
    'data shift',
    'truth recovery',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
