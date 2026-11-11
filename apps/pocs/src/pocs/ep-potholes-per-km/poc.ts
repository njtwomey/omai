import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'EP atom: potholes on roads of different lengths',
  summary:
    'A Poisson count whose rate is a per-km rate times the length of each road section (an exposure). EP against the exact gamma posterior, the sections drawn as length against count.',
  tags: ['expectation propagation', 'ep', 'atom', 'poisson', 'exposure', 'gamma', 'counts', 'conjugate'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
