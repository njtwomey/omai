import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'A softmax over many classes with coefficients that differ between two groups',
  summary:
    'Three pieces composed in one small model: a softmax over many classes (eight by default, past the six that the old product grid reached, through the auxiliary-variable rule), scores linear in a covariate, and one coefficient for each of two groups drawn around a shared mean with a learned, gamma-distributed precision. The data are simulated from stated coefficients; EP recovers the two coefficients, the shared mean and the precision, against a three-dimensional grid with the mean integrated out. The knobs set the number of classes, the rows in each group and the true difference between the groups.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'softmax',
    'multiclass',
    'many classes',
    'hierarchy',
    'groups',
    'shared mean',
    'learned precision',
    'gaussian-gamma-precision',
    'parameters learned',
  ],
  added: '2026-11-12',
  group: 'composed-atoms',
}
