import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'A mixture whose groups have a mean and a precision of their own',
  summary:
    'Several pieces of the model language composed in one small setup: two groups in two directions, each direction of each group with a precision of its own and a centre as wide as that precision allows (the conjugate normal-gamma pair), and a gate on the group of each row. The compiler makes each pair one normal-gamma variable, so the messages are closed form. The rows are simulated from two clusters. EP recovers the centres and the precisions, and reports for each row the probability that it is in the second group, against the exact posterior with every assignment of the rows enumerated.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composition',
    'gates',
    'mixture',
    'gaussian mixture',
    'normal-gamma',
    'conjugate pair',
    'precision',
    'label',
    'exact reference',
  ],
  added: '2026-11-12',
}
