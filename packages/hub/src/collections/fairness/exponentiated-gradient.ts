import { type ReductionConstraint, exponentiatedGradient as reduce } from 'whitebox-engine/methods/fairness'
import { choice, dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fairnessDataOf } from './_shared.ts'

/** In-processing: fair classification by reduction to cost-sensitive problems, traced over iterations. */
export const exponentiatedGradient: HubEntry = defineEntry({
  id: 'fairness/exponentiated-gradient',
  title: 'Exponentiated gradient',
  summary: 'A learner and an auditor play a game: the violation falls towards ε while the error rises a little.',
  collection: 'fairness',
  tags: ['fairness', 'in-processing', 'reductions', 'Lagrangian', 'demographic parity'],
  datasets: ['fairnessTabular'],
  maxRows: 1500,
  params: dict({
    constraint: choice(['demographicParity', 'equalisedOdds'], { default: 'demographicParity' }),
    epsilon: real(0, 0.3, { default: 0.02, label: 'ε (allowed violation)' }),
    iterations: int(2, 60, { default: 20 }),
  }),
  run: (input) =>
    reduce(fairnessDataOf(input), {
      constraint: input.params['constraint'] as ReductionConstraint,
      epsilon: input.params['epsilon'] as number,
      iterations: input.params['iterations'] as number,
      key: input.key,
    }),
  views: [
    { record: 'curves', index: 0, title: 'Error, violation and price by iteration' },
    { record: 'curves', index: 1, title: 'Path from the unconstrained model', area: 'side' },
    { record: 'groupReport', index: 1, title: 'After', area: 'below' },
    { record: 'groupReport', index: 0, title: 'Unconstrained', area: 'below' },
  ],
  methods: [{ file: 'methods/fairness/in-processing.ts', name: 'exponentiatedGradient' }],
  references: ['Agarwal et al. (2018), A reductions approach to fair classification'],
})
