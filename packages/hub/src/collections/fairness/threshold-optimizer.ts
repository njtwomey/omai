import { thresholdOptimizer as optimise, type ThresholdConstraint } from 'whitebox-engine/methods/fairness'
import { choice, dict } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { awareParam, fairnessDataOf, fitClassifier } from './_shared.ts'

/** Post-processing: per-group randomised thresholds meeting a fairness constraint. */
export const thresholdOptimizer: HubEntry = defineEntry({
  id: 'fairness/threshold-optimizer',
  title: 'Threshold optimiser',
  summary: 'Per-group randomised thresholds that meet demographic parity or equalised odds at the least cost.',
  collection: 'fairness',
  tags: ['fairness', 'post-processing', 'thresholds', 'equalised odds', 'demographic parity'],
  datasets: ['fairnessTabular'],
  params: dict({
    constraint: choice(['demographicParity', 'equalisedOdds', 'truePositiveRateParity', 'falsePositiveRateParity'], {
      default: 'demographicParity',
    }),
    aware: awareParam,
  }),
  run: (input) => {
    const data = fairnessDataOf(input)
    const aware = input.params['aware'] as boolean
    return optimise(fitClassifier(data, aware, input), data, {
      constraint: input.params['constraint'] as ThresholdConstraint,
      dropSensitive: !aware,
    })
  },
  views: [
    { record: 'groupReport', index: 1, title: 'After: expected decisions' },
    { record: 'curves', title: 'The trade-off the choice is made on', area: 'side' },
    { record: 'groupReport', index: 0, title: 'Before: one threshold', area: 'below' },
  ],
  methods: [{ file: 'methods/fairness/post-processing.ts', name: 'thresholdOptimizer' }],
  references: ['Hardt, Price and Srebro (2016), Equality of opportunity in supervised learning'],
})
