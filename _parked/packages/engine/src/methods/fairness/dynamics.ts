/**
 * Studies of fairness over time: the delayed impact of lending policies on two groups' scores (Liu et al.), as their
 * outcome curves with each policy's selection marked, and the feedback loop of repeated decisions, as each group's
 * mean score over the rounds (both `curves` records).
 */

import { curveSet, type Explained } from '../../lib/exposition/index.ts'
import {
  delayedImpact,
  type ImpactGroup,
  type ImpactModel,
  type ImpactPolicy,
  outcomeCurve,
  repeatedDecisions,
} from '../../lib/fairness/index.ts'

/** Options of {@link delayedImpactStudy}. */
export interface DelayedImpactStudyOptions {
  /** Rounds of the feedback loop (default 10). */
  readonly rounds?: number
  /** The policy of the feedback loop (default demographic parity). */
  readonly policy?: ImpactPolicy['name']
}

/**
 * The delayed impact of lending policies (`delayedImpact`, `repeatedDecisions` of `mdml-engine/fairness`):
 * each group's outcome curve (expected mean score change against its selection rate) with the maximum-utility,
 * demographic-parity and equal-opportunity selections marked, and each group's mean score over repeated rounds under a
 * policy. Exposition: two `curves` records.
 *
 * @param model - The score grid, repayment probabilities, score changes and utilities.
 * @param groups - The groups, their shares and score distributions.
 * @param options - Rounds and the policy of the loop.
 * @returns The policies and the rounds, with the records.
 */
export function delayedImpactStudy(
  model: ImpactModel,
  groups: readonly ImpactGroup[],
  options: DelayedImpactStudyOptions = {},
): Explained<{ readonly policies: readonly ImpactPolicy[]; readonly meanScore: readonly (readonly number[])[] }> {
  const policies = delayedImpact(model, groups, 201)
  const curves = groups.map((g) => outcomeCurve(model, g.distribution, 101))
  const loop = repeatedDecisions(model, groups, {
    rounds: options.rounds ?? 10,
    policy: options.policy ?? 'demographicParity',
  })
  return {
    policies,
    meanScore: loop.meanScore,
    exposition: [
      curveSet({
        x: curves[0]?.selection ?? [],
        xLabel: 'selection rate',
        yLabel: 'expected change in mean score',
        series: groups.map((g, j) => ({ name: g.name, y: curves[j]?.meanChange ?? [] })),
        marks: policies.flatMap((p) =>
          groups.map((g, j) => ({ x: p.selection[j] ?? 0, label: `${p.name}: ${g.name}` })),
        ),
      }),
      curveSet({
        x: loop.meanScore.map((_, t) => t),
        xLabel: 'round',
        yLabel: 'mean score',
        series: groups.map((g, j) => ({ name: g.name, y: loop.meanScore.map((m) => m[j] ?? 0) })),
      }),
    ],
  }
}
