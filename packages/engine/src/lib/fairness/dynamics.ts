/**
 * Fairness over time: the delayed impact of decisions on the groups they are made about (Liu, Dean, Rolf, Simchowitz
 * and Hardt, 2018), each group's outcome curve (the change in its mean score against its selection rate) and the
 * groups' changes under the bank's maximum-utility, demographic-parity and equal-opportunity policies; and the
 * feedback loop of repeated decisions, where each round's decisions move the score distributions the next round sees.
 */

import { WhiteboxError } from '../base/index.ts'

/** A lending model on a grid of scores: each score's probability of repaying, and what each outcome is worth. */
export interface ImpactModel {
  /** The score grid, ascending. */
  readonly scores: readonly number[]
  /** P(repay | score) on the grid. */
  readonly repay: readonly number[]
  /** The change in an applicant's score when they repay and when they default (default +75 and -150, Liu et al.). */
  readonly scoreChange?: readonly [number, number]
  /** The lender's utility of a repaid and a defaulted loan (default +1 and -4). */
  readonly utility?: readonly [number, number]
}

/** One group: its share of the population and its distribution over the score grid. */
export interface ImpactGroup {
  readonly name: string
  readonly share: number
  readonly distribution: readonly number[]
}

/** The selection of the top fraction beta of a distribution: the mass taken at each grid point (highest first). */
function selectTop(dist: readonly number[], beta: number): number[] {
  const taken = dist.map(() => 0)
  let left = beta
  for (let i = dist.length - 1; i >= 0 && left > 1e-15; i--) {
    const t = Math.min(dist[i] ?? 0, left)
    taken[i] = t
    left -= t
  }
  return taken
}

/** A group's expected score change, lender utility and true-positive rate when its top beta is selected. */
function evaluate(model: ImpactModel, dist: readonly number[], beta: number) {
  const [up, down] = model.scoreChange ?? [75, -150]
  const [gain, loss] = model.utility ?? [1, -4]
  const taken = selectTop(dist, beta)
  let change = 0
  let utility = 0
  let repaid = 0
  let qualified = 0
  for (const [i, t] of taken.entries()) {
    const r = model.repay[i] ?? 0
    change += t * (r * up + (1 - r) * down)
    utility += t * (r * gain + (1 - r) * loss)
    repaid += t * r
    qualified += (dist[i] ?? 0) * r
  }
  return { change, utility, tpr: qualified === 0 ? 0 : repaid / qualified }
}

/** A group's outcome curve: against selection rate, its expected mean score change, the lender's utility and its TPR. */
export interface OutcomeCurve {
  readonly selection: readonly number[]
  readonly meanChange: readonly number[]
  readonly utility: readonly number[]
  readonly truePositiveRate: readonly number[]
}

/**
 * A group's outcome curve (Liu et al., 2018): lending to the group's top fraction beta of scores, the expected change
 * in the group's mean score (repaid loans raise scores, defaults lower them), the lender's expected utility and the
 * true-positive rate, for beta from 0 to 1. The change rises, peaks and then falls (lending too much harms the group):
 * relative improvement, then relative harm, then active harm once it drops below zero.
 *
 * @param model - The score grid, repayment probabilities, score changes and utilities.
 * @param distribution - The group's distribution over the grid.
 * @param steps - Selection rates evaluated (default 101).
 * @returns The curve.
 * @throws WhiteboxError `shape` when the distribution and the grid differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { outcomeCurve } from 'whitebox-engine/fairness'
 * const c = outcomeCurve({ scores: [300, 600, 800], repay: [0.2, 0.7, 0.95] }, [0.3, 0.4, 0.3], 3)
 * c.meanChange.map((v) => Math.round(v * 100) / 100) // => [0, 20.62, -9.37]
 * ```
 */
export function outcomeCurve(model: ImpactModel, distribution: readonly number[], steps = 101): OutcomeCurve {
  if (distribution.length !== model.scores.length || model.repay.length !== model.scores.length)
    throw new WhiteboxError('shape', 'outcomeCurve: one probability and repayment per score')
  const selection = Array.from({ length: steps }, (_, i) => (steps === 1 ? 0 : i / (steps - 1)))
  const points = selection.map((b) => evaluate(model, distribution, b))
  return {
    selection,
    meanChange: points.map((p) => p.change),
    utility: points.map((p) => p.utility),
    truePositiveRate: points.map((p) => p.tpr),
  }
}

/** A policy's selection rate, score change and utility for each group. */
export interface ImpactPolicy {
  readonly name: 'maxUtility' | 'demographicParity' | 'equalOpportunity'
  readonly selection: readonly number[]
  readonly meanChange: readonly number[]
  readonly utility: number
}

/**
 * Delayed impact of three lending policies (Liu et al., 2018): the lender's maximum utility (each group its own best
 * selection rate), demographic parity (one selection rate for all, the best for the lender) and equal opportunity (one
 * true-positive rate for all, the best for the lender), with each group's expected change in mean score under each. A
 * fairness constraint can help a group (raise its score) or hurt it (lend to it past its peak).
 *
 * @param model - The score grid, repayment probabilities, score changes and utilities.
 * @param groups - The groups, their shares and distributions.
 * @param steps - The grid of selection rates searched (default 1001).
 * @returns Each policy's selection rates, changes and utility.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { delayedImpact } from 'whitebox-engine/fairness'
 * const model = { scores: [300, 600, 800], repay: [0.2, 0.7, 0.95] }
 * const r = delayedImpact(model, [
 *   { name: 'a', share: 0.5, distribution: [0.1, 0.4, 0.5] },
 *   { name: 'b', share: 0.5, distribution: [0.5, 0.4, 0.1] },
 * ], 101)
 * r.map((p) => p.name) // => ['maxUtility', 'demographicParity', 'equalOpportunity']
 * ```
 */
export function delayedImpact(model: ImpactModel, groups: readonly ImpactGroup[], steps = 1001): ImpactPolicy[] {
  const curves = groups.map((g) => outcomeCurve(model, g.distribution, steps))
  const grid = curves[0]?.selection ?? []
  const total = (betas: readonly number[]): number =>
    groups.reduce((s, g, j) => s + g.share * evaluate(model, g.distribution, betas[j] ?? 0).utility, 0)
  const policy = (name: ImpactPolicy['name'], betas: number[]): ImpactPolicy => ({
    name,
    selection: betas,
    meanChange: groups.map((g, j) => evaluate(model, g.distribution, betas[j] ?? 0).change),
    utility: total(betas),
  })
  const argmax = (values: readonly number[]): number => values.indexOf(Math.max(...values))
  const maxUtility = curves.map((c) => grid[argmax(c.utility)] ?? 0)
  const dp = grid[argmax(grid.map((b) => total(groups.map(() => b))))] ?? 0
  // Equal opportunity: for each true-positive rate on the grid, each group's selection rate reaching it.
  const betaAt = (c: OutcomeCurve, tpr: number): number => {
    const i = c.truePositiveRate.findIndex((t) => t >= tpr - 1e-12)
    return i < 0 ? 1 : (c.selection[i] ?? 1)
  }
  const eoBetas = grid.map((t) => curves.map((c) => betaAt(c, t)))
  const eo = eoBetas[argmax(eoBetas.map(total))] ?? groups.map(() => 0)
  return [
    policy('maxUtility', maxUtility),
    policy(
      'demographicParity',
      groups.map(() => dp),
    ),
    policy('equalOpportunity', eo),
  ]
}

/** The state of the groups after each round of {@link repeatedDecisions}. */
export interface DecisionRounds {
  /** distributions[t][g]: group g's distribution over the grid after round t (round 0 is the start). */
  readonly distributions: readonly (readonly (readonly number[])[])[]
  /** meanScore[t][g] and selection[t][g] (the selection rate used in round t + 1). */
  readonly meanScore: readonly (readonly number[])[]
  readonly selection: readonly (readonly number[])[]
}

/**
 * The feedback loop of repeated decisions: each round a policy (its selection rate per group, recomputed from the
 * current distributions) lends to each group's top scores; repaid loans move applicants up the grid by the score
 * change, defaults move them down, and the next round sees the moved distributions. A policy that looks fair in one
 * round can widen or close the gap over many.
 *
 * @param model - The score grid, repayment probabilities, score changes and utilities.
 * @param groups - The groups at the start.
 * @param options - The policy (default maxUtility) and the rounds (default 10).
 * @returns The distributions, mean scores and selection rates round by round.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { repeatedDecisions } from 'whitebox-engine/fairness'
 * const model = { scores: [300, 450, 600, 750], repay: [0.1, 0.5, 0.8, 0.95], scoreChange: [150, -150] as const }
 * const r = repeatedDecisions(model, [{ name: 'a', share: 1, distribution: [0.25, 0.25, 0.25, 0.25] }], { rounds: 3 })
 * r.meanScore.length // => 4
 * ```
 */
export function repeatedDecisions(
  model: ImpactModel,
  groups: readonly ImpactGroup[],
  options: { readonly policy?: ImpactPolicy['name']; readonly rounds?: number; readonly steps?: number } = {},
): DecisionRounds {
  const rounds = options.rounds ?? 10
  const [up, down] = model.scoreChange ?? [75, -150]
  const grid = model.scores
  // The grid index an applicant moves to: the nearest grid score to theirs plus the change, clamped.
  const moveTo = (i: number, delta: number): number => {
    const target = (grid[i] ?? 0) + delta
    let best = 0
    for (let j = 1; j < grid.length; j++)
      if (Math.abs((grid[j] ?? 0) - target) < Math.abs((grid[best] ?? 0) - target)) best = j
    return best
  }
  const mean = (d: readonly number[]): number => {
    const z = d.reduce((s, v) => s + v, 0)
    return z === 0 ? 0 : d.reduce((s, v, i) => s + v * (grid[i] ?? 0), 0) / z
  }
  let current = groups.map((g) => [...g.distribution])
  const distributions: number[][][] = [current.map((d) => [...d])]
  const meanScore: number[][] = [current.map(mean)]
  const selection: number[][] = []
  for (let t = 0; t < rounds; t++) {
    const now = groups.map((g, j) => ({ ...g, distribution: current[j] ?? [] }))
    const chosen = delayedImpact(model, now, options.steps ?? 201).find(
      (p) => p.name === (options.policy ?? 'maxUtility'),
    )
    const betas = chosen?.selection ?? groups.map(() => 0)
    selection.push([...betas])
    current = current.map((d, j) => {
      const taken = selectTop(d, betas[j] ?? 0)
      const next = d.map((v, i) => v - (taken[i] ?? 0))
      for (const [i, m] of taken.entries()) {
        if (m === 0) continue
        const r = model.repay[i] ?? 0
        const a = moveTo(i, up)
        const b = moveTo(i, down)
        next[a] = (next[a] ?? 0) + m * r
        next[b] = (next[b] ?? 0) + m * (1 - r)
      }
      return next
    })
    distributions.push(current.map((d) => [...d]))
    meanScore.push(current.map(mean))
  }
  return { distributions, meanScore, selection }
}
