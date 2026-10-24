// TODO: expected model change, BALD (needs Bayesian neural nets), core-sets (k-centre greedy), density-weighted and
// diversity strategies, BADGE.
import { WhiteboxError } from '../../lib/base/index.ts'
import {
  consensusEntropySampling,
  entropySampling,
  expectedErrorReduction,
  marginSampling,
  maxDisagreementSampling,
  type QueryStrategy,
  randomSampling,
  rankedBatchSampling,
  takeRows,
  uncertaintySampling,
  voteEntropySampling,
} from '../../lib/learn/index.ts'
import { foldIn, permutation } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'

/** The query strategies by name (the uncertainty, committee, error-reduction and batch families of modAL). */
export type StrategyName =
  | 'random'
  | 'least-confident'
  | 'margin'
  | 'entropy'
  | 'vote-entropy'
  | 'consensus-entropy'
  | 'max-disagreement'
  | 'expected-error-reduction'
  | 'ranked-batch'

/** Every strategy name, for controls. */
export const STRATEGY_NAMES: readonly StrategyName[] = [
  'random',
  'least-confident',
  'margin',
  'entropy',
  'vote-entropy',
  'consensus-entropy',
  'max-disagreement',
  'expected-error-reduction',
  'ranked-batch',
]

/**
 * Restrict a strategy to a keyed random subset of the pool each round (for costly strategies such as expected error
 * reduction, which refits the model once per candidate and class).
 *
 * @param strategy - The strategy.
 * @param candidates - Pool rows considered per round.
 * @returns The restricted strategy (same name with a `-lite` suffix).
 */
export function subsampled(strategy: QueryStrategy, candidates: number): QueryStrategy {
  if (!(Number.isInteger(candidates) && candidates >= 1))
    throw new WhiteboxError('value', 'subsampled: candidates must be a positive integer')
  return {
    name: `${strategy.name}-lite`,
    ...(strategy.committee === undefined ? {} : { committee: strategy.committee }),
    select: (ctx, n) => {
      if (ctx.pool.n <= candidates) return strategy.select(ctx, n)
      const subset = Array.from(toFlat(permutation(foldIn(ctx.key, 7), ctx.pool.n))).slice(0, candidates)
      const picks = strategy.select({ ...ctx, pool: takeRows(ctx.pool, subset) }, Math.min(n, candidates))
      return picks.map((p) => subset[p] ?? 0)
    },
  }
}

/**
 * A query strategy of the core (`lib/learn/active-learning.ts`) by name; expected error reduction considers at most
 * `candidates` pool rows a round.
 *
 * @param name - The strategy.
 * @param options - Candidates per round for expected error reduction (default 20).
 * @returns The strategy.
 */
export function queryStrategy(name: StrategyName, options: { readonly candidates?: number } = {}): QueryStrategy {
  switch (name) {
    case 'random':
      return randomSampling()
    case 'least-confident':
      return uncertaintySampling()
    case 'margin':
      return marginSampling()
    case 'entropy':
      return entropySampling()
    case 'vote-entropy':
      return voteEntropySampling()
    case 'consensus-entropy':
      return consensusEntropySampling()
    case 'max-disagreement':
      return maxDisagreementSampling()
    case 'expected-error-reduction':
      return subsampled(expectedErrorReduction(), options.candidates ?? 20)
    case 'ranked-batch':
      return rankedBatchSampling()
  }
}
