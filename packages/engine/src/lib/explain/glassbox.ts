/**
 * Glass-box models, interpretable by construction: explainable boosting machines (GA2M: a shape function per feature
 * and a few pairwise terms, learnt by cyclic gradient boosting over binned features, as InterpretML's EBM; the booster
 * and the model's predictions, contributions and term importances live in `ensembles/cyclic.ts` and are re-exported
 * here) and points scorecards from logistic models.
 */
import { binIndex, quantileEdges } from '../ensembles/cyclic-binning.ts'
import { columnOf, matrixView } from '../learn/validation.ts'
import type { TensorLike } from '../tensor/index.ts'

export {
  type ExplainableBoostingOptions,
  explainableBoosting,
  type GlassBoxModel,
  type GlassBoxTerm,
  glassBoxContributions,
  glassBoxPredict,
  termImportances,
} from '../ensembles/cyclic.ts'

/** Options of {@link scorecard}. */
export interface ScorecardOptions {
  /** Bins per feature (default 5). */
  readonly bins?: number
  /** Points to double the odds (default 20), the score at the base odds (default 600) and those odds (default 50). */
  readonly pdo?: number
  readonly baseScore?: number
  readonly baseOdds?: number
}

/** A points scorecard: per feature, its bins with their points. */
export interface Scorecard {
  readonly features: readonly {
    readonly feature: number
    /** Cut points between bins. */
    readonly edges: readonly number[]
    /** Points per bin. */
    readonly points: readonly number[]
  }[]
  /** Points everyone starts with. */
  readonly basePoints: number
}

/**
 * A points scorecard from a logistic model (Siddiqi 2006, as credit scorecards are built): each feature split into
 * quantile bins, each bin's points factor * w_j * (bin mean of x_j) with factor = pdo / ln 2, plus base points
 * offset + factor * b, where offset = baseScore - factor ln(baseOdds); a row's score is the base points plus its bins'
 * points, which approximates offset + factor times its log odds.
 *
 * @param coef - The logistic model's weights `[d]` (log odds of the good outcome).
 * @param intercept - Its intercept.
 * @param x - Data `[n, d]` for the bins.
 * @param options - bins, pdo, baseScore and baseOdds.
 * @returns The scorecard.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * const card = explain.scorecard([Math.LN2], 0, [[0], [0], [1], [1]], { bins: 2, pdo: 20, baseScore: 600, baseOdds: 1 })
 * card.features[0].points.map((p) => Math.round(p)) // => [0, 20]
 * ```
 */
export function scorecard(
  coef: readonly number[],
  intercept: number,
  x: TensorLike,
  options: ScorecardOptions = {},
): Scorecard {
  const m = matrixView(x, 'scorecard', { nFeatures: coef.length })
  const factor = (options.pdo ?? 20) / Math.LN2
  const offset = (options.baseScore ?? 600) - factor * Math.log(options.baseOdds ?? 50)
  const features = coef.map((w, j) => {
    const col = columnOf(m, j)
    const e = quantileEdges(col, options.bins ?? 5)
    const sums = new Array<number>(e.length + 1).fill(0)
    const counts = new Array<number>(e.length + 1).fill(0)
    for (const v of col) {
      const b = binIndex(e, v)
      sums[b] = (sums[b] ?? 0) + v
      counts[b] = (counts[b] ?? 0) + 1
    }
    return {
      feature: j,
      edges: e,
      points: sums.map((s, b) => factor * w * ((counts[b] ?? 0) > 0 ? s / (counts[b] ?? 1) : 0)),
    }
  })
  return { features, basePoints: offset + factor * intercept }
}
