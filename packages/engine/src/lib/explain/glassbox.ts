/**
 * Glass-box models, interpretable by construction: explainable boosting machines (GA2M: a shape function per feature
 * and a few pairwise terms, learnt by cyclic gradient boosting over binned features, as InterpretML's EBM), their
 * predictions, per-row contributions and term importances, and points scorecards from logistic models.
 */
import { WhiteboxError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { columnOf, matrixView, targetVector } from '../learn/validation.ts'
import type { TensorLike } from '../tensor/index.ts'

/** A term of a glass-box model: a shape function of one feature, or a surface over a pair, piecewise constant. */
export type GlassBoxTerm =
  | {
      readonly features: readonly [number]
      /** Bin edges (cut points between bins), ascending. */
      readonly edges: readonly number[]
      /** The term's value in each bin (one more than the edges). */
      readonly values: readonly number[]
    }
  | {
      readonly features: readonly [number, number]
      readonly edges: readonly [readonly number[], readonly number[]]
      /** values[i][j] for bin i of the first feature and bin j of the second. */
      readonly values: readonly (readonly number[])[]
    }

/** An explainable boosting machine: an intercept plus its terms (on the link scale for classification). */
export interface GlassBoxModel {
  readonly task: 'regression' | 'classification'
  readonly intercept: number
  readonly terms: readonly GlassBoxTerm[]
}

/** Options of {@link explainableBoosting}. */
export interface ExplainableBoostingOptions {
  /** `regression` (squared error) or `classification` (binary log loss; labels 0/1). Default regression. */
  readonly task?: 'regression' | 'classification'
  /** Quantile bins per feature (default 32) and per feature of a pair (default 8). */
  readonly bins?: number
  readonly pairBins?: number
  /** Boosting rounds over all features (default 200) and the learning rate (default 0.05). */
  readonly rounds?: number
  readonly learningRate?: number
  /** Pairwise terms added after the main effects, the strongest by residual gain (default 0). */
  readonly interactions?: number
  /** Rounds for the pairwise terms (default 100). */
  readonly pairRounds?: number
}

function edgesOf(col: ArrayLike<number>, bins: number): number[] {
  const sorted = Float64Array.from(col).sort()
  const n = sorted.length
  const distinct = Array.from(new Set(sorted))
  // Few values: cut between each pair of consecutive values, so no bin is empty.
  if (distinct.length <= bins) return distinct.slice(1).map((v, i) => ((distinct[i] ?? 0) + v) / 2)
  const cuts: number[] = []
  for (let b = 1; b < bins; b++) {
    const h = (b / bins) * (n - 1)
    const lo = Math.floor(h)
    const v = (sorted[lo] ?? 0) + (h - lo) * ((sorted[Math.min(lo + 1, n - 1)] ?? 0) - (sorted[lo] ?? 0))
    if (v > (sorted[0] ?? 0) && v <= (sorted[n - 1] ?? 0) && (cuts.length === 0 || v > (cuts[cuts.length - 1] ?? 0)))
      cuts.push(v)
  }
  return cuts
}

function binOf(edges: readonly number[], v: number): number {
  let lo = 0
  let hi = edges.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (v < (edges[mid] ?? 0)) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * Explainable boosting (Lou, Caruana and Gehrke 2012 and 2013; InterpretML's ExplainableBoostingMachine): a
 * generalised additive model with a few pairwise interactions, f(x) = b + sum_j f_j(x_j) + sum_(j,k) f_jk(x_j, x_k),
 * fitted by cyclic gradient boosting: each round visits every feature in turn and moves its shape function by a small
 * step towards the mean residual (the negative gradient of the loss) in each of its quantile bins. Pairs are then
 * ranked by how much a coarse 2-D bin table reduces the residuals (FAST) and boosted the same way. Each term is
 * centred to mean zero over the data, the intercept carrying the rest, so a term reads as the feature's effect.
 *
 * @param data - Training rows and targets (0/1 for classification).
 * @param options - task, bins, pairBins, rounds, learningRate, interactions and pairRounds.
 * @returns The model.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const x = Array.from({ length: 40 }, (_, i) => [i % 4, Math.floor(i / 4) % 2])
 * const y = x.map(([a, b]) => 2 * a + (b === 1 ? 5 : 0))
 * const m = explain.explainableBoosting({ x, y }, { bins: 4, rounds: 300, learningRate: 0.2 })
 * // The shape of the second feature: a step of 5 between its two values.
 * Math.round(m.terms[1].values[1] - m.terms[1].values[0]) // => 5
 * ```
 */
export function explainableBoosting(data: LearnData, options: ExplainableBoostingOptions = {}): GlassBoxModel {
  const m = matrixView(data.x, 'explainableBoosting')
  if (data.y === undefined) throw new WhiteboxError('value', 'explainableBoosting: needs targets')
  const y = targetVector(data.y, 'explainableBoosting', m.n)
  const task = options.task ?? 'regression'
  const lr = options.learningRate ?? 0.05
  const cols = Array.from({ length: m.d }, (_, j) => columnOf(m, j))
  const edges = cols.map((c) => edgesOf(c, options.bins ?? 32))
  const bins = cols.map((c, j) => Array.from(c, (v) => binOf(edges[j] ?? [], v)))
  const mean = y.reduce((a, b) => a + b, 0) / m.n
  const intercept0 = task === 'classification' ? Math.log(Math.max(1e-6, mean) / Math.max(1e-6, 1 - mean)) : mean
  const f = new Float64Array(m.n).fill(intercept0)
  const residual = (): Float64Array =>
    Float64Array.from(f, (v, i) => (y[i] ?? 0) - (task === 'classification' ? 1 / (1 + Math.exp(-v)) : v))
  const shapes = edges.map((e) => new Array<number>(e.length + 1).fill(0))
  for (let round = 0; round < (options.rounds ?? 200); round++) {
    for (let j = 0; j < m.d; j++) {
      const r = residual()
      const nb = (edges[j]?.length ?? 0) + 1
      const sum = new Float64Array(nb)
      const count = new Float64Array(nb)
      const hess = new Float64Array(nb)
      for (let i = 0; i < m.n; i++) {
        const b = bins[j]?.[i] ?? 0
        sum[b] = (sum[b] ?? 0) + (r[i] ?? 0)
        count[b] = (count[b] ?? 0) + 1
        if (task === 'classification') {
          const p = 1 / (1 + Math.exp(-(f[i] ?? 0)))
          hess[b] = (hess[b] ?? 0) + p * (1 - p)
        }
      }
      const step = Array.from(sum, (s, b) => {
        const den = task === 'classification' ? (hess[b] ?? 0) + 1e-9 : (count[b] ?? 0)
        return den > 0 ? (lr * s) / den : 0
      })
      const shape = shapes[j] as number[]
      for (let b = 0; b < nb; b++) shape[b] = (shape[b] ?? 0) + (step[b] ?? 0)
      for (let i = 0; i < m.n; i++) f[i] = (f[i] ?? 0) + (step[bins[j]?.[i] ?? 0] ?? 0)
    }
  }
  const terms: GlassBoxTerm[] = []
  let intercept = intercept0
  for (let j = 0; j < m.d; j++) {
    const shape = shapes[j] ?? []
    let c = 0
    for (let i = 0; i < m.n; i++) c += (shape[bins[j]?.[i] ?? 0] ?? 0) / m.n
    intercept += c
    terms.push({ features: [j] as const, edges: edges[j] ?? [], values: shape.map((v) => v - c) })
  }
  const nPairs = options.interactions ?? 0
  if (nPairs > 0 && m.d >= 2) {
    const pb = options.pairBins ?? 8
    const pEdges = cols.map((c) => edgesOf(c, pb))
    const pBins = cols.map((c, j) => Array.from(c, (v) => binOf(pEdges[j] ?? [], v)))
    const r0 = residual()
    // FAST: rank pairs by the residual sum of squares a 2-D table of bin means removes.
    const gains: { j: number; k: number; gain: number }[] = []
    for (let j = 0; j < m.d; j++)
      for (let k = j + 1; k < m.d; k++) {
        const nj = (pEdges[j]?.length ?? 0) + 1
        const nk = (pEdges[k]?.length ?? 0) + 1
        const s = new Float64Array(nj * nk)
        const c = new Float64Array(nj * nk)
        for (let i = 0; i < m.n; i++) {
          const cell = (pBins[j]?.[i] ?? 0) * nk + (pBins[k]?.[i] ?? 0)
          s[cell] = (s[cell] ?? 0) + (r0[i] ?? 0)
          c[cell] = (c[cell] ?? 0) + 1
        }
        let gain = 0
        for (let t = 0; t < s.length; t++) if ((c[t] ?? 0) > 0) gain += (s[t] ?? 0) ** 2 / (c[t] ?? 1)
        gains.push({ j, k, gain })
      }
    gains.sort((a, b) => b.gain - a.gain || a.j - b.j || a.k - b.k)
    const chosen = gains.slice(0, nPairs)
    const tables = chosen.map(({ j, k }) =>
      Array.from({ length: (pEdges[j]?.length ?? 0) + 1 }, () =>
        new Array<number>((pEdges[k]?.length ?? 0) + 1).fill(0),
      ),
    )
    for (let round = 0; round < (options.pairRounds ?? 100); round++) {
      chosen.forEach(({ j, k }, t) => {
        const r = residual()
        const table = tables[t] as number[][]
        const nk = table[0]?.length ?? 1
        const s = new Float64Array(table.length * nk)
        const c = new Float64Array(table.length * nk)
        for (let i = 0; i < m.n; i++) {
          const cell = (pBins[j]?.[i] ?? 0) * nk + (pBins[k]?.[i] ?? 0)
          s[cell] = (s[cell] ?? 0) + (r[i] ?? 0)
          if (task === 'classification') {
            const p = 1 / (1 + Math.exp(-(f[i] ?? 0)))
            c[cell] = (c[cell] ?? 0) + p * (1 - p)
          } else c[cell] = (c[cell] ?? 0) + 1
        }
        for (let a = 0; a < table.length; a++)
          for (let b = 0; b < nk; b++) {
            const cell = a * nk + b
            if ((c[cell] ?? 0) > 0)
              (table[a] as number[])[b] = (table[a]?.[b] ?? 0) + (lr * (s[cell] ?? 0)) / ((c[cell] ?? 0) + 1e-9)
          }
        for (let i = 0; i < m.n; i++) {
          const a = pBins[j]?.[i] ?? 0
          const b = pBins[k]?.[i] ?? 0
          const cell = a * nk + b
          if ((c[cell] ?? 0) > 0) f[i] = (f[i] ?? 0) + (lr * (s[cell] ?? 0)) / ((c[cell] ?? 0) + 1e-9)
        }
      })
    }
    chosen.forEach(({ j, k }, t) => {
      const table = tables[t] as number[][]
      let c = 0
      for (let i = 0; i < m.n; i++) c += (table[pBins[j]?.[i] ?? 0]?.[pBins[k]?.[i] ?? 0] ?? 0) / m.n
      intercept += c
      terms.push({
        features: [j, k] as const,
        edges: [pEdges[j] ?? [], pEdges[k] ?? []] as const,
        values: table.map((r) => r.map((v) => v - c)),
      })
    })
  }
  return { task, intercept, terms }
}

/**
 * Each row's contribution from each term of a glass-box model: the term's value at the row (its local explanation,
 * exact by construction), so the intercept plus a row's contributions is its score.
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns contributions[row][term].
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const model = { task: 'regression', intercept: 1, terms: [{ features: [0], edges: [0.5], values: [-1, 1] }] }
 * explain.glassBoxContributions(model, [[0], [1]]) // => [[-1], [1]]
 * ```
 */
export function glassBoxContributions(model: GlassBoxModel, x: TensorLike): number[][] {
  const m = matrixView(x, 'glassBoxContributions')
  return Array.from({ length: m.n }, (_, i) =>
    model.terms.map((t): number => {
      const v = (j: number): number => m.data[i * m.d + j] ?? 0
      if (t.features.length === 1)
        return (t.values as readonly number[])[binOf(t.edges as readonly number[], v(t.features[0]))] ?? 0
      const [ex, ey] = t.edges as readonly [readonly number[], readonly number[]]
      const [a, b] = t.features as readonly [number, number]
      return (t.values as readonly (readonly number[])[])[binOf(ex, v(a))]?.[binOf(ey, v(b))] ?? 0
    }),
  )
}

/**
 * A glass-box model's predictions: the intercept plus the terms (a probability through the logistic function for
 * classification).
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns One prediction per row.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const model = { task: 'regression', intercept: 1, terms: [{ features: [0], edges: [0.5], values: [-1, 1] }] }
 * explain.glassBoxPredict(model, [[0], [1]]) // => [0, 2]
 * ```
 */
export function glassBoxPredict(model: GlassBoxModel, x: TensorLike): number[] {
  return glassBoxContributions(model, x).map((c) => {
    const s = c.reduce((a, b) => a + b, model.intercept)
    return model.task === 'classification' ? 1 / (1 + Math.exp(-s)) : s
  })
}

/**
 * The importance of each term (InterpretML's term importances): the mean absolute contribution over the rows.
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns One importance per term.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const model = { task: 'regression', intercept: 0, terms: [{ features: [0], edges: [0.5], values: [-1, 1] }] }
 * explain.termImportances(model, [[0], [1], [1]]) // => [1]
 * ```
 */
export function termImportances(model: GlassBoxModel, x: TensorLike): number[] {
  const c = glassBoxContributions(model, x)
  return model.terms.map((_, t) => c.reduce((s, row) => s + Math.abs(row[t] ?? 0), 0) / Math.max(1, c.length))
}

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
 * import * as explain from 'whitebox-engine/explain'
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
    const e = edgesOf(col, options.bins ?? 5)
    const sums = new Array<number>(e.length + 1).fill(0)
    const counts = new Array<number>(e.length + 1).fill(0)
    for (const v of col) {
      const b = binOf(e, v)
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
