/**
 * Missing-data diagnostics and Little's (1988) chi-square test for Missing Completely At Random (MCAR): column, row,
 * and pattern-level missingness profiles (`missingnessProfile`), and Little's multivariate MCAR test (`littleMcarTest`)
 * using Gaussian EM estimation of the pooled mean and covariance across arbitrary missingness patterns.
 */

import { OmaiError, type TestResult } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { invSpdOrPinv } from './_linear.ts'
import { chi2Sf, mat, rowsOf, vec } from './_util.ts'
import { testResult } from './tests/result.ts'

/**
 * Summary of a single missingness pattern across the $p$ columns of a dataset.
 */
export interface MissingPatternSummary {
  /** Binary string key (e.g. `'110'`) where `'1'` marks an observed column and `'0'` marks a missing (`NaN`) column. */
  readonly key: string
  /** Boolean mask of length $p$ (`true` where the column is missing / `NaN`, `false` where observed). */
  readonly missingMask: readonly boolean[]
  /** Indices of observed columns in this pattern. */
  readonly observedColumns: readonly number[]
  /** Indices of missing (`NaN`) columns in this pattern. */
  readonly missingColumns: readonly number[]
  /** Number of rows exhibiting this pattern. */
  readonly count: number
  /** Row indices exhibiting this pattern. */
  readonly rowIndices: readonly number[]
  /** Sample mean of the observed columns within this pattern (length `observedColumns.length`). */
  readonly observedMeans: Tensor
}

/**
 * Complete missingness profile of a 2-D dataset with `NaN` entries representing missing values.
 */
export interface MissingnessProfile {
  /** Number of rows $n$. */
  readonly rows: number
  /** Number of columns $p$. */
  readonly cols: number
  /** Total cell count $n \times p$. */
  readonly totalCells: number
  /** Total count of missing (`NaN`) cells. */
  readonly missingCells: number
  /** Overall fraction of missing cells in `[0, 1]`. */
  readonly missingFraction: number
  /** Number of complete cases (rows with zero `NaN` entries). */
  readonly completeCaseCount: number
  /** Fraction of complete rows in `[0, 1]`. */
  readonly completeCaseFraction: number
  /** Count of missing values in each column (shape `[cols]`). */
  readonly missingPerColumn: Tensor
  /** Fraction of missing values in each column (shape `[cols]`). */
  readonly fractionPerColumn: Tensor
  /** Count of missing values in each row (shape `[rows]`). */
  readonly missingPerRow: Tensor
  /** Distinct missingness patterns sorted by descending row count. */
  readonly patterns: readonly MissingPatternSummary[]
}

function readMatrixAllowingNaN(
  name: string,
  x: TensorLike,
): { readonly rows: number; readonly cols: number; readonly data: Float64Array } {
  const { rows, cols, data } = rowsOf(name, x)
  if (rows < 1 || cols < 1) {
    throw new OmaiError('shape', `${name}: expected a non-empty 2-D matrix, got [${rows}, ${cols}]`)
  }
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? Number.NaN
    if (v === Number.POSITIVE_INFINITY || v === Number.NEGATIVE_INFINITY) {
      throw new OmaiError('value', `${name}: infinite value at flat index ${i} (use NaN for missing values)`)
    }
  }
  return { rows, cols, data }
}

/**
 * Summarize missing-value (`NaN`) counts, rates, complete cases, and unique missingness patterns in a 2-D dataset.
 *
 * @param x - 2-D matrix of shape `[n, p]` where `NaN` indicates a missing entry.
 * @returns Column, row, and pattern-level missingness diagnostics.
 * @throws OmaiError with code `shape` if `x` is not a non-empty 2-D matrix, or `value` if `x` contains `±Infinity`.
 * @example
 * ```ts
 * import { missingnessProfile } from 'omai-engine/stats'
 * const prof = missingnessProfile([
 *   [1, 2],
 *   [3, Number.NaN],
 *   [5, 6],
 * ])
 * prof.missingCells // => 1
 * prof.completeCaseCount // => 2
 * prof.patterns.length // => 2
 * ```
 */
export function missingnessProfile(x: TensorLike): MissingnessProfile {
  const { rows, cols, data } = readMatrixAllowingNaN('missingnessProfile', x)
  const missingPerCol = new Float64Array(cols)
  const fractionPerCol = new Float64Array(cols)
  const missingPerRow = new Float64Array(rows)

  let missingCells = 0
  let completeCaseCount = 0

  const patternMap = new Map<
    string,
    {
      readonly key: string
      readonly missingMask: boolean[]
      readonly observedColumns: number[]
      readonly missingColumns: number[]
      readonly rowIndices: number[]
      readonly obsSums: Float64Array
    }
  >()

  for (let i = 0; i < rows; i++) {
    let rowMiss = 0
    let key = ''
    for (let j = 0; j < cols; j++) {
      const v = data[i * cols + j] ?? Number.NaN
      if (Number.isNaN(v)) {
        rowMiss++
        missingCells++
        missingPerCol[j] = (missingPerCol[j] ?? 0) + 1
        key += '0'
      } else {
        key += '1'
      }
    }
    missingPerRow[i] = rowMiss
    if (rowMiss === 0) completeCaseCount++

    let entry = patternMap.get(key)
    if (entry === undefined) {
      const missingMask: boolean[] = []
      const observedColumns: number[] = []
      const missingColumns: number[] = []
      for (let j = 0; j < cols; j++) {
        const isMiss = key[j] === '0'
        missingMask.push(isMiss)
        if (isMiss) missingColumns.push(j)
        else observedColumns.push(j)
      }
      entry = {
        key,
        missingMask,
        observedColumns,
        missingColumns,
        rowIndices: [],
        obsSums: new Float64Array(observedColumns.length),
      }
      patternMap.set(key, entry)
    }
    entry.rowIndices.push(i)
    for (let k = 0; k < entry.observedColumns.length; k++) {
      const col = entry.observedColumns[k] ?? 0
      entry.obsSums[k] = (entry.obsSums[k] ?? 0) + (data[i * cols + col] ?? 0)
    }
  }

  for (let j = 0; j < cols; j++) {
    fractionPerCol[j] = (missingPerCol[j] ?? 0) / rows
  }

  const patterns: MissingPatternSummary[] = Array.from(patternMap.values())
    .sort((a, b) => b.rowIndices.length - a.rowIndices.length || a.key.localeCompare(b.key))
    .map((e) => {
      const m = e.rowIndices.length
      const means = new Float64Array(e.observedColumns.length)
      for (let k = 0; k < means.length; k++) {
        means[k] = (e.obsSums[k] ?? 0) / m
      }
      return {
        key: e.key,
        missingMask: e.missingMask,
        observedColumns: e.observedColumns,
        missingColumns: e.missingColumns,
        count: m,
        rowIndices: e.rowIndices,
        observedMeans: vec(means),
      }
    })

  const totalCells = rows * cols
  return {
    rows,
    cols,
    totalCells,
    missingCells,
    missingFraction: missingCells / totalCells,
    completeCaseCount,
    completeCaseFraction: completeCaseCount / rows,
    missingPerColumn: vec(missingPerCol),
    fractionPerColumn: vec(fractionPerCol),
    missingPerRow: vec(missingPerRow),
    patterns,
  }
}

/**
 * Per-pattern breakdown in Little's MCAR chi-square test.
 */
export interface LittlePatternBreakdown {
  /** Binary pattern key (`'1'` = observed, `'0'` = missing). */
  readonly key: string
  /** Indices of observed columns in this pattern. */
  readonly observedColumns: readonly number[]
  /** Indices of missing (`NaN`) columns in this pattern. */
  readonly missingColumns: readonly number[]
  /** Number of rows $m_j$ in this pattern. */
  readonly count: number
  /** Observed subvector mean $\bar{y}_{j, O_j}$ in this pattern. */
  readonly observedMeans: Tensor
  /** Quadratic-form contribution $d_j^2 = m_j (\bar{y}_{j, O_j} - \hat{\mu}_{O_j})^\top \hat{\Sigma}_{O_j, O_j}^{-1} (\bar{y}_{j, O_j} - \hat{\mu}_{O_j})$ to Little's $d_0^2$. */
  readonly distanceSquared: number
}

/**
 * Result of {@link littleMcarTest}: standard {@link TestResult} plus pattern breakdown and EM-estimated pooled moments.
 */
export interface LittleMcarResult extends TestResult {
  /** Breakdown of each non-empty missingness pattern and its contribution $d_j^2$ to the test statistic. */
  readonly patterns: readonly LittlePatternBreakdown[]
  /** Pooled maximum-likelihood mean estimate $\hat{\mu}$ of shape `[cols]`. */
  readonly pooledMean: Tensor
  /** Pooled maximum-likelihood covariance estimate $\hat{\Sigma}$ of shape `[cols, cols]`. */
  readonly pooledCovariance: Tensor
  /** Number of Gaussian EM iterations used to estimate $(\hat{\mu}, \hat{\Sigma})$. */
  readonly emIterations: number
}

/**
 * Options for {@link littleMcarTest}.
 */
export interface LittleMcarOptions {
  /** Maximum EM iterations for pooled mean and covariance estimation (default `200`). */
  readonly maxIter?: number
  /** Relative convergence tolerance on pooled parameters in EM (default `1e-8`). */
  readonly tol?: number
}

/**
 * Little's (1988) multivariate $\chi^2$ test for Missing Completely At Random (MCAR):
 * estimates the pooled mean $\hat{\mu}$ and covariance $\hat{\Sigma}$ by the Gaussian EM algorithm across all rows with
 * at least one observed variable, then forms
 * $d_0^2 = \sum_{j=1}^J m_j (\bar{y}_{j, O_j} - \hat{\mu}_{O_j})^\top \hat{\Sigma}_{O_j, O_j}^{-1} (\bar{y}_{j, O_j} - \hat{\mu}_{O_j})$
 * with $\text{df} = \sum_{j=1}^J p_j - p$ degrees of freedom under $H_0$ (MCAR).
 *
 * When the dataset has only a single missingness pattern (e.g. no missing values at all), $\text{df} = 0$,
 * `statistic = 0`, and `pValue = 1`.
 *
 * @param x - 2-D matrix of shape `[n, p]` ($n \ge 2, p \ge 1$) with `NaN` for missing values.
 * @param options - `maxIter` (default `200`) and `tol` (default `1e-8`).
 * @returns `LittleMcarResult` (`TestResult` plus per-pattern $d_j^2$, `pooledMean`, `pooledCovariance`, and `emIterations`).
 * @throws OmaiError with code `shape` or `value` if a column is completely missing or has zero variance.
 * @example
 * ```ts
 * import { littleMcarTest } from 'omai-engine/stats'
 * const data = [
 *   [1.0, 2.1],
 *   [2.0, 3.9],
 *   [3.0, 6.1],
 *   [4.0, 8.0],
 *   [1.5, Number.NaN],
 *   [3.5, Number.NaN],
 * ]
 * const res = littleMcarTest(data)
 * res.df // => 1
 * res.pValue > 0.5 // => true
 * ```
 */
export function littleMcarTest(x: TensorLike, options: LittleMcarOptions = {}): LittleMcarResult {
  const { rows, cols, data } = readMatrixAllowingNaN('littleMcarTest', x)
  if (rows < 2) {
    throw new OmaiError('value', `littleMcarTest: need at least 2 rows, got ${rows}`)
  }
  const maxIter = options.maxIter ?? 200
  const tol = options.tol ?? 1e-8
  const prof = missingnessProfile(x)

  // Keep only patterns that observe at least one variable.
  const activePatterns = prof.patterns.filter((p) => p.observedColumns.length > 0)
  if (activePatterns.length === 0) {
    throw new OmaiError('value', 'littleMcarTest: all entries in the dataset are NaN')
  }

  // Initialize mu and Sigma from column-wise available cases.
  const mu = new Float64Array(cols)
  const colCounts = new Float64Array(cols)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = data[i * cols + j] ?? Number.NaN
      if (!Number.isNaN(v)) {
        mu[j] = (mu[j] ?? 0) + v
        colCounts[j] = (colCounts[j] ?? 0) + 1
      }
    }
  }
  for (let j = 0; j < cols; j++) {
    const c = colCounts[j] ?? 0
    if (c < 1) {
      throw new OmaiError('value', `littleMcarTest: column ${j} has no observed values`)
    }
    mu[j] = (mu[j] ?? 0) / c
  }

  const sigma = new Float64Array(cols * cols)
  for (let j = 0; j < cols; j++) {
    let ss = 0
    let c = 0
    for (let i = 0; i < rows; i++) {
      const v = data[i * cols + j] ?? Number.NaN
      if (!Number.isNaN(v)) {
        const d = v - (mu[j] ?? 0)
        ss += d * d
        c++
      }
    }
    sigma[j * cols + j] = Math.max(ss / Math.max(1, c), 1e-8)
  }

  // Total active rows (having at least one observed variable).
  let nActive = 0
  for (let pIdx = 0; pIdx < activePatterns.length; pIdx++) {
    nActive += activePatterns[pIdx]?.count ?? 0
  }

  let emIterations = 0
  const yComplete = new Float64Array(cols)
  if (activePatterns.some((p) => p.missingColumns.length > 0)) {
    for (let iter = 0; iter < maxIter; iter++) {
      emIterations = iter + 1
      const muNext = new Float64Array(cols)
      const secondMoment = new Float64Array(cols * cols)

      for (let pIdx = 0; pIdx < activePatterns.length; pIdx++) {
        const pat = activePatterns[pIdx]
        if (pat === undefined) continue
        const obs = pat.observedColumns
        const miss = pat.missingColumns
        const pObs = obs.length
        const pMiss = miss.length

        // Extract Sigma_OO and its inverse, plus regression matrix B = Sigma_MO * Sigma_OO^-1 and conditional cov
        // V_MM|O = Sigma_MM - B * Sigma_OM.
        const bMat = new Float64Array(pMiss * pObs)
        const condCov = new Float64Array(pMiss * pMiss)

        if (pMiss > 0) {
          const sigOO = new Float64Array(pObs * pObs)
          for (let a = 0; a < pObs; a++) {
            const ca = obs[a] ?? 0
            for (let b = 0; b < pObs; b++) {
              const cb = obs[b] ?? 0
              sigOO[a * pObs + b] = sigma[ca * cols + cb] ?? 0
            }
          }
          const invOO = invSpdOrPinv(sigOO, pObs)
          const sigMO = new Float64Array(pMiss * pObs)
          for (let a = 0; a < pMiss; a++) {
            const ma = miss[a] ?? 0
            for (let b = 0; b < pObs; b++) {
              const ob = obs[b] ?? 0
              sigMO[a * pObs + b] = sigma[ma * cols + ob] ?? 0
            }
          }
          for (let a = 0; a < pMiss; a++) {
            for (let b = 0; b < pObs; b++) {
              let acc = 0
              for (let k = 0; k < pObs; k++) {
                acc += (sigMO[a * pObs + k] ?? 0) * (invOO[k * pObs + b] ?? 0)
              }
              bMat[a * pObs + b] = acc
            }
          }
          for (let a = 0; a < pMiss; a++) {
            const ma = miss[a] ?? 0
            for (let b = 0; b < pMiss; b++) {
              const mb = miss[b] ?? 0
              let red = 0
              for (let k = 0; k < pObs; k++) {
                red += (bMat[a * pObs + k] ?? 0) * (sigMO[b * pObs + k] ?? 0)
              }
              const val = (sigma[ma * cols + mb] ?? 0) - red
              condCov[a * pMiss + b] = a === b ? Math.max(0, val) : val
            }
          }
        }

        for (let r = 0; r < pat.rowIndices.length; r++) {
          const row = pat.rowIndices[r] ?? 0
          for (let a = 0; a < pObs; a++) {
            const ca = obs[a] ?? 0
            yComplete[ca] = data[row * cols + ca] ?? 0
          }
          for (let a = 0; a < pMiss; a++) {
            const ma = miss[a] ?? 0
            let pred = mu[ma] ?? 0
            for (let b = 0; b < pObs; b++) {
              const ob = obs[b] ?? 0
              pred += (bMat[a * pObs + b] ?? 0) * ((yComplete[ob] ?? 0) - (mu[ob] ?? 0))
            }
            yComplete[ma] = pred
          }

          for (let j = 0; j < cols; j++) {
            muNext[j] = (muNext[j] ?? 0) + (yComplete[j] ?? 0)
          }
          for (let j = 0; j < cols; j++) {
            const yj = yComplete[j] ?? 0
            for (let k = 0; k < cols; k++) {
              secondMoment[j * cols + k] = (secondMoment[j * cols + k] ?? 0) + yj * (yComplete[k] ?? 0)
            }
          }
          for (let a = 0; a < pMiss; a++) {
            const ma = miss[a] ?? 0
            for (let b = 0; b < pMiss; b++) {
              const mb = miss[b] ?? 0
              secondMoment[ma * cols + mb] = (secondMoment[ma * cols + mb] ?? 0) + (condCov[a * pMiss + b] ?? 0)
            }
          }
        }
      }

      let maxDiff = 0
      for (let j = 0; j < cols; j++) {
        const mNew = (muNext[j] ?? 0) / nActive
        maxDiff = Math.max(maxDiff, Math.abs(mNew - (mu[j] ?? 0)))
        mu[j] = mNew
      }
      for (let j = 0; j < cols; j++) {
        for (let k = 0; k < cols; k++) {
          const sNew = (secondMoment[j * cols + k] ?? 0) / nActive - (mu[j] ?? 0) * (mu[k] ?? 0)
          maxDiff = Math.max(maxDiff, Math.abs(sNew - (sigma[j * cols + k] ?? 0)))
          sigma[j * cols + k] = j === k ? Math.max(sNew, 1e-12) : sNew
        }
      }
      if (maxDiff < tol) break
    }
  } else {
    // Complete data: compute exact sample MLE moments in one pass.
    emIterations = 1
    sigma.fill(0)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const dj = (data[i * cols + j] ?? 0) - (mu[j] ?? 0)
        for (let k = 0; k < cols; k++) {
          const dk = (data[i * cols + k] ?? 0) - (mu[k] ?? 0)
          sigma[j * cols + k] = (sigma[j * cols + k] ?? 0) + (dj * dk) / rows
        }
      }
    }
  }

  // Little's d_0^2 uses the maximum-likelihood moments (not the N / (N - 1) rescaled covariance).
  const sigmaTilde = sigma

  let d02 = 0
  let sumPj = 0
  const breakdowns: LittlePatternBreakdown[] = []

  for (let pIdx = 0; pIdx < activePatterns.length; pIdx++) {
    const pat = activePatterns[pIdx]
    if (pat === undefined) continue
    const obs = pat.observedColumns
    const pj = obs.length
    sumPj += pj
    const sigOO = new Float64Array(pj * pj)
    const diff = new Float64Array(pj)
    for (let a = 0; a < pj; a++) {
      const ca = obs[a] ?? 0
      diff[a] = (pat.observedMeans.data[a] ?? 0) - (mu[ca] ?? 0)
      for (let b = 0; b < pj; b++) {
        const cb = obs[b] ?? 0
        sigOO[a * pj + b] = sigmaTilde[ca * cols + cb] ?? 0
      }
    }
    const invOO = invSpdOrPinv(sigOO, pj)
    let quad = 0
    for (let a = 0; a < pj; a++) {
      for (let b = 0; b < pj; b++) {
        quad += (diff[a] ?? 0) * (invOO[a * pj + b] ?? 0) * (diff[b] ?? 0)
      }
    }
    const dj2 = Math.max(0, pat.count * quad)
    d02 += dj2
    breakdowns.push({
      key: pat.key,
      observedColumns: pat.observedColumns,
      missingColumns: pat.missingColumns,
      count: pat.count,
      observedMeans: pat.observedMeans,
      distanceSquared: dj2,
    })
  }

  const df = Math.max(0, sumPj - cols)
  const stat = df === 0 ? 0 : d02
  const pVal = df === 0 ? 1 : chi2Sf(df, stat)
  const base = testResult({
    test: "Little's MCAR test",
    statistic: stat,
    pValue: pVal,
    alternative: 'greater',
    df,
    method: 'Gaussian EM moments',
  })

  return {
    ...base,
    patterns: breakdowns,
    pooledMean: vec(mu),
    pooledCovariance: mat(sigmaTilde, cols, cols),
    emIterations,
  }
}
