/**
 * Epistemic and aleatoric uncertainty of any regressor by a bootstrap ensemble, after deep ensembles (Lakshminarayanan
 * et al., 2017): members fitted on resamples disagree where the model is unsure (epistemic), while the residual noise
 * they share is aleatoric. Records both along the input.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { curveSet, type Explained } from '../../lib/exposition/index.ts'
import {
  type Estimator,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  requireVerb,
  takeData,
  targetVector,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, randint } from '../../lib/random/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'

/** Options of {@link ensembleUncertainty}. */
export interface EnsembleUncertaintyOptions {
  /** Bootstrap members (default 20). */
  readonly members?: number
  /** The feature the predictions are drawn along (default 0); the others sit at their means. */
  readonly feature?: number
  /** Grid points along it (default 120). */
  readonly points?: number
  /** How far beyond the data the grid reaches, as a share of its span on each side (default 0.25). */
  readonly extend?: number
  readonly key?: Key
}

/** The result of {@link ensembleUncertainty}. */
export interface EnsembleUncertainty {
  readonly members: readonly Fitted[]
  readonly grid: readonly number[]
  readonly mean: readonly number[]
  /** The members' spread at each grid point: epistemic (model) uncertainty, which more data would shrink. */
  readonly epistemic: readonly number[]
  /** The noise's standard deviation from out-of-bag residuals: aleatoric uncertainty, which more data would not. */
  readonly aleatoric: number
}

/**
 * Epistemic and aleatoric uncertainty of any regressor by a bootstrap ensemble (deep ensembles' idea, Lakshminarayanan
 * et al., 2017): fit members on resamples; their spread is the model's uncertainty, large away from the data, and the
 * out-of-bag residuals (less that spread) measure the noise, which no amount of data removes. Drawn along one feature
 * beyond the data's range. Exposition: a `curves` record of the ensemble mean with a band of two epistemic standard
 * deviations and one of two total, a few members, and the data's range marked.
 *
 * @param estimator - A regressor.
 * @param data - Labelled rows.
 * @param options - Members, feature, grid and key.
 * @returns The members, grid, mean and both uncertainties, with the record.
 */
export function ensembleUncertainty(
  estimator: Estimator,
  data: LearnData,
  options: EnsembleUncertaintyOptions = {},
): Explained<EnsembleUncertainty> {
  if (data.y === undefined) throw new WhiteboxError('value', 'ensembleUncertainty: the data needs targets')
  const m = matrixView(data.x, 'ensembleUncertainty')
  const y = targetVector(data.y, 'ensembleUncertainty', m.n)
  const key = options.key ?? makeKey(0)
  const count = options.members ?? 20
  const j = options.feature ?? 0
  const col = Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0)
  const lo = Math.min(...col)
  const hi = Math.max(...col)
  const ext = (options.extend ?? 0.25) * (hi - lo)
  const size = options.points ?? 120
  const grid = Array.from({ length: size }, (_, t) => lo - ext + ((hi - lo + 2 * ext) * t) / (size - 1))
  const means = Array.from({ length: m.d }, (_, c) => {
    let s = 0
    for (let i = 0; i < m.n; i++) s += m.data[i * m.d + c] ?? 0
    return s / m.n
  })
  const gx = new Float64Array(size * m.d)
  for (let t = 0; t < size; t++)
    for (let c = 0; c < m.d; c++) gx[t * m.d + c] = c === j ? (grid[t] ?? 0) : (means[c] ?? 0)
  const gridX = fromStorage(gx, [size, m.d], 'float64')
  const members: Fitted[] = []
  const curves: number[][] = []
  const oobSum = new Float64Array(m.n)
  const oobCount = new Float64Array(m.n)
  for (let b = 0; b < count; b++) {
    const rows = Array.from(randint(foldIn(key, b), [m.n], 0, m.n).data)
    const fitted = estimator.fit(takeData({ x: data.x, y }, rows), { key: foldIn(key, 1000 + b), explain: false })
    members.push(fitted)
    const predict = requireVerb(fitted, 'predict', 'ensembleUncertainty')
    curves.push(Array.from(toFlat(predict(gridX))))
    const inBag = new Set(rows)
    const p = toFlat(predict(data.x))
    for (let i = 0; i < m.n; i++)
      if (!inBag.has(i)) {
        oobSum[i] = (oobSum[i] ?? 0) + (p[i] ?? 0)
        oobCount[i] = (oobCount[i] ?? 0) + 1
      }
  }
  const mean = grid.map((_, t) => curves.reduce((s, c) => s + (c[t] ?? 0), 0) / count)
  const epistemic = grid.map((_, t) =>
    Math.sqrt(curves.reduce((s, c) => s + ((c[t] ?? 0) - (mean[t] ?? 0)) ** 2, 0) / Math.max(1, count - 1)),
  )
  let sq = 0
  let used = 0
  for (let i = 0; i < m.n; i++)
    if ((oobCount[i] ?? 0) > 0) {
      sq += ((y[i] ?? 0) - (oobSum[i] ?? 0) / (oobCount[i] ?? 1)) ** 2
      used++
    }
  const meanEpistemic = epistemic.reduce((s, v) => s + v * v, 0) / size
  const aleatoric = Math.sqrt(Math.max(0, sq / Math.max(1, used) - meanEpistemic))
  const total = epistemic.map((e) => Math.sqrt(e * e + aleatoric * aleatoric))
  const name = featureNamesOf(data, m.d)[j] ?? `x${j}`
  return {
    members,
    grid,
    mean,
    epistemic,
    aleatoric,
    exposition: [
      curveSet({
        x: grid,
        xLabel: name,
        yLabel: 'prediction',
        series: [
          {
            name: 'mean ± 2 total sd (noise too)',
            y: mean,
            lower: mean.map((v, t) => v - 2 * (total[t] ?? 0)),
            upper: mean.map((v, t) => v + 2 * (total[t] ?? 0)),
            group: 'total',
          },
          {
            name: 'mean ± 2 epistemic sd (the members disagree)',
            y: mean,
            lower: mean.map((v, t) => v - 2 * (epistemic[t] ?? 0)),
            upper: mean.map((v, t) => v + 2 * (epistemic[t] ?? 0)),
            group: 'epistemic',
          },
          ...curves.slice(0, 3).map((c, b) => ({ name: `member ${b + 1}`, y: c, group: 'members' })),
        ],
        marks: [{ x: lo }, { x: hi }],
      }),
    ],
  }
}
