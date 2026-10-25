/**
 * Isotonic regression (as in scikit-learn's `IsotonicRegression`): the monotone step function closest to the targets in
 * weighted least squares, found by the core's pool-adjacent-violators algorithm after merging tied inputs, with
 * interpolation between the knots for prediction.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { posteriorGrid } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type Fitted,
  type LearnData,
  poolAdjacentViolators,
  sampleWeightsOf,
  targetVector,
} from '../../lib/learn/index.ts'
import { interp } from '../../lib/numerics/interpolate/index.ts'
import { bool, dict, type Space } from '../../lib/space/index.ts'
import { asTensor, primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { vectorTensor } from './_design.ts'

/** Hyperparameters of {@link isotonicRegression}. */
export interface IsotonicHyper {
  /** Fit a non-decreasing (default) or non-increasing function. */
  readonly increasing: boolean
}

/** Parameters of a fitted isotonic regression: the step function's knots. */
export type IsotonicParams = { readonly x: number[]; readonly y: number[] }

/** Verbs of a fitted isotonic regression. */
export interface IsotonicVerbs {
  /** Linear interpolation between the knots, clipped to the end values outside them. */
  readonly predict: (x: TensorLike) => Tensor
}

/** A fitted isotonic regression. */
export type FittedIsotonic = Fitted<IsotonicParams> & IsotonicVerbs

function column(x: TensorLike, who: string): Float64Array {
  const t = asTensor(x)
  if (t.shape.length === 2 && t.shape[1] !== 1) throw new WhiteboxError('shape', `${who}: expects one feature`)
  return Float64Array.from(toFlat(primalOf(t)))
}

/**
 * Isotonic regression (sklearn.isotonic.IsotonicRegression): the monotone function closest to the targets in weighted
 * least squares, from the core's pool-adjacent-violators after merging tied inputs. Predictions interpolate linearly
 * between the knots and clip outside them. Exposition: the fitted function as a grid with the observations (zero
 * variance: the fit is not probabilistic).
 *
 * @param hyper - Direction (default increasing).
 * @returns The estimator.
 */
export function isotonicRegression(
  hyper: Partial<IsotonicHyper> = {},
): ExplainedEstimator<IsotonicHyper, FittedIsotonic> {
  return defineExplainedEstimator(
    {
      name: 'isotonic-regression',
      kind: 'regressor',
      hyperSpace: (h: IsotonicHyper): Space =>
        dict({
          increasing: bool({
            label: 'increasing',
            description: 'Fit a non-decreasing function (otherwise a non-increasing one).',
            default: h.increasing,
          }),
        }),
      fit: (data: LearnData, h: IsotonicHyper) => {
        const xs = column(data.x, 'isotonicRegression')
        const ys = targetVector(data.y ?? [], 'isotonicRegression', xs.length)
        const ws = sampleWeightsOf(data.weights, xs.length, 'isotonicRegression')
        const order = Array.from(xs.keys()).sort((a, b) => (xs[a] ?? 0) - (xs[b] ?? 0))
        // Merge tied inputs into their weighted mean (scikit-learn's _make_unique).
        const ux: number[] = []
        const uy: number[] = []
        const uw: number[] = []
        for (const i of order) {
          const xi = xs[i] ?? 0
          const last = ux.length - 1
          if (last >= 0 && ux[last] === xi) {
            const w = (uw[last] ?? 0) + (ws[i] ?? 0)
            uy[last] = ((uy[last] ?? 0) * (uw[last] ?? 0) + (ys[i] ?? 0) * (ws[i] ?? 0)) / w
            uw[last] = w
          } else {
            ux.push(xi)
            uy.push(ys[i] ?? 0)
            uw.push(ws[i] ?? 0)
          }
        }
        const fitted = h.increasing
          ? poolAdjacentViolators(uy, uw)
          : poolAdjacentViolators(
              uy.map((v) => -v),
              uw,
            ).map((v) => -v)
        const params: IsotonicParams = { x: ux, y: fitted }
        return {
          params,
          exposition: [
            posteriorGrid({
              inputs: ux.map((v) => [v]),
              mean: fitted,
              variance: fitted.map(() => 0),
              observed: { inputs: Array.from(xs, (v) => [v]), targets: Array.from(ys) },
            }),
          ],
        }
      },
      methods: (p: IsotonicParams) => ({
        predict: (x: TensorLike) =>
          vectorTensor(toFlat(primalOf(interp(vectorTensor(column(x, 'isotonicRegression')), p.x, p.y)))),
      }),
    },
    { increasing: hyper.increasing ?? true },
  )
}

/** {@link isotonicRegression} as a class. */
export class IsotonicRegression extends ExplainedEstimatorClass<IsotonicHyper, FittedIsotonic> {
  /** @param hyper - As {@link isotonicRegression}. */
  constructor(hyper: Partial<IsotonicHyper> = {}) {
    super(isotonicRegression(hyper))
  }
}
