/**
 * The two slots of a learnable spline: where its knots sit (`splineKnots`: evenly spaced and fixed, at quantiles of
 * the data and fixed, or free and trained with the weights) and what shape its coefficients are held to
 * (`splineShapes`: unconstrained, monotone increasing or decreasing, convex, concave, or bounded). Each is a table of
 * options with its knobs and properties, so a configuration names an option the way the library's optimisers and
 * schedules do. The shape options reparameterise the raw coefficients (cumulative softplus for monotone and convex
 * splines, a sigmoid for bounded ones), so the constraint holds exactly at every step of training.
 */

import { MdmlError } from '../../base/index.ts'
import { choice, dict, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import { placeKnots } from '../../stats/smoothing/index.ts'
import {
  add,
  asTensor,
  concat,
  cumsum,
  mul,
  neg,
  sigmoid,
  sliceAxis,
  softplus,
  sub,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'

/** How the interior knots of one feature are first placed: evenly, or at quantiles of a sample of the feature. */
export type KnotRule = 'uniform' | 'quantile'

/** A knot placement: the rule for the first placement, whether the knots then train, and the floor on their gaps. */
export interface KnotPlacement {
  readonly rule: KnotRule
  /** Whether the interior knots are parameters (free-knot splines) or fixed. */
  readonly learnable: boolean
  /** The least gap between neighbouring knots, as a share of the range (the free knots' floor). */
  readonly minGap: number
}

/** What a knot placement does, for validity rules and selectors. */
export interface KnotProperties {
  readonly learnable: boolean
  /** Whether placing the knots needs a sample of the feature. */
  readonly needsData: boolean
}

const placement = (
  properties: KnotProperties,
  description: string,
  space: Option<KnotPlacement, KnotProperties>['space'],
  make: Option<KnotPlacement, KnotProperties>['make'],
): Option<KnotPlacement, KnotProperties> => ({ description, space, properties, make })

/**
 * The knots of a learnable spline as one slot: `uniform` (evenly spaced, fixed; the default), `quantile` (at quantiles
 * of the data, fixed) and `learnable` (free knots, parameterised as a softmax of gaps so they stay ordered, trained
 * with the weights from a uniform or quantile start).
 *
 * @example
 * ```ts
 * import { splineKnots } from 'mdml-engine/nn'
 * splineKnots.make({ kind: 'learnable', init: 'quantile' }).learnable // => true
 * splineKnots.properties({ kind: 'quantile' }).needsData // => true
 * ```
 */
export const splineKnots: OptionTable<KnotPlacement, KnotProperties> = optionTable({
  uniform: placement({ learnable: false, needsData: false }, 'evenly spaced interior knots, fixed', dict({}), () => ({
    rule: 'uniform',
    learnable: false,
    minGap: 0,
  })),
  quantile: placement(
    { learnable: false, needsData: true },
    'interior knots at equally spaced quantiles of the feature, fixed',
    dict({}),
    () => ({ rule: 'quantile', learnable: false, minGap: 0 }),
  ),
  learnable: placement(
    { learnable: true, needsData: false },
    'free knots, trained with the coefficients (softmax of gaps, so they stay ordered)',
    dict({
      init: choice(['uniform', 'quantile'], { default: 'uniform', label: 'initial placement' }),
      minGap: real(0, 0.2, { default: 1e-3, label: 'least gap (share of the range)' }),
    }),
    (k) => ({
      rule: (k['init'] ?? 'uniform') as KnotRule,
      learnable: true,
      minGap: (k['minGap'] ?? 1e-3) as number,
    }),
  ),
})

/**
 * Place `count` interior knots strictly inside `(lo, hi)`: evenly, or at quantiles of the sample (falling back to
 * even spacing wherever the quantiles collide or touch a boundary, which heavy ties in the data cause).
 *
 * @param rule - `uniform` or `quantile`.
 * @param count - The number of interior knots.
 * @param lo - The lower boundary.
 * @param hi - The upper boundary.
 * @param sample - The feature's values (required for `quantile`).
 * @param minGap - Neighbouring knots at least this share of the range apart (default 1e-3).
 * @returns The interior knots, ascending.
 * @throws MdmlError `value` for `quantile` without a sample.
 * @example
 * ```ts
 * import { placeInterior } from 'mdml-engine/nn'
 * placeInterior('uniform', 3, 0, 1) // => [0.25, 0.5, 0.75]
 * placeInterior('quantile', 1, 0, 10, [0, 1, 2, 9, 10]) // => [2]
 * ```
 */
export function placeInterior(
  rule: KnotRule,
  count: number,
  lo: number,
  hi: number,
  sample?: ArrayLike<number>,
  minGap = 1e-3,
): number[] {
  const even = Array.from({ length: count }, (_, i) => lo + ((hi - lo) * (i + 1)) / (count + 1))
  if (count === 0 || rule === 'uniform') return even
  if (sample === undefined || sample.length === 0)
    throw new MdmlError('value', 'placeInterior: quantile knots need a sample')
  const placed = toFlat(asTensor(placeKnots(Array.from(sample), count + 2, { rule: 'quantile' }))).slice(1, -1)
  const edges = [lo, ...placed, hi]
  const apart = edges.every((e, i) => i === 0 || e - (edges[i - 1] as number) > minGap * (hi - lo))
  return apart ? placed : even
}

/**
 * A shape constraint on a spline's coefficients: a reparameterisation of the raw coefficients (last axis `K`) that
 * makes the constraint hold for every raw value. `apply` maps raw to effective coefficients given the Greville
 * abscissae of the knots; `initialise` shifts a concrete initial raw array so that training starts from a gentle curve.
 */
export interface ShapeTransform {
  readonly name: string
  apply(raw: TensorLike, greville: TensorLike): Value
  /** The shift added to raw entries from this index on at initialisation (so cumulative increments start small). */
  readonly smallFrom: number
  readonly smallShift: number
}

/** What a shape constraint guarantees, for validity rules and selectors. */
export interface ShapeProperties {
  readonly monotone: 'increasing' | 'decreasing' | undefined
  readonly curvature: 'convex' | 'concave' | undefined
  readonly bounded: boolean
}

const shape = (
  properties: ShapeProperties,
  description: string,
  space: Option<ShapeTransform, ShapeProperties>['space'],
  make: Option<ShapeTransform, ShapeProperties>['make'],
): Option<ShapeTransform, ShapeProperties> => ({ description, space, properties, make })

/** A shape transform whose `apply` takes tensor-likes and hands values to `f`. */
const transform = (
  name: string,
  f: (raw: Value, greville: Value) => Value,
  smallFrom: number,
  smallShift: number,
): ShapeTransform => ({ name, apply: (raw, g) => f(asTensor(raw), asTensor(g)), smallFrom, smallShift })

const last = (x: Value): number => x.shape.length - 1
const width = (x: Value): number => x.shape[last(x)] ?? 0
const head = (x: Value, from: number, to: number): Value => sliceAxis(x, last(x), from, to)

/** The cumulative sum along the last axis of `[first, increments...]`: a sequence starting at `first`. */
const accumulate = (first: Value, increments: Value | undefined): Value =>
  increments === undefined ? first : cumsum(concat([first, increments], last(first)), last(first))

/**
 * Coefficients whose slopes against the abscissae (last axis) start at the raw slope and change by `sign * softplus`
 * each step: convex for `sign` 1, concave for -1. The abscissae broadcast against the leading axes of the coefficients.
 */
function curvature(raw: Value, greville: Value, sign: 1 | -1): Value {
  const k = width(raw)
  if (k < 3) return raw
  const level = head(raw, 0, 1)
  const startSlope = head(raw, 1, 2)
  const bends = softplus(head(raw, 2, k))
  const slopes = accumulate(startSlope, sign === 1 ? bends : neg(bends))
  const ga = last(greville)
  const steps = sub(sliceAxis(greville, ga, 1, k), sliceAxis(greville, ga, 0, k - 1))
  return concat([level, add(level, cumsum(mul(slopes, steps), last(raw)))], last(raw))
}

/**
 * The shape constraints of a spline's coefficients as one slot: `none` (default), `increasing` and `decreasing`
 * (cumulative softplus increments: the spline is monotone), `convex` and `concave` (slopes of the coefficients against
 * the Greville abscissae change by softplus increments: the spline has a sign-definite second derivative) and `bounded`
 * (a sigmoid into `[lower, upper]`: the spline cannot leave the interval, as the basis is a partition of unity).
 *
 * @example
 * ```ts
 * import { splineShapes } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const up = splineShapes.make({ kind: 'increasing' })
 * toFlat(up.apply([0, -1, 5, -9], [0, 1, 2, 3])).every((c, i, a) => i === 0 || c >= a[i - 1]) // => true
 * ```
 */
export const splineShapes: OptionTable<ShapeTransform, ShapeProperties> = optionTable({
  none: shape(
    { monotone: undefined, curvature: undefined, bounded: false },
    'unconstrained coefficients',
    dict({}),
    () => transform('none', (raw) => raw, 0, 0),
  ),
  increasing: shape(
    { monotone: 'increasing', curvature: undefined, bounded: false },
    'non-decreasing coefficients, so a non-decreasing spline',
    dict({}),
    () =>
      transform(
        'increasing',
        (raw) => accumulate(head(raw, 0, 1), width(raw) > 1 ? softplus(head(raw, 1, width(raw))) : undefined),
        1,
        -3,
      ),
  ),
  decreasing: shape(
    { monotone: 'decreasing', curvature: undefined, bounded: false },
    'non-increasing coefficients, so a non-increasing spline',
    dict({}),
    () =>
      transform(
        'decreasing',
        (raw) => accumulate(head(raw, 0, 1), width(raw) > 1 ? neg(softplus(head(raw, 1, width(raw)))) : undefined),
        1,
        -3,
      ),
  ),
  convex: shape(
    { monotone: undefined, curvature: 'convex', bounded: false },
    'coefficient slopes that never decrease, so a convex spline',
    dict({}),
    () => transform('convex', (raw, g) => curvature(raw, g, 1), 2, -3),
  ),
  concave: shape(
    { monotone: undefined, curvature: 'concave', bounded: false },
    'coefficient slopes that never increase, so a concave spline',
    dict({}),
    () => transform('concave', (raw, g) => curvature(raw, g, -1), 2, -3),
  ),
  bounded: shape(
    { monotone: undefined, curvature: undefined, bounded: true },
    'coefficients in [lower, upper], so a spline in [lower, upper]',
    dict({
      lower: real(-1e6, 1e6, { default: 0, label: 'lower bound' }),
      upper: real(-1e6, 1e6, { default: 1, label: 'upper bound' }),
    }),
    (k) => {
      const lower = (k['lower'] ?? 0) as number
      const upper = (k['upper'] ?? 1) as number
      if (!(upper > lower)) throw new MdmlError('value', 'bounded: upper must exceed lower')
      return transform('bounded', (raw) => add(mul(sigmoid(raw), upper - lower), lower), 0, 0)
    },
  ),
})
