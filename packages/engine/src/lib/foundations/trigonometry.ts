import { WhiteboxError } from '../base/index.ts'
import {
  abs,
  add,
  arccos,
  arctan2,
  cos,
  div,
  isfinite,
  linspace,
  logicalAnd,
  maximum,
  minimum,
  mod,
  mul,
  promoteFloat,
  sin,
  sqrt,
  stack,
  sub,
  type TensorLike,
  tan,
  toFlat,
  type Value,
} from '../tensor/index.ts'

const TAU = 2 * Math.PI

/**
 * Degrees to radians (numpy `radians`), differentiable.
 *
 * @param degrees - Angles in degrees.
 * @returns The angles in radians.
 * @concept angle-measure
 * @example
 * ```ts
 * import { radians } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(radians([180, 90])) // => [3.141592653589793, 1.5707963267948966]
 * ```
 */
export function radians(degrees: TensorLike): Value {
  const [d] = promoteFloat(degrees) as [Value]
  return mul(d, Math.PI / 180)
}

/**
 * Radians to degrees (numpy `degrees`), differentiable.
 *
 * @param radians - Angles in radians.
 * @returns The angles in degrees.
 * @concept angle-measure
 * @example
 * ```ts
 * import { degrees } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(degrees([Math.PI, Math.PI / 2])) // => [180, 90]
 * ```
 */
export function degrees(radians: TensorLike): Value {
  const [r] = promoteFloat(radians) as [Value]
  return mul(r, 180 / Math.PI)
}

/**
 * Wrap angles into one turn: `[-pi, pi)` ('signed', the convention of numpy's `(x + pi) % (2 pi) - pi`) or `[0, 2 pi)`
 * ('positive'). Differentiable away from the wrap points (the derivative is 1).
 *
 * @param theta - Angles in radians.
 * @param range - 'signed' (default) or 'positive'.
 * @returns The wrapped angles.
 * @concept angle-measure
 * @example
 * ```ts
 * import { wrapAngle } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(wrapAngle([3 * Math.PI / 2], 'signed')) // => [-1.5707963267948966]
 * toArray(wrapAngle([-Math.PI / 2], 'positive')) // => [4.71238898038469]
 * ```
 */
export function wrapAngle(theta: TensorLike, range: 'signed' | 'positive' = 'signed'): Value {
  const [t] = promoteFloat(theta) as [Value]
  return range === 'positive' ? mod(t, TAU) : sub(mod(add(t, Math.PI), TAU), Math.PI)
}

/**
 * The point (cos t, sin t) on the unit circle for each angle: the picture behind the definitions of sine and cosine.
 *
 * @param theta - Angles in radians, any shape `[...]`.
 * @returns Points `[..., 2]`.
 * @concept unit-circle
 * @example
 * ```ts
 * import { unitCirclePoint } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(unitCirclePoint(0)) // => [1, 0]
 * ```
 */
export function unitCirclePoint(theta: TensorLike): Value {
  const [t] = promoteFloat(theta) as [Value]
  return stack([cos(t), sin(t)], -1)
}

/**
 * The quadrant of an angle (after wrapping into `[0, 2 pi)`): 1 to 4 counter-clockwise from the positive x-axis, and 0
 * for an angle on an axis.
 *
 * @param theta - An angle in radians.
 * @returns 0, 1, 2, 3 or 4.
 * @concept unit-circle
 * @example
 * ```ts
 * import { quadrant } from 'whitebox-engine/foundations'
 * quadrant(2) // => 2
 * quadrant(-0.5) // => 4
 * quadrant(Math.PI) // => 0
 * ```
 */
export function quadrant(theta: number): 0 | 1 | 2 | 3 | 4 {
  const t = ((theta % TAU) + TAU) % TAU
  const quarter = t / (Math.PI / 2)
  const k = Math.round(quarter)
  if (Math.abs(quarter - k) < 1e-12) return 0
  return (Math.floor(quarter) + 1) as 1 | 2 | 3 | 4
}

/**
 * The reference angle: the acute angle between the terminal side of `theta` and the x-axis, in `[0, pi/2]`.
 *
 * @param theta - An angle in radians.
 * @returns The reference angle.
 * @concept unit-circle
 * @example
 * ```ts
 * import { referenceAngle } from 'whitebox-engine/foundations'
 * referenceAngle(5 * Math.PI / 6) // => 0.5235987755982989
 * ```
 */
export function referenceAngle(theta: number): number {
  const t = ((theta % Math.PI) + Math.PI) % Math.PI
  return Math.min(t, Math.PI - t)
}

/**
 * The direction angle of points `(x, y)` in `[0, 2 pi)`: the quadrant-aware inverse of the tangent (numpy `arctan2`,
 * shifted to one positive turn).
 *
 * @param y - Ordinates.
 * @param x - Abscissae.
 * @returns Angles in `[0, 2 pi)`.
 * @concept unit-circle
 * @example
 * ```ts
 * import { directionAngle } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(directionAngle([-1], [0])) // => [4.71238898038469]
 * ```
 */
export function directionAngle(y: TensorLike, x: TensorLike): Value {
  return mod(arctan2(y, x), TAU)
}

/** The three primary trigonometric functions. */
export type TrigFunction = 'sin' | 'cos' | 'tan'

const forward = (fn: TrigFunction, t: number): number =>
  fn === 'sin' ? Math.sin(t) : fn === 'cos' ? Math.cos(t) : Math.tan(t)

/**
 * The angle in a given quadrant with a given sine, cosine or tangent: the quadrant-aware inverse that the principal
 * branches (`asin` in `[-pi/2, pi/2]`, `acos` in `[0, pi]`, `atan`) cannot give on their own.
 *
 * @param fn - Which function the value belongs to.
 * @param value - The value of the function.
 * @param inQuadrant - The quadrant (1 to 4) the angle lies in.
 * @returns The angle in `[0, 2 pi)`.
 * @throws WhiteboxError with code `domain` when no angle in that quadrant has the value (wrong sign or out of range).
 * @concept inverse-trigonometric-functions
 * @example
 * ```ts
 * import { inverseInQuadrant } from 'whitebox-engine/foundations'
 * inverseInQuadrant('sin', 0.5, 2) // => 2.6179938779914944
 * inverseInQuadrant('tan', 1, 3) // => 3.9269908169872414
 * ```
 */
export function inverseInQuadrant(fn: TrigFunction, value: number, inQuadrant: 1 | 2 | 3 | 4): number {
  const ref =
    fn === 'tan'
      ? Math.atan(Math.abs(value))
      : Math.abs(value) <= 1
        ? fn === 'sin'
          ? Math.asin(Math.abs(value))
          : Math.acos(Math.abs(value))
        : Number.NaN
  if (Number.isNaN(ref)) throw new WhiteboxError('domain', `inverseInQuadrant: |${fn}| cannot be ${value}`)
  const angle = [ref, Math.PI - ref, Math.PI + ref, TAU - ref][inQuadrant - 1] ?? Number.NaN
  if (Math.abs(forward(fn, angle) - value) > 1e-12 * Math.max(1, Math.abs(value))) {
    throw new WhiteboxError(
      'domain',
      `inverseInQuadrant: ${fn} has the wrong sign for ${value} in quadrant ${inQuadrant}`,
    )
  }
  return angle
}

/**
 * Every solution of `fn(t) = value` in a closed interval, ascending: the principal value plus the symmetric and periodic
 * copies (sine: `t` and `pi - t`; cosine: `t` and `-t`; tangent: `t`; all plus multiples of the period).
 *
 * @param fn - The function.
 * @param value - The target value.
 * @param lo - Lower end of the interval (default 0).
 * @param hi - Upper end of the interval (default 2 pi).
 * @returns The solutions (empty when the value is out of range).
 * @concept inverse-trigonometric-functions
 * @example
 * ```ts
 * import { trigSolutions } from 'whitebox-engine/foundations'
 * trigSolutions('cos', 0.5).map((t) => t / Math.PI) // => [0.33333333333333337, 1.6666666666666665]
 * trigSolutions('sin', 2) // => []
 * ```
 */
export function trigSolutions(fn: TrigFunction, value: number, lo = 0, hi: number = TAU): number[] {
  if (fn !== 'tan' && !(Math.abs(value) <= 1)) return []
  const base =
    fn === 'sin'
      ? [Math.asin(value), Math.PI - Math.asin(value)]
      : fn === 'cos'
        ? [Math.acos(value), -Math.acos(value)]
        : [Math.atan(value)]
  const period = fn === 'tan' ? Math.PI : TAU
  const out: number[] = []
  for (const b of base) {
    for (let k = Math.ceil((lo - b) / period - 1e-12); b + k * period <= hi + 1e-12; k++) {
      const t = b + k * period
      if (t >= lo - 1e-12 && !out.some((s) => Math.abs(s - t) < 1e-12)) out.push(t)
    }
  }
  return out.sort((a, b) => a - b)
}

/** A trigonometric identity `lhs(t) = rhs(t)` (or of two angles, `lhs(a, b) = rhs(a, b)`), with its formula. */
export interface TrigIdentity {
  readonly name: string
  /** The identity in TeX. */
  readonly formula: string
  readonly lhs: (a: Value, b: Value) => Value
  readonly rhs: (a: Value, b: Value) => Value
}

const sq = (x: Value): Value => mul(x, x)

/**
 * The standard identities as checkable functions: Pythagorean, double and half angle, sum and difference, product to
 * sum, and the tangent forms. Identities of one angle ignore the second argument.
 *
 * @concept trigonometric-identities
 * @example
 * ```ts
 * import { trigIdentities } from 'whitebox-engine/foundations'
 * trigIdentities.map((i) => i.name).includes('double angle (sine)') // => true
 * ```
 */
export const trigIdentities: readonly TrigIdentity[] = [
  {
    name: 'Pythagorean',
    formula: '\\sin^2 t + \\cos^2 t = 1',
    lhs: (t) => add(sq(sin(t)), sq(cos(t))),
    rhs: (t) => add(mul(t, 0), 1),
  },
  {
    name: 'Pythagorean (tangent)',
    formula: '1 + \\tan^2 t = \\sec^2 t',
    lhs: (t) => add(1, sq(tan(t))),
    rhs: (t) => div(1, sq(cos(t))),
  },
  {
    name: 'double angle (sine)',
    formula: '\\sin 2t = 2 \\sin t \\cos t',
    lhs: (t) => sin(mul(t, 2)),
    rhs: (t) => mul(2, mul(sin(t), cos(t))),
  },
  {
    name: 'double angle (cosine)',
    formula: '\\cos 2t = \\cos^2 t - \\sin^2 t',
    lhs: (t) => cos(mul(t, 2)),
    rhs: (t) => sub(sq(cos(t)), sq(sin(t))),
  },
  {
    name: 'double angle (tangent)',
    formula: '\\tan 2t = \\frac{2 \\tan t}{1 - \\tan^2 t}',
    lhs: (t) => tan(mul(t, 2)),
    rhs: (t) => div(mul(2, tan(t)), sub(1, sq(tan(t)))),
  },
  {
    name: 'half angle (sine)',
    formula: '\\sin^2 \\tfrac{t}{2} = \\tfrac{1 - \\cos t}{2}',
    lhs: (t) => sq(sin(mul(t, 0.5))),
    rhs: (t) => mul(sub(1, cos(t)), 0.5),
  },
  {
    name: 'half angle (cosine)',
    formula: '\\cos^2 \\tfrac{t}{2} = \\tfrac{1 + \\cos t}{2}',
    lhs: (t) => sq(cos(mul(t, 0.5))),
    rhs: (t) => mul(add(1, cos(t)), 0.5),
  },
  {
    name: 'triple angle (sine)',
    formula: '\\sin 3t = 3 \\sin t - 4 \\sin^3 t',
    lhs: (t) => sin(mul(t, 3)),
    rhs: (t) => sub(mul(3, sin(t)), mul(4, mul(sin(t), sq(sin(t))))),
  },
  {
    name: 'sum (sine)',
    formula: '\\sin(a + b) = \\sin a \\cos b + \\cos a \\sin b',
    lhs: (a, b) => sin(add(a, b)),
    rhs: (a, b) => add(mul(sin(a), cos(b)), mul(cos(a), sin(b))),
  },
  {
    name: 'difference (sine)',
    formula: '\\sin(a - b) = \\sin a \\cos b - \\cos a \\sin b',
    lhs: (a, b) => sin(sub(a, b)),
    rhs: (a, b) => sub(mul(sin(a), cos(b)), mul(cos(a), sin(b))),
  },
  {
    name: 'sum (cosine)',
    formula: '\\cos(a + b) = \\cos a \\cos b - \\sin a \\sin b',
    lhs: (a, b) => cos(add(a, b)),
    rhs: (a, b) => sub(mul(cos(a), cos(b)), mul(sin(a), sin(b))),
  },
  {
    name: 'difference (cosine)',
    formula: '\\cos(a - b) = \\cos a \\cos b + \\sin a \\sin b',
    lhs: (a, b) => cos(sub(a, b)),
    rhs: (a, b) => add(mul(cos(a), cos(b)), mul(sin(a), sin(b))),
  },
  {
    name: 'sum (tangent)',
    formula: '\\tan(a + b) = \\frac{\\tan a + \\tan b}{1 - \\tan a \\tan b}',
    lhs: (a, b) => tan(add(a, b)),
    rhs: (a, b) => div(add(tan(a), tan(b)), sub(1, mul(tan(a), tan(b)))),
  },
  {
    name: 'product to sum (sine sine)',
    formula: '\\sin a \\sin b = \\tfrac{1}{2}(\\cos(a - b) - \\cos(a + b))',
    lhs: (a, b) => mul(sin(a), sin(b)),
    rhs: (a, b) => mul(sub(cos(sub(a, b)), cos(add(a, b))), 0.5),
  },
  {
    name: 'product to sum (cosine cosine)',
    formula: '\\cos a \\cos b = \\tfrac{1}{2}(\\cos(a - b) + \\cos(a + b))',
    lhs: (a, b) => mul(cos(a), cos(b)),
    rhs: (a, b) => mul(add(cos(sub(a, b)), cos(add(a, b))), 0.5),
  },
  {
    name: 'sum to product (sine)',
    formula: '\\sin a + \\sin b = 2 \\sin \\tfrac{a + b}{2} \\cos \\tfrac{a - b}{2}',
    lhs: (a, b) => add(sin(a), sin(b)),
    rhs: (a, b) => mul(2, mul(sin(mul(add(a, b), 0.5)), cos(mul(sub(a, b), 0.5)))),
  },
]

/** Result of {@link checkIdentity}. */
export interface IdentityCheck {
  /** Whether both sides agree within the tolerance at every finite sample. */
  readonly holds: boolean
  /** The largest |lhs - rhs| over the finite samples. */
  readonly maxError: number
  /** The number of samples where both sides are finite. */
  readonly samples: number
  /** The first sample (a, b) where the sides disagree, if any. */
  readonly counterexample: readonly [number, number] | null
}

/**
 * Check an identity numerically on a grid of angle pairs in `[lo, hi]^2` (samples where either side is not finite,
 * such as poles of the tangent, are skipped). Agreement on a grid is evidence, not proof; a single disagreement is a
 * counterexample.
 *
 * @param identity - Two functions of one or two angles.
 * @param options - The interval (default `[-pi, pi]`), points per axis (default 25) and tolerance (default 1e-9,
 *   relative to the size of the sides).
 * @returns Whether it holds, the largest error and any counterexample.
 * @concept trigonometric-identities
 * @example
 * ```ts
 * import { checkIdentity } from 'whitebox-engine/foundations'
 * import { cos, mul, sin } from 'whitebox-engine/tensor'
 * checkIdentity({ lhs: (t) => sin(mul(t, 2)), rhs: (t) => mul(2, sin(t)) }).holds // => false
 * checkIdentity({ lhs: (t) => sin(mul(t, 2)), rhs: (t) => mul(2, mul(sin(t), cos(t))) }).holds // => true
 * ```
 */
export function checkIdentity(
  identity: Pick<TrigIdentity, 'lhs' | 'rhs'>,
  options: { readonly lo?: number; readonly hi?: number; readonly n?: number; readonly tol?: number } = {},
): IdentityCheck {
  const n = options.n ?? 25
  const grid = toFlat(linspace(options.lo ?? -Math.PI, options.hi ?? Math.PI, n))
  const a = grid.flatMap((x) => grid.map(() => x))
  const b = grid.flatMap(() => grid)
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const l = identity.lhs(av, bv)
  const r = identity.rhs(av, bv)
  const err = toFlat(abs(sub(l, r)))
  const scale = toFlat(maximum(1, maximum(abs(l), abs(r))))
  const finite = toFlat(logicalAnd(isfinite(l), isfinite(r)))
  const tol = options.tol ?? 1e-9
  let maxError = 0
  let samples = 0
  let counterexample: [number, number] | null = null
  for (let i = 0; i < err.length; i++) {
    if (!finite[i]) continue
    const e = err[i] ?? 0
    // Near a pole both sides are huge and differ by rounding; the tolerance is relative there.
    if ((scale[i] ?? 1) > 1e8) continue
    samples++
    maxError = Math.max(maxError, e)
    if (counterexample === null && e > tol * (scale[i] ?? 1)) counterexample = [a[i] ?? 0, b[i] ?? 0]
  }
  return { holds: counterexample === null, maxError, samples, counterexample }
}

/**
 * The law of cosines: the side opposite an angle from the two sides that enclose it, `c^2 = a^2 + b^2 - 2ab cos C`.
 * Differentiable.
 *
 * @param a - One enclosing side.
 * @param b - The other enclosing side.
 * @param angleC - The included angle (radians).
 * @returns The opposite side c.
 * @concept law-of-cosines
 * @example
 * ```ts
 * import { lawOfCosines } from 'whitebox-engine/foundations'
 * import { item } from 'whitebox-engine/tensor'
 * item(lawOfCosines(3, 4, Math.PI / 2)) // => 5
 * ```
 */
export function lawOfCosines(a: TensorLike, b: TensorLike, angleC: TensorLike): Value {
  const [av, bv, cv] = promoteFloat(a, b, angleC) as [Value, Value, Value]
  // (a - b)^2 + 4ab sin^2(C/2): the same quantity without cancellation for small angles.
  const s = sin(mul(cv, 0.5))
  return sqrt(add(sq(sub(av, bv)), mul(4, mul(mul(av, bv), sq(s)))))
}

/**
 * The law of cosines solved for an angle: the angle opposite side c, `C = acos((a^2 + b^2 - c^2) / 2ab)`.
 * Differentiable inside the triangle inequality.
 *
 * @param a - A side next to the angle.
 * @param b - The other side next to the angle.
 * @param c - The side opposite the angle.
 * @returns The angle C (radians).
 * @concept law-of-cosines
 * @example
 * ```ts
 * import { lawOfCosinesAngle } from 'whitebox-engine/foundations'
 * import { item } from 'whitebox-engine/tensor'
 * item(lawOfCosinesAngle(1, 1, 1)) // => 1.0471975511965979
 * ```
 */
export function lawOfCosinesAngle(a: TensorLike, b: TensorLike, c: TensorLike): Value {
  const [av, bv, cv] = promoteFloat(a, b, c) as [Value, Value, Value]
  return arccos(div(sub(add(sq(av), sq(bv)), sq(cv)), mul(2, mul(av, bv))))
}

/**
 * The law of sines solved for a side: `b = a sin B / sin A`. Differentiable.
 *
 * @param a - A known side.
 * @param angleA - The angle opposite `a` (radians).
 * @param angleB - The angle opposite the wanted side (radians).
 * @returns The side b.
 * @concept law-of-sines
 * @example
 * ```ts
 * import { lawOfSines } from 'whitebox-engine/foundations'
 * import { item } from 'whitebox-engine/tensor'
 * item(lawOfSines(1, Math.PI / 6, Math.PI / 2)) // => 2.0000000000000004
 * ```
 */
export function lawOfSines(a: TensorLike, angleA: TensorLike, angleB: TensorLike): Value {
  const [av, A, B] = promoteFloat(a, angleA, angleB) as [Value, Value, Value]
  return div(mul(av, sin(B)), sin(A))
}

/**
 * Heron's formula for the area of a triangle from its three sides, in Kahan's stable arrangement (the sides sorted,
 * the factors grouped so that needle-like triangles keep their digits). Differentiable.
 *
 * @param a - A side.
 * @param b - A side.
 * @param c - A side.
 * @returns The area (NaN when the sides break the triangle inequality).
 * @concept herons-formula
 * @example
 * ```ts
 * import { heronArea } from 'whitebox-engine/foundations'
 * import { item } from 'whitebox-engine/tensor'
 * item(heronArea(3, 4, 5)) // => 6
 * ```
 */
export function heronArea(a: TensorLike, b: TensorLike, c: TensorLike): Value {
  const [x, y, z] = promoteFloat(a, b, c) as [Value, Value, Value]
  const hi = maximum(x, maximum(y, z))
  const lo = minimum(x, minimum(y, z))
  const mid = maximum(minimum(x, y), minimum(maximum(x, y), z))
  const p = mul(mul(add(hi, add(mid, lo)), sub(lo, sub(hi, mid))), mul(add(lo, sub(hi, mid)), add(hi, sub(mid, lo))))
  return mul(sqrt(p), 0.25)
}

/** A solved triangle: sides a, b, c opposite angles A, B, C (radians), with vertices for drawing. */
export interface Triangle {
  readonly a: number
  readonly b: number
  readonly c: number
  readonly A: number
  readonly B: number
  readonly C: number
  readonly area: number
  readonly perimeter: number
  /** Vertices A at the origin, B on the positive x-axis, C above it. */
  readonly vertices: readonly [readonly [number, number], readonly [number, number], readonly [number, number]]
}

/** Known parts of a triangle: any three of the sides a, b, c and the opposite angles A, B, C (radians). */
export interface TriangleParts {
  readonly a?: number
  readonly b?: number
  readonly c?: number
  readonly A?: number
  readonly B?: number
  readonly C?: number
}

/** How three known parts determine a triangle. */
export type TriangleCase = 'SSS' | 'SAS' | 'ASA' | 'AAS' | 'SSA'

const SIDES = ['a', 'b', 'c'] as const
const ANGLES = ['A', 'B', 'C'] as const

/**
 * Classify three known parts of a triangle: three sides (SSS), two sides and the angle between them (SAS), two angles
 * and the side between them (ASA), two angles and a side opposite one of them (AAS), or two sides and an angle opposite
 * one of them (SSA, the ambiguous case).
 *
 * @param parts - Exactly three known parts, at least one a side.
 * @returns The case.
 * @throws WhiteboxError with code `value` unless exactly three parts are given with at least one side.
 * @concept triangle-solving
 * @example
 * ```ts
 * import { triangleCase } from 'whitebox-engine/foundations'
 * triangleCase({ a: 3, b: 4, C: 1 }) // => 'SAS'
 * triangleCase({ a: 3, b: 4, A: 0.5 }) // => 'SSA'
 * triangleCase({ A: 1, B: 1, c: 2 }) // => 'ASA'
 * ```
 */
export function triangleCase(parts: TriangleParts): TriangleCase {
  const sides = SIDES.filter((s) => parts[s] !== undefined)
  const angles = ANGLES.filter((s) => parts[s] !== undefined)
  if (sides.length + angles.length !== 3 || sides.length === 0) {
    throw new WhiteboxError('value', 'triangleCase: give exactly three parts, at least one of them a side')
  }
  if (sides.length === 3) return 'SSS'
  const opposite = (s: (typeof SIDES)[number]): (typeof ANGLES)[number] => s.toUpperCase() as (typeof ANGLES)[number]
  if (sides.length === 2) {
    const angle = angles[0] as (typeof ANGLES)[number]
    return sides.some((s) => opposite(s) === angle) ? 'SSA' : 'SAS'
  }
  return angles.includes(opposite(sides[0] as (typeof SIDES)[number])) ? 'AAS' : 'ASA'
}

function build(a: number, b: number, c: number, A: number, B: number, C: number): Triangle {
  const area = 0.5 * b * c * Math.sin(A)
  return {
    a,
    b,
    c,
    A,
    B,
    C,
    area,
    perimeter: a + b + c,
    vertices: [
      [0, 0],
      [c, 0],
      [b * Math.cos(A), b * Math.sin(A)],
    ],
  }
}

/** Rotate a labelled triangle so that the part names move round: side i becomes side i + k. */
function rotateParts(parts: TriangleParts, k: number): TriangleParts {
  const out: Record<string, number> = {}
  for (let i = 0; i < 3; i++) {
    const j = (i + k) % 3
    const side = parts[SIDES[i] as (typeof SIDES)[number]]
    const angle = parts[ANGLES[i] as (typeof ANGLES)[number]]
    if (side !== undefined) out[SIDES[j] as string] = side
    if (angle !== undefined) out[ANGLES[j] as string] = angle
  }
  return out
}

function unrotate(t: Triangle, k: number): Triangle {
  const s = [t.a, t.b, t.c]
  const g = [t.A, t.B, t.C]
  const at = (xs: number[], i: number): number => xs[(i + k) % 3] ?? Number.NaN
  return build(at(s, 0), at(s, 1), at(s, 2), at(g, 0), at(g, 1), at(g, 2))
}

/**
 * Solve a triangle from three known parts (SSS, SAS, ASA, AAS or SSA): the remaining sides and angles by the laws of
 * cosines and sines. The ambiguous SSA case returns zero, one or two triangles.
 *
 * @param parts - Three known parts (sides positive, angles in radians).
 * @returns The triangles consistent with the parts (empty when none exists).
 * @throws WhiteboxError with code `value` when the parts do not name a case (see {@link triangleCase}).
 * @concept triangle-solving
 * @example
 * ```ts
 * import { solveTriangle } from 'whitebox-engine/foundations'
 * solveTriangle({ a: 3, b: 4, c: 5 })[0].C // => 1.5707963267948966
 * solveTriangle({ a: 6, b: 8, A: Math.PI / 6 }).length // => 2
 * solveTriangle({ a: 1, b: 8, A: Math.PI / 6 }).length // => 0
 * ```
 */
export function solveTriangle(parts: TriangleParts): Triangle[] {
  const kind = triangleCase(parts)
  const sides = SIDES.filter((s) => parts[s] !== undefined)
  const angles = ANGLES.filter((s) => parts[s] !== undefined)
  // Rotate the labels so that a fixed pattern applies, then rotate the answer back.
  const index = (name: string): number => Math.max(SIDES.indexOf(name as 'a'), ANGLES.indexOf(name as 'A'))
  let k = 0
  if (kind === 'SAS') k = (3 - index(angles[0] as string)) % 3 // the angle becomes A
  // ASA: the side becomes a, so the known angles are B and C; AAS: the side becomes a.
  if (kind === 'ASA' || kind === 'AAS') k = (3 - index(sides[0] as string)) % 3
  if (kind === 'SSA') k = (3 - index(angles[0] as string)) % 3 // the angle becomes A, its side a
  const p = rotateParts(parts, k)
  const finish = (ts: Triangle[]): Triangle[] =>
    ts
      .filter((t) => [t.a, t.b, t.c, t.A, t.B, t.C].every((v) => Number.isFinite(v) && v > 0))
      .map((t) => unrotate(t, k))
  const angleOf = (x: number, y: number, z: number): number =>
    Math.acos(Math.max(-1, Math.min(1, (x * x + y * y - z * z) / (2 * x * y))))
  if (kind === 'SSS') {
    const { a = 0, b = 0, c = 0 } = p
    if (a + b <= c || a + c <= b || b + c <= a) return []
    const A = angleOf(b, c, a)
    const B = angleOf(a, c, b)
    return finish([build(a, b, c, A, B, Math.PI - A - B)])
  }
  if (kind === 'SAS') {
    const { b = 0, c = 0, A = 0 } = p
    if (!(A > 0 && A < Math.PI)) return []
    const a = Math.sqrt((b - c) ** 2 + 4 * b * c * Math.sin(A / 2) ** 2)
    const B = angleOf(a, c, b)
    return finish([build(a, b, c, A, B, Math.PI - A - B)])
  }
  if (kind === 'ASA' || kind === 'AAS') {
    const A = p.A ?? Math.PI - (p.B ?? 0) - (p.C ?? 0)
    const B = p.B ?? Math.PI - A - (p.C ?? 0)
    const C = Math.PI - A - B
    if (!(A > 0 && B > 0 && C > 0)) return []
    const a = p.a ?? 0
    return finish([build(a, (a * Math.sin(B)) / Math.sin(A), (a * Math.sin(C)) / Math.sin(A), A, B, C)])
  }
  // SSA: a and A known with one more side; make it b.
  const A = p.A ?? 0
  const a = p.a ?? 0
  const other = p.b ?? p.c ?? 0
  const sinB = (other * Math.sin(A)) / a
  if (!(A > 0 && A < Math.PI) || sinB > 1 + 1e-15) return []
  const b1 = Math.asin(Math.min(1, sinB))
  const candidates = Math.abs(sinB - 1) < 1e-15 ? [b1] : [b1, Math.PI - b1]
  const out: Triangle[] = []
  for (const B of candidates) {
    const C = Math.PI - A - B
    if (!(C > 1e-15)) continue
    const c = (a * Math.sin(C)) / Math.sin(A)
    out.push(p.b !== undefined ? build(a, other, c, A, B, C) : build(a, c, other, A, C, B))
  }
  return finish(out)
}
