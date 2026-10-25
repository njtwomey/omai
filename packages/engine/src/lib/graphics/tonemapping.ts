/**
 * Tone mapping and colour encoding, which turn unbounded linear radiance into displayable values: exposure in stops,
 * Reinhard and ACES filmic curves, gamma encoding and decoding, the sRGB transfer functions, and relative luminance.
 */

import {
  add,
  clip,
  div,
  greater,
  maximum,
  mul,
  pow,
  promoteFloat,
  sub,
  sum,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { positivePow } from './_ops.ts'

/**
 * Scale radiance by an exposure in stops: `x 2^ev`.
 *
 * @param x - Radiance (any shape).
 * @param ev - Exposure value in stops.
 * @returns The scaled radiance.
 * @concept tone-mapping
 * @example
 * ```ts
 * import { exposure } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(exposure(0.25, 2)) // => 1
 * ```
 */
export function exposure(x: TensorLike, ev: TensorLike): Value {
  const [a, e] = promoteFloat(x, ev) as [Value, Value]
  return mul(a, pow(2, e))
}

/**
 * Reinhard's global tone-mapping operator `x / (1 + x)`, which squeezes unbounded radiance into [0, 1); with `white`
 * it is the extended form `x (1 + x / white^2) / (1 + x)`, which maps `white` to exactly 1. Differentiable.
 *
 * @param x - Radiance, at least 0 (any shape).
 * @param white - Optional smallest radiance mapped to white.
 * @returns Display values.
 * @concept tone-mapping
 * @example
 * ```ts
 * import { reinhard } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(reinhard(1)) // => 0.5
 * item(reinhard(4, 4)) // => 1
 * ```
 */
export function reinhard(x: TensorLike, white?: number): Value {
  const [a] = promoteFloat(x) as [Value]
  const base = div(a, add(a, 1))
  return white === undefined ? base : mul(base, add(div(a, white * white), 1))
}

/**
 * Narkowicz's fit of the ACES filmic tone curve, `x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14)`, clamped to [0, 1].
 *
 * @param x - Radiance, at least 0 (any shape).
 * @returns Display values in [0, 1].
 * @concept tone-mapping
 * @example
 * ```ts
 * import { acesFilm } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(acesFilm(0)) // => 0
 * item(acesFilm(100)) // => 1
 * ```
 */
export function acesFilm(x: TensorLike): Value {
  const [a] = promoteFloat(x) as [Value]
  const num = mul(a, add(mul(a, 2.51), 0.03))
  const den = add(mul(a, add(mul(a, 2.43), 0.59)), 0.14)
  return clip(div(num, den), 0, 1)
}

/**
 * Gamma encoding `x^(1 / gamma)` for display (gamma 2.2 by default), and {@link gammaDecode} its inverse.
 *
 * @param x - Linear values in [0, 1].
 * @param gamma - The display gamma.
 * @returns Encoded values.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { gammaEncode } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(gammaEncode(0.25, 2)) // => 0.5
 * ```
 */
export function gammaEncode(x: TensorLike, gamma = 2.2): Value {
  return positivePow(promoteFloat(x)[0] as Value, promoteFloat(1 / gamma)[0] as Value)
}

/**
 * Gamma decoding `x^gamma`, the inverse of {@link gammaEncode}.
 *
 * @param x - Encoded values in [0, 1].
 * @param gamma - The display gamma.
 * @returns Linear values.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { gammaDecode } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(gammaDecode(0.5, 2)) // => 0.25
 * ```
 */
export function gammaDecode(x: TensorLike, gamma = 2.2): Value {
  return positivePow(promoteFloat(x)[0] as Value, promoteFloat(gamma)[0] as Value)
}

/**
 * The sRGB transfer function (IEC 61966-2-1): `12.92 x` up to 0.0031308, then `1.055 x^(1 / 2.4) - 0.055`.
 * Differentiable.
 *
 * @param x - Linear values in [0, 1].
 * @returns sRGB-encoded values.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { linearToSrgb } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * Math.round(item(linearToSrgb(1)) * 1e12) / 1e12 // => 1
 * Math.round(item(linearToSrgb(0.214041140482232)) * 1e9) / 1e9 // => 0.5
 * ```
 */
export function linearToSrgb(x: TensorLike): Value {
  const [a] = promoteFloat(x) as [Value]
  const high = sub(mul(pow(maximum(a, 0.0031308), 1 / 2.4), 1.055), 0.055)
  return where(greater(a, 0.0031308), high, mul(a, 12.92))
}

/**
 * The inverse sRGB transfer function: `x / 12.92` up to 0.04045, then `((x + 0.055) / 1.055)^2.4`. Differentiable.
 *
 * @param x - sRGB-encoded values in [0, 1].
 * @returns Linear values.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { srgbToLinear } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * Math.round(item(srgbToLinear(0.5)) * 1e12) / 1e12 // => 0.214041140482
 * ```
 */
export function srgbToLinear(x: TensorLike): Value {
  const [a] = promoteFloat(x) as [Value]
  const high = pow(div(add(maximum(a, 0.04045), 0.055), 1.055), 2.4)
  return where(greater(a, 0.04045), high, div(a, 12.92))
}

/**
 * Relative luminance of linear RGB (Rec. 709 weights `0.2126 R + 0.7152 G + 0.0722 B`).
 *
 * @param rgb - Linear colours `[..., 3]`.
 * @returns Luminances `[...]`.
 * @concept tone-mapping
 * @example
 * ```ts
 * import { luminance } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * Math.round(item(luminance([1, 1, 1])) * 1e12) / 1e12 // => 1
 * ```
 */
export function luminance(rgb: TensorLike): Value {
  const [c] = promoteFloat(rgb) as [Value]
  return sum(mul(c, [0.2126, 0.7152, 0.0722]), -1)
}
