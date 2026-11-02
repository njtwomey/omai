/**
 * Differentiable volume rendering, the emission-absorption model of NeRF: along a ray with sample distances `t_i`,
 * densities `sigma_i` and colours `c_i`, the opacity of a sample is `alpha_i = 1 - exp(-sigma_i delta_i)`, the light
 * reaching it is the transmittance `T_i = exp(-sum_{j<i} sigma_j delta_j)`, and the pixel's colour is the sum of
 * `w_i c_i` with `w_i = T_i alpha_i`. Everything is built from the library's differentiable tensor operations, so the
 * gradient flows to the densities and colours (and so to the network that produced them) without a custom backward pass.
 */

import { MdmlError } from '../base/index.ts'
import {
  add,
  asTensor,
  concat,
  cumsum,
  exp,
  expandDims,
  fromStorage,
  mul,
  neg,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Value,
  zeros,
} from '../tensor/index.ts'

/** What {@link volumeRender} returns. */
export interface VolumeRender {
  /** The rendered colour `[rays, 3]`. */
  readonly colour: Value
  /** The expected distance along the ray, `sum w_i t_i` `[rays]` (with the accumulated opacity, a depth map). */
  readonly depth: Value
  /** The accumulated opacity `sum w_i` `[rays]`: 0 for empty space, 1 for a fully opaque ray. */
  readonly alpha: Value
  /** The weights `w_i = T_i (1 - exp(-sigma_i delta_i))` `[rays, n]`. */
  readonly weights: Value
  /** The transmittance `T_i` reaching each sample `[rays, n]`. */
  readonly transmittance: Value
}

/** Options of {@link volumeRender}. */
export interface VolumeOptions {
  /** The spacing of each sample `[rays, n]`; by default the gaps between consecutive distances, then `lastDelta`. */
  readonly delta?: Value
  /** The last sample's spacing when `delta` is not given (default 1e10: it absorbs whatever is left, as NeRF does). */
  readonly lastDelta?: number
  /** Composite onto white: add `1 - alpha` to every channel (the synthetic NeRF scenes). */
  readonly whiteBackground?: boolean
  /** Composite onto this colour instead (a triple); wins over `whiteBackground`. */
  readonly background?: readonly [number, number, number]
}

/**
 * Render rays by the volume rendering equation. Differentiable in `density`, `colour` and (through the spacings) `t`.
 *
 * @param density - The densities `sigma_i >= 0` `[rays, n]`.
 * @param colour - The colours `c_i` `[rays, n, 3]`.
 * @param t - The sample distances `[rays, n]` (increasing), for the depth and the spacings.
 * @param options - Spacings, the last spacing, and the background.
 * @returns The colour, depth, accumulated opacity, weights and transmittance.
 * @throws MdmlError with code `shape` when the shapes do not agree.
 * @concept volume-rendering
 * @reference Mildenhall et al. (2020), NeRF: representing scenes as neural radiance fields for view synthesis.
 * @example
 * ```ts
 * import { volumeRender } from 'mdml-engine/fields'
 * import { toArray, tensor } from 'mdml-engine/tensor'
 * // Constant density 1 over a unit-length ray (two samples, spacing 0.5): alpha = 1 - exp(-1).
 * const r = volumeRender(tensor([[1, 1]]), tensor([[[1, 0, 0], [1, 0, 0]]]), tensor([[0, 0.5]]), { delta: tensor([[0.5, 0.5]]) })
 * [(toArray(r.alpha) as number[])[0].toFixed(6), (toArray(r.colour) as number[][])[0].map((v) => v.toFixed(3))] // => ['0.632121', ['0.632', '0.000', '0.000']]
 * ```
 */
export function volumeRender(density: Value, colour: Value, t: Value, options: VolumeOptions = {}): VolumeRender {
  const [rays, n] = density.shape
  if (density.shape.length !== 2 || rays === undefined || n === undefined)
    throw new MdmlError('shape', `volumeRender: densities must be [rays, n], got [${density.shape.join(', ')}]`)
  if (colour.shape.length !== 3 || colour.shape[0] !== rays || colour.shape[1] !== n || colour.shape[2] !== 3)
    throw new MdmlError('shape', `volumeRender: colours must be [${rays}, ${n}, 3], got [${colour.shape.join(', ')}]`)
  if (t.shape.length !== 2 || t.shape[0] !== rays || t.shape[1] !== n)
    throw new MdmlError('shape', `volumeRender: distances must be [${rays}, ${n}], got [${t.shape.join(', ')}]`)
  const delta =
    options.delta ??
    (n === 1
      ? fromStorage(new Float64Array(rays).fill(options.lastDelta ?? 1e10), [rays, 1], 'float64')
      : concat(
          [
            sub(sliceAxis(t, 1, 1, n), sliceAxis(t, 1, 0, n - 1)),
            fromStorage(new Float64Array(rays).fill(options.lastDelta ?? 1e10), [rays, 1], 'float64'),
          ],
          1,
        ))
  const optical = mul(density, delta)
  // The optical depth in front of each sample: the running sum of the samples before it. (Summing the shifted values,
  // rather than subtracting a sample from the inclusive sum, keeps the huge last spacing from swamping the others.)
  const before = n === 1 ? zeros([rays, 1]) : concat([zeros([rays, 1]), sliceAxis(optical, 1, 0, n - 1)], 1)
  const transmittance = exp(neg(cumsum(before, 1)))
  const weights = mul(transmittance, sub(1, exp(neg(optical))))
  const alpha = sum(weights, 1)
  let rgb = sum(mul(expandDims(weights, -1), colour), 1)
  if (options.background !== undefined)
    rgb = add(rgb, mul(expandDims(sub(1, alpha), -1), asTensor([...options.background]) as Value))
  else if (options.whiteBackground === true) rgb = add(rgb, reshape(sub(1, alpha), [rays, 1]))
  return { colour: rgb, depth: sum(mul(weights, t), 1), alpha, weights, transmittance }
}
