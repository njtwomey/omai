/**
 * Taking Gaussians out of the library: `gaussianEllipsoids` lists each Gaussian as plain numbers (centre, the three
 * axes of its one-standard-deviation ellipsoid, colour, opacity) for a view to draw in 3-D, and `gaussiansToPly` writes
 * the set in the binary `.ply` layout the 3-D Gaussian splatting viewers read.
 */

import { toFlat } from '../tensor/index.ts'
import { gaussianColour, gaussianCount, gaussianOpacity, type GaussianSet, quaternionToMatrix } from './gaussians.ts'

/** One Gaussian as the ellipsoid of its one-standard-deviation surface. */
export interface GaussianEllipsoid {
  readonly index: number
  readonly centre: readonly [number, number, number]
  /** The three principal axes as vectors in world units (direction times standard deviation), the longest first. */
  readonly axes: readonly [
    readonly [number, number, number],
    readonly [number, number, number],
    readonly [number, number, number],
  ]
  readonly rgb: readonly [number, number, number]
  readonly opacity: number
}

/**
 * The Gaussians of a set as ellipsoids: centre, principal axes (each the rotation's column times the standard deviation
 * along it, ordered longest first), colour (the degree-0 colour) and opacity.
 *
 * @param set - The Gaussians.
 * @returns One record per Gaussian.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussianEllipsoids, gaussiansFromPoints } from 'mdml-engine/fields'
 * const e = gaussianEllipsoids(gaussiansFromPoints([[1, 2, 3]], [[0.2, 0.4, 0.6]], { scale: 0.5 }))[0]
 * [e?.centre, e?.axes[0].map((v) => Math.abs(v))] // => [[1, 2, 3], [0.5, 0, 0]]
 * ```
 */
export function gaussianEllipsoids(set: GaussianSet): GaussianEllipsoid[] {
  const n = gaussianCount(set)
  const mean = toFlat(set.means)
  const scale = toFlat(set.logScales).map((v) => Math.exp(v))
  const rotation = toFlat(quaternionToMatrix(set.rotations))
  const colour = toFlat(gaussianColour(set))
  const opacity = toFlat(gaussianOpacity(set))
  return Array.from({ length: n }, (_, i) => {
    // The columns of the rotation are the Gaussian's own axes.
    const axes = [0, 1, 2]
      .map((k) => {
        const s = scale[3 * i + k] as number
        return [0, 1, 2].map((r) => (rotation[9 * i + 3 * r + k] as number) * s) as [number, number, number]
      })
      .sort((a, b) => Math.hypot(...b) - Math.hypot(...a)) as unknown as GaussianEllipsoid['axes']
    return {
      index: i,
      centre: [mean[3 * i] as number, mean[3 * i + 1] as number, mean[3 * i + 2] as number],
      axes,
      rgb: [colour[3 * i] as number, colour[3 * i + 1] as number, colour[3 * i + 2] as number],
      opacity: opacity[i] as number,
    }
  })
}

const PLY_PROPERTIES = [
  'x',
  'y',
  'z',
  'nx',
  'ny',
  'nz',
  'f_dc_0',
  'f_dc_1',
  'f_dc_2',
  'opacity',
  'scale_0',
  'scale_1',
  'scale_2',
  'rot_0',
  'rot_1',
  'rot_2',
  'rot_3',
] as const

/**
 * The set as a binary little-endian `.ply` file in the layout of the 3-D Gaussian splatting reference code: per Gaussian
 * `x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3` as 32-bit floats, where `f_dc` are the degree-0 spherical
 * harmonic coefficients, `opacity` the logit, `scale` the natural logarithm of the standard deviations and `rot` the
 * normalised quaternion `(w, x, y, z)`. The degree-1 coefficients are not written (the viewers read a missing
 * `f_rest` as zero). The coordinates are the library's (y up); a viewer that expects COLMAP's y-down shows the scene
 * upside down, which its rotate or flip control fixes.
 *
 * @param set - The Gaussians.
 * @returns The bytes of the file.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussiansToPly, randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const bytes = gaussiansToPly(randomGaussians(key(0), 3))
 * new TextDecoder().decode(bytes.slice(0, 3)) // => 'ply'
 * ```
 */
export function gaussiansToPly(set: GaussianSet): Uint8Array {
  const n = gaussianCount(set)
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${n}`,
    ...PLY_PROPERTIES.map((p) => `property float ${p}`),
    'end_header',
    '',
  ].join('\n')
  const head = new TextEncoder().encode(header)
  const stride = PLY_PROPERTIES.length * 4
  const out = new Uint8Array(head.length + n * stride)
  out.set(head)
  const view = new DataView(out.buffer)
  const mean = toFlat(set.means)
  const colour = toFlat(set.colours)
  const logit = toFlat(set.opacityLogits)
  const logScale = toFlat(set.logScales)
  const quaternion = toFlat(set.rotations)
  for (let i = 0; i < n; i++) {
    const norm = Math.hypot(
      quaternion[4 * i] as number,
      quaternion[4 * i + 1] as number,
      quaternion[4 * i + 2] as number,
      quaternion[4 * i + 3] as number,
    )
    const row = [
      mean[3 * i],
      mean[3 * i + 1],
      mean[3 * i + 2],
      0,
      0,
      0,
      colour[3 * i],
      colour[3 * i + 1],
      colour[3 * i + 2],
      logit[i],
      logScale[3 * i],
      logScale[3 * i + 1],
      logScale[3 * i + 2],
      ...[0, 1, 2, 3].map((k) => (quaternion[4 * i + k] as number) / norm),
    ] as number[]
    for (const [k, v] of row.entries()) view.setFloat32(head.length + i * stride + 4 * k, v, true)
  }
  return out
}
