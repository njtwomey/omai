/**
 * Images and textures for rendering: linear floating-point images with conversion from and to display pixels, texture
 * wrapping and nearest or bilinear sampling, mipmap chains with level-of-detail selection and trilinear filtering,
 * texture-coordinate mappings (spherical, planar, cylindrical, cube map), and procedural textures (checkerboard,
 * bricks, keyed noise).
 */

import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { type Pixels, toPixels, type Vec3 } from '../scene/index.ts'
import { concreteOrThrow, fromStorage, type Value } from '../tensor/index.ts'
import { type NoiseOptions, perlinNoise, valueNoise } from '../worlds/index.ts'
import { dot3, sub3 } from './_vec3.ts'
import { acesFilm, exposure, gammaEncode, linearToSrgb, reinhard, srgbToLinear } from './tonemapping.ts'

/**
 * A floating-point image in linear RGB, row-major from the top-left, three values per pixel (and an optional alpha
 * per pixel). Textures and rendered frames both use it; {@link imageToPixels} turns it into display bytes.
 *
 * @concept texture-mapping
 */
export interface FloatImage {
  readonly width: number
  readonly height: number
  readonly rgb: Float64Array
  readonly alpha?: Float64Array
}

/** A colour with alpha, as the samplers return it. */
export type Rgba = readonly [number, number, number, number]

const size = (n: number, what: string): number => {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', `graphics: ${what} must be a positive integer`)
  return n
}

/**
 * A float image, black or filled by a function of the pixel's column and row.
 *
 * @param width - Width in pixels.
 * @param height - Height in pixels.
 * @param fill - The colour of pixel `(x, y)` (default black).
 * @returns The image.
 * @throws MdmlError with code `value` for a non-positive size.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { floatImage } from 'mdml-engine/graphics'
 * Array.from(floatImage(2, 1, (x) => [x, 0, 1]).rgb) // => [0, 0, 1, 1, 0, 1]
 * ```
 */
export function floatImage(width: number, height: number, fill?: (x: number, y: number) => Vec3): FloatImage {
  const rgb = new Float64Array(3 * size(width, 'width') * size(height, 'height'))
  if (fill !== undefined) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) rgb.set(fill(x, y), 3 * (y * width + x))
    }
  }
  return { width, height, rgb }
}

/**
 * A float image from display pixels: bytes scaled to [0, 1], decoded from sRGB to linear (unless `srgb` is false),
 * with alpha kept.
 *
 * @param pixels - The pixels.
 * @param options - Whether the bytes are sRGB-encoded (default true).
 * @returns The linear image.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { imageFromPixels } from 'mdml-engine/graphics'
 * const img = imageFromPixels({ width: 1, height: 1, rgba: new Uint8ClampedArray([255, 0, 0, 128]) })
 * [img.rgb[0], img.rgb[1], Math.round((img.alpha?.[0] ?? 0) * 255)] // => [1, 0, 128]
 * ```
 */
export function imageFromPixels(pixels: Pixels, options: { readonly srgb?: boolean } = {}): FloatImage {
  const n = pixels.width * pixels.height
  const rgb = new Float64Array(3 * n)
  const alpha = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) rgb[3 * i + c] = (pixels.rgba[4 * i + c] ?? 0) / 255
    alpha[i] = (pixels.rgba[4 * i + 3] ?? 255) / 255
  }
  const linear =
    options.srgb === false
      ? rgb
      : concreteOrThrow(srgbToLinear(fromStorage(rgb, [3 * n], 'float64')), 'imageFromPixels').data
  return { width: pixels.width, height: pixels.height, rgb: Float64Array.from(linear), alpha }
}

/** How {@link imageToPixels} turns radiance into display values. */
export interface DisplayOptions {
  /** Exposure in stops (default 0). */
  readonly exposure?: number
  /** Tone mapping (default `'none'`, which clamps to [0, 1]). */
  readonly toneMap?: 'none' | 'reinhard' | 'aces'
  /** Output encoding: `'srgb'` (default), `'linear'`, or a power-law gamma. */
  readonly encoding?: 'srgb' | 'linear' | number
}

/**
 * Display pixels from a linear float image: exposure, tone mapping, clamping, then sRGB (or gamma) encoding, and bytes
 * by rounding (through `scene`'s `toPixels`).
 *
 * @param img - The image.
 * @param options - Exposure, tone mapping and encoding.
 * @returns The pixels.
 * @concept tone-mapping
 * @example
 * ```ts
 * import { floatImage, imageToPixels } from 'mdml-engine/graphics'
 * Array.from(imageToPixels(floatImage(1, 1, () => [1, 0.5, 4]), { encoding: 'linear' }).rgba) // => [255, 128, 255, 255]
 * ```
 */
export function imageToPixels(img: FloatImage, options: DisplayOptions = {}): Pixels {
  let v: Value = fromStorage(Float64Array.from(img.rgb), [img.rgb.length], 'float64')
  if ((options.exposure ?? 0) !== 0) v = exposure(v, options.exposure ?? 0)
  if (options.toneMap === 'reinhard') v = reinhard(v)
  else if (options.toneMap === 'aces') v = acesFilm(v)
  const clamped = Float64Array.from(concreteOrThrow(v, 'imageToPixels').data, (x) => Math.min(1, Math.max(0, x)))
  const encoding = options.encoding ?? 'srgb'
  const encoded =
    encoding === 'linear'
      ? clamped
      : concreteOrThrow(
          encoding === 'srgb'
            ? linearToSrgb(fromStorage(clamped, [clamped.length], 'float64'))
            : gammaEncode(fromStorage(clamped, [clamped.length], 'float64'), encoding),
          'imageToPixels',
        ).data
  const n = img.width * img.height
  const c = img.alpha === undefined ? 3 : 4
  const data = new Float64Array(c * n)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) data[c * i + k] = encoded[3 * i + k] ?? 0
    if (c === 4) data[c * i + 3] = Math.min(1, Math.max(0, img.alpha?.[i] ?? 1))
  }
  return toPixels({ shape: [img.height, img.width, c], data }, { range: [0, 1] })
}

// ------------------------------------------------------------------------------------------------------ sampling

/** How texture coordinates outside [0, 1] are treated: tiled, clamped to the edge, or mirrored at each edge. */
export type Wrap = 'repeat' | 'clamp' | 'mirror'

/**
 * Wrap a texel index into `[0, n)`: `repeat` tiles (numpy's `wrap` padding), `clamp` repeats the edge texel (`edge`),
 * `mirror` reflects at each edge including the edge texel (`symmetric`, scipy's `reflect`).
 *
 * @param i - The index (any integer).
 * @param n - The number of texels.
 * @param wrap - The wrap mode.
 * @returns The index in `[0, n)`.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { wrapIndex } from 'mdml-engine/graphics'
 * [wrapIndex(-1, 4, 'repeat'), wrapIndex(-1, 4, 'clamp'), wrapIndex(-1, 4, 'mirror'), wrapIndex(5, 4, 'mirror')] // => [3, 0, 0, 2]
 * ```
 */
export function wrapIndex(i: number, n: number, wrap: Wrap): number {
  if (wrap === 'clamp') return i < 0 ? 0 : i >= n ? n - 1 : i
  if (wrap === 'repeat') return ((i % n) + n) % n
  const k = ((i % (2 * n)) + 2 * n) % (2 * n)
  return k < n ? k : 2 * n - 1 - k
}

function texel(tex: FloatImage, x: number, y: number, wrap: Wrap): Rgba {
  const i = wrapIndex(x, tex.width, wrap)
  const j = wrapIndex(y, tex.height, wrap)
  const p = j * tex.width + i
  return [tex.rgb[3 * p] ?? 0, tex.rgb[3 * p + 1] ?? 0, tex.rgb[3 * p + 2] ?? 0, tex.alpha?.[p] ?? 1]
}

/**
 * Nearest-texel sampling: texture coordinates `(u, v)` (u to the right, v down, the unit square covering the image)
 * pick texel `(floor(u w), floor(v h))`.
 *
 * @param tex - The texture.
 * @param u - Horizontal coordinate.
 * @param v - Vertical coordinate.
 * @param wrap - Wrap mode (default `'repeat'`).
 * @returns The texel colour and alpha.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { floatImage, sampleNearest } from 'mdml-engine/graphics'
 * const t = floatImage(2, 1, (x) => [x, x, x])
 * [sampleNearest(t, 0.4, 0)[0], sampleNearest(t, 0.6, 0)[0], sampleNearest(t, 1.4, 0)[0]] // => [0, 1, 0]
 * ```
 */
export function sampleNearest(tex: FloatImage, u: number, v: number, wrap: Wrap = 'repeat'): Rgba {
  return texel(tex, Math.floor(u * tex.width), Math.floor(v * tex.height), wrap)
}

/**
 * Bilinear sampling: texel centres sit at `((i + 0.5) / w, (j + 0.5) / h)`, and a sample blends the four nearest
 * texels by its distance to their centres (as scipy's `map_coordinates` with `order=1` at `(v h - 0.5, u w - 0.5)`).
 *
 * @param tex - The texture.
 * @param u - Horizontal coordinate.
 * @param v - Vertical coordinate.
 * @param wrap - Wrap mode (default `'repeat'`).
 * @returns The blended colour and alpha.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { floatImage, sampleBilinear } from 'mdml-engine/graphics'
 * sampleBilinear(floatImage(2, 1, (x) => [x, x, x]), 0.5, 0.5, 'clamp')[0] // => 0.5
 * ```
 */
export function sampleBilinear(tex: FloatImage, u: number, v: number, wrap: Wrap = 'repeat'): Rgba {
  const x = u * tex.width - 0.5
  const y = v * tex.height - 0.5
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const fx = x - x0
  const fy = y - y0
  const a = texel(tex, x0, y0, wrap)
  const b = texel(tex, x0 + 1, y0, wrap)
  const c = texel(tex, x0, y0 + 1, wrap)
  const d = texel(tex, x0 + 1, y0 + 1, wrap)
  const mix = (k: number): number => {
    const top = (a[k] ?? 0) + fx * ((b[k] ?? 0) - (a[k] ?? 0))
    const bottom = (c[k] ?? 0) + fx * ((d[k] ?? 0) - (c[k] ?? 0))
    return top + fy * (bottom - top)
  }
  return [mix(0), mix(1), mix(2), mix(3)]
}

/** A mipmap chain: level 0 is the texture, each next level half the size (at least 1 texel), box-filtered. */
export interface MipChain {
  readonly levels: readonly FloatImage[]
}

/**
 * Build a mipmap chain by repeated 2 x 2 box filtering down to 1 x 1 (an odd size folds its last texel into the last
 * box, so every texel counts).
 *
 * @param tex - The texture.
 * @returns The chain.
 * @concept mipmapping
 * @example
 * ```ts
 * import { floatImage, mipmaps } from 'mdml-engine/graphics'
 * const chain = mipmaps(floatImage(4, 2, (x) => [x, 0, 0]))
 * [chain.levels.length, chain.levels[2].rgb[0]] // => [3, 1.5]
 * ```
 */
export function mipmaps(tex: FloatImage): MipChain {
  const levels: FloatImage[] = [tex]
  let cur = tex
  while (cur.width > 1 || cur.height > 1) {
    const w = Math.max(1, Math.floor(cur.width / 2))
    const h = Math.max(1, Math.floor(cur.height / 2))
    const rgb = new Float64Array(3 * w * h)
    const alpha = cur.alpha === undefined ? undefined : new Float64Array(w * h)
    for (let j = 0; j < h; j++) {
      const y0 = Math.floor((j * cur.height) / h)
      const y1 = Math.floor(((j + 1) * cur.height) / h)
      for (let i = 0; i < w; i++) {
        const x0 = Math.floor((i * cur.width) / w)
        const x1 = Math.floor(((i + 1) * cur.width) / w)
        const acc = [0, 0, 0, 0]
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const t = texel(cur, x, y, 'clamp')
            for (let k = 0; k < 4; k++) acc[k] = (acc[k] ?? 0) + (t[k] ?? 0)
          }
        }
        const count = (x1 - x0) * (y1 - y0)
        for (let k = 0; k < 3; k++) rgb[3 * (j * w + i) + k] = (acc[k] ?? 0) / count
        if (alpha !== undefined) alpha[j * w + i] = (acc[3] ?? 0) / count
      }
    }
    cur = alpha === undefined ? { width: w, height: h, rgb } : { width: w, height: h, rgb, alpha }
    levels.push(cur)
  }
  return { levels }
}

/**
 * The level of detail for a pixel's texture footprint (the OpenGL rule): `log2` of the longer of the screen-space
 * derivatives of the texel coordinates, at least 0.
 *
 * @param dudx - Change of u per pixel step in x.
 * @param dvdx - Change of v per pixel step in x.
 * @param dudy - Change of u per pixel step in y.
 * @param dvdy - Change of v per pixel step in y.
 * @param width - Texture width in texels.
 * @param height - Texture height in texels.
 * @returns The level of detail.
 * @concept mipmapping
 * @example
 * ```ts
 * import { mipLevel } from 'mdml-engine/graphics'
 * mipLevel(4 / 64, 0, 0, 4 / 64, 64, 64) // => 2
 * ```
 */
export function mipLevel(
  dudx: number,
  dvdx: number,
  dudy: number,
  dvdy: number,
  width: number,
  height: number,
): number {
  const rho = Math.max(Math.hypot(dudx * width, dvdx * height), Math.hypot(dudy * width, dvdy * height))
  return rho > 1 ? Math.log2(rho) : 0
}

/**
 * Sample a mipmap chain at a level of detail: `'trilinear'` (default) blends bilinear samples of the two nearest
 * levels, `'nearest'` takes the bilinear sample of the nearest level.
 *
 * @param chain - The mipmap chain.
 * @param u - Horizontal coordinate.
 * @param v - Vertical coordinate.
 * @param lod - Level of detail (clamped to the chain).
 * @param options - Wrap mode and level filter.
 * @returns The colour and alpha.
 * @concept mipmapping
 * @example
 * ```ts
 * import { floatImage, mipmaps, sampleMipmap } from 'mdml-engine/graphics'
 * const chain = mipmaps(floatImage(2, 2, (x, y) => [x + 2 * y, 0, 0]))
 * [sampleMipmap(chain, 0.1, 0.1, 1)[0], sampleMipmap(chain, 0.25, 0.25, 0.5, { wrap: 'clamp' })[0]] // => [1.5, 0.75]
 * ```
 */
export function sampleMipmap(
  chain: MipChain,
  u: number,
  v: number,
  lod: number,
  options: { readonly wrap?: Wrap; readonly filter?: 'trilinear' | 'nearest' } = {},
): Rgba {
  const wrap = options.wrap ?? 'repeat'
  const top = chain.levels.length - 1
  const l = Math.min(top, Math.max(0, lod))
  const level = (k: number): FloatImage => chain.levels[k] ?? (chain.levels[0] as FloatImage)
  if (options.filter === 'nearest') return sampleBilinear(level(Math.round(l)), u, v, wrap)
  const lo = Math.floor(l)
  const hi = Math.min(top, lo + 1)
  const t = l - lo
  const a = sampleBilinear(level(lo), u, v, wrap)
  if (t === 0 || hi === lo) return a
  const b = sampleBilinear(level(hi), u, v, wrap)
  return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), a[3] + t * (b[3] - a[3])]
}

// ---------------------------------------------------------------------------------------------------- UV mapping

/**
 * Spherical (equirectangular) texture coordinates of a unit direction, y up: `u = 1/2 + atan2(z, x) / (2 pi)` around
 * the axis and `v = 1/2 - asin(y) / pi` from the north pole (v = 0) to the south (v = 1).
 *
 * @param d - A unit direction (or a normal of a sphere).
 * @returns `[u, v]`.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { sphereUv } from 'mdml-engine/graphics'
 * [sphereUv([1, 0, 0]), sphereUv([0, 1, 0])[1]] // => [[0.5, 0.5], 0]
 * ```
 */
export function sphereUv(d: Vec3): [number, number] {
  return [0.5 + Math.atan2(d[2], d[0]) / (2 * Math.PI), 0.5 - Math.asin(Math.max(-1, Math.min(1, d[1]))) / Math.PI]
}

/**
 * Planar texture coordinates: the coordinates of `p - origin` along two axes, `u = (p - o) . a / |a|^2` and
 * `v = (p - o) . b / |b|^2`, so the parallelogram on `a` and `b` covers the unit square.
 *
 * @param p - The point.
 * @param origin - Where `(0, 0)` lies.
 * @param uAxis - The edge along which u grows to 1.
 * @param vAxis - The edge along which v grows to 1.
 * @returns `[u, v]`.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { planarUv } from 'mdml-engine/graphics'
 * planarUv([1, 0, 3], [0, 0, 0], [2, 0, 0], [0, 0, 4]) // => [0.5, 0.75]
 * ```
 */
export function planarUv(p: Vec3, origin: Vec3, uAxis: Vec3, vAxis: Vec3): [number, number] {
  const d = sub3(p, origin)
  return [dot3(d, uAxis) / dot3(uAxis, uAxis), dot3(d, vAxis) / dot3(vAxis, vAxis)]
}

/**
 * Cylindrical texture coordinates about the y axis through `base`: `u = 1/2 + atan2(z, x) / (2 pi)` around, and
 * `v = 1 - (y - base_y) / height` down the side.
 *
 * @param p - The point.
 * @param base - The centre of the cylinder's base.
 * @param height - Its height.
 * @returns `[u, v]`.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { cylinderUv } from 'mdml-engine/graphics'
 * cylinderUv([-1, 1, 0], [0, 0, 0], 2) // => [1, 0.5]
 * ```
 */
export function cylinderUv(p: Vec3, base: Vec3, height: number): [number, number] {
  const d = sub3(p, base)
  return [0.5 + Math.atan2(d[2], d[0]) / (2 * Math.PI), 1 - d[1] / height]
}

/**
 * Cube-map lookup (the OpenGL convention): the face a direction points at (`0..5` for +x, -x, +y, -y, +z, -z) and the
 * coordinates on that face.
 *
 * @param d - A direction (any length, not zero).
 * @returns The face and `[u, v]` in the unit square.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { cubeUv } from 'mdml-engine/graphics'
 * cubeUv([1, 0, 0]) // => { face: 0, uv: [0.5, 0.5] }
 * cubeUv([0.5, -2, 1]).face // => 3
 * ```
 */
export function cubeUv(d: Vec3): { readonly face: number; readonly uv: [number, number] } {
  const [x, y, z] = d
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)
  let face: number
  let sc: number
  let tc: number
  let ma: number
  if (ax >= ay && ax >= az) {
    face = x >= 0 ? 0 : 1
    ma = ax
    sc = x >= 0 ? -z : z
    tc = -y
  } else if (ay >= az) {
    face = y >= 0 ? 2 : 3
    ma = ay
    sc = x
    tc = y >= 0 ? z : -z
  } else {
    face = z >= 0 ? 4 : 5
    ma = az
    sc = z >= 0 ? x : -x
    tc = -y
  }
  if (!(ma > 0)) throw new MdmlError('value', 'cubeUv: the direction must not be zero')
  return { face, uv: [(sc / ma + 1) / 2, (tc / ma + 1) / 2] }
}

// ------------------------------------------------------------------------------------------- procedural textures

/**
 * A checkerboard: squares of `scale` per unit of `(u, v)`, colour `a` where `floor(u s) + floor(v s)` is even.
 *
 * @param u - Horizontal coordinate.
 * @param v - Vertical coordinate.
 * @param options - Squares per unit (default 8) and the two colours (default white and dark grey).
 * @returns The colour.
 * @concept procedural-texture
 * @example
 * ```ts
 * import { checkerboard } from 'mdml-engine/graphics'
 * [checkerboard(0.05, 0.05)[0], checkerboard(0.2, 0.05)[0]] // => [1, 0.1]
 * ```
 */
export function checkerboard(
  u: number,
  v: number,
  options: { readonly scale?: number; readonly a?: Vec3; readonly b?: Vec3 } = {},
): Vec3 {
  const s = options.scale ?? 8
  const even = (Math.floor(u * s) + Math.floor(v * s)) % 2 === 0
  return even ? (options.a ?? [1, 1, 1]) : (options.b ?? [0.1, 0.1, 0.1])
}

/** Options of {@link bricks}. */
export interface BrickOptions {
  /** Brick rows per unit of v (default 4). */
  readonly rows?: number
  /** Bricks per unit of u in each row (default 2). */
  readonly cols?: number
  /** Mortar width as a fraction of a brick's height (default 0.08). */
  readonly mortar?: number
  /** How far odd rows are shifted, as a fraction of a brick (default 0.5). */
  readonly offset?: number
  readonly brick?: Vec3
  readonly mortarColour?: Vec3
}

/**
 * A brick wall: rows of bricks, odd rows shifted by `offset`, separated by mortar lines of width `mortar` (measured
 * in brick heights) along the top and left edge of each brick.
 *
 * @param u - Horizontal coordinate.
 * @param v - Vertical coordinate.
 * @param options - Layout and colours.
 * @returns The colour.
 * @concept procedural-texture
 * @example
 * ```ts
 * import { bricks } from 'mdml-engine/graphics'
 * [bricks(0.25, 0.125)[0], bricks(0.25, 0.005)[0]] // => [0.6, 0.8]
 * ```
 */
export function bricks(u: number, v: number, options: BrickOptions = {}): Vec3 {
  const rows = options.rows ?? 4
  const cols = options.cols ?? 2
  const mortar = options.mortar ?? 0.08
  const y = v * rows
  const row = Math.floor(y)
  const x = u * cols + (row % 2 !== 0 ? (options.offset ?? 0.5) : 0)
  const fy = y - row
  // The distance from the brick's left edge, measured in brick heights (a brick is rows / cols heights wide).
  const fx = (x - Math.floor(x)) * (rows / cols)
  const inMortar = fy < mortar || fx < mortar
  return inMortar ? (options.mortarColour ?? [0.8, 0.8, 0.8]) : (options.brick ?? [0.6, 0.2, 0.15])
}

/**
 * Bake a texture from a function of texture coordinates, sampled at texel centres `((x + 0.5) / w, (y + 0.5) / h)`.
 *
 * @param width - Width in texels.
 * @param height - Height in texels.
 * @param f - The colour at `(u, v)`.
 * @returns The texture.
 * @concept procedural-texture
 * @example
 * ```ts
 * import { bakeTexture, checkerboard } from 'mdml-engine/graphics'
 * const t = bakeTexture(4, 4, (u, v) => checkerboard(u, v, { scale: 2 }))
 * [t.rgb[0], t.rgb[3 * 2]] // => [1, 0.1]
 * ```
 */
export function bakeTexture(width: number, height: number, f: (u: number, v: number) => Vec3): FloatImage {
  return floatImage(width, height, (x, y) => f((x + 0.5) / width, (y + 0.5) / height))
}

/**
 * A keyed noise texture: value or Perlin noise (from `worlds`' lattice noise, with octaves), mapped to [0, 1] and
 * blended between two colours.
 *
 * @param key - The key.
 * @param width - Width in texels.
 * @param height - Height in texels.
 * @param options - Noise kind (default `'perlin'`), lattice scale and octaves, and the colours for 0 and 1.
 * @returns The texture.
 * @concept procedural-texture
 * @example
 * ```ts
 * import { noiseTexture } from 'mdml-engine/graphics'
 * import { key } from 'mdml-engine/random'
 * const t = noiseTexture(key(0), 8, 8, { scale: 4 })
 * [t.rgb.length, t.rgb.every((x) => x >= 0 && x <= 1)] // => [192, true]
 * ```
 */
export function noiseTexture(
  key: Key,
  width: number,
  height: number,
  options: NoiseOptions & { readonly kind?: 'value' | 'perlin'; readonly a?: Vec3; readonly b?: Vec3 } = {},
): FloatImage {
  const { kind, a = [0, 0, 0], b = [1, 1, 1], ...noise } = options
  const field =
    kind === 'value'
      ? valueNoise(key, size(height, 'height'), size(width, 'width'), noise)
      : perlinNoise(key, size(height, 'height'), size(width, 'width'), noise).map((x) => (x + 1) / 2)
  return floatImage(width, height, (x, y) => {
    const t = Math.min(1, Math.max(0, field[y * width + x] ?? 0))
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
  })
}
