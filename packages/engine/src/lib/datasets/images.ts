/**
 * Image datasets: rendered shapes for classification or detection, MNIST-like digit glyphs with their clean images,
 * MacKay's bars and stripes, geometric scenes with exact edge maps for edge and corner detectors, and a parser for the
 * IDX format with Fashion-MNIST's test split as a pinned remote asset.
 */

import { MdmlError, maxOf, minOf } from '../base/index.ts'
import { type DatasetGenerator, type DatasetRef, gunzip, type ImageData, images } from '../data/index.ts'
import { foldIn, normal, randint, uniform } from '../random/index.ts'
import { bool, choice, dict, int, real } from '../space/index.ts'
import { fromStorage } from '../tensor/index.ts'
import { stamp } from './_util.ts'

/** A shape to rasterise: a membership test in pixel coordinates and its bounding box. */
interface RasterShape {
  readonly inside: (x: number, y: number) => boolean
  readonly box: readonly [number, number, number, number]
}

/** Fraction of 2 by 2 sub-pixel samples inside a shape (light anti-aliasing). */
function coverage(shape: RasterShape, px: number, py: number): number {
  let c = 0
  for (const dx of [0.25, 0.75]) for (const dy of [0.25, 0.75]) if (shape.inside(px + dx, py + dy)) c++
  return c / 4
}

function polygon(points: readonly (readonly [number, number])[]): RasterShape {
  const inside = (x: number, y: number) => {
    let odd = false
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const [xi, yi] = points[i] as readonly [number, number]
      const [xj, yj] = points[j] as readonly [number, number]
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) odd = !odd
    }
    return odd
  }
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  return { inside, box: [minOf(xs), minOf(ys), maxOf(xs), maxOf(ys)] }
}

function regular(cx: number, cy: number, r: number, sides: number, angle: number): [number, number][] {
  return Array.from({ length: sides }, (_, i) => {
    const t = angle + (2 * Math.PI * i) / sides
    return [cx + r * Math.cos(t), cy + r * Math.sin(t)] as [number, number]
  })
}

const SHAPE_CLASSES = ['circle', 'square', 'triangle'] as const

/**
 * Images of simple shapes (circles, squares and triangles, rotated, at random positions and sizes) on a dark
 * background with optional pixel noise: one shape per image for classification (labels) or several for detection and
 * segmentation, with exact bounding boxes, box labels and per-pixel class masks (0 is background).
 *
 * @index dataset Images of circles, squares and triangles with labels.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { shapes } from 'mdml-engine/datasets'
 * generate(shapes, { knobs: { n: 4, size: 16 } }).data.images.shape // => [4, 16, 16, 1]
 * ```
 */
export const shapes: DatasetGenerator<ImageData> = {
  name: 'shapes',
  kind: 'image',
  description: 'Circles, squares and triangles with boxes and masks.',
  knobs: dict({
    n: int(1, 100_000, { label: 'images', description: 'The number of images.', default: 200 }),
    size: int(8, 512, {
      description: 'The side of each square image in pixels.',
      default: 28,
      label: 'image side in pixels',
    }),
    perImage: int(1, 10, { description: 'The number of shapes in each image.', default: 1, label: 'shapes per image' }),
    noise: real(0, 1, {
      description: 'The standard deviation of the pixel noise.',
      default: 0.05,
      label: 'pixel noise standard deviation',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const s = knobs['size'] as number
    const per = knobs['perImage'] as number
    const noise = knobs['noise'] as number
    const pixels = new Float64Array(n * s * s)
    const masks = new Int32Array(n * s * s)
    const boxes: number[][][] = []
    const boxLabels: number[][] = []
    const lab: number[] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      const cls = randint(foldIn(ki, 0), [per], 0, 3).data
      const geo = uniform(foldIn(ki, 1), [per, 4]).data
      const b: number[][] = []
      const bl: number[] = []
      for (let j = 0; j < per; j++) {
        const r = s * (0.12 + 0.18 * (geo[4 * j] ?? 0))
        const cx = r + (s - 2 * r) * (geo[4 * j + 1] ?? 0)
        const cy = r + (s - 2 * r) * (geo[4 * j + 2] ?? 0)
        const ang = 2 * Math.PI * (geo[4 * j + 3] ?? 0)
        const c = cls[j] ?? 0
        const shape: RasterShape =
          c === 0
            ? { inside: (x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r, box: [cx - r, cy - r, cx + r, cy + r] }
            : polygon(regular(cx, cy, r, c === 1 ? 4 : 3, ang))
        const brightness = 0.6 + 0.4 * ((j + 1) / per)
        for (let y = 0; y < s; y++)
          for (let x = 0; x < s; x++) {
            const cov = coverage(shape, x, y)
            if (cov > 0) {
              const idx = i * s * s + y * s + x
              pixels[idx] = Math.max(pixels[idx] ?? 0, cov * brightness)
              if (cov >= 0.5) masks[idx] = c + 1
            }
          }
        b.push([...shape.box])
        bl.push(c)
      }
      boxes.push(b)
      boxLabels.push(bl)
      lab.push(cls[0] ?? 0)
    }
    const z = normal(foldIn(k, 9), [n * s * s]).data
    const noisy = Float64Array.from(pixels, (v, i) => Math.min(1, Math.max(0, v + noise * (z[i] ?? 0))))
    return {
      data: images(fromStorage(noisy, [n, s, s, 1], 'float64'), {
        ...(per === 1 ? { labels: lab } : {}),
        classes: SHAPE_CLASSES,
        boxes: boxes.map((b) => fromStorage(Float64Array.from(b.flat()), [b.length, 4], 'float64')),
        boxLabels: boxLabels.map((b) => fromStorage(Int32Array.from(b), [b.length], 'int32')),
        masks: fromStorage(masks, [n, s, s], 'int32'),
        maskClasses: ['background', ...SHAPE_CLASSES],
        meta: stamp('shapes', shapes.description, knobs),
      }),
      truth: { latents: { clean: fromStorage(pixels, [n, s, s, 1], 'float64') }, noiseScale: noise },
    }
  },
}

/** A 5 by 7 bitmap font for the digits 0 to 9 (row strings, `1` is ink). */
const FONT: readonly (readonly string[])[] = [
  ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  ['11111', '00010', '00100', '00010', '00001', '10001', '01110'],
  ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
]

/**
 * Digit glyphs: the digits 0 to 9 from a 5 by 7 bitmap font, drawn with random scale, shift and slant and pixel noise,
 * a cheap MNIST-like task whose clean images are known (`truth.latents.clean`).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { digitGlyphs } from 'mdml-engine/datasets'
 * generate(digitGlyphs, { knobs: { n: 10 } }).data.classes?.length // => 10
 * ```
 */
export const digitGlyphs: DatasetGenerator<ImageData> = {
  name: 'digitGlyphs',
  kind: 'image',
  description: 'Bitmap-font digits with random scale, shift, slant and noise.',
  knobs: dict({
    n: int(1, 100_000, { label: 'images', description: 'The number of digit images.', default: 500 }),
    size: int(8, 128, { label: 'image side', description: 'The side of each square image in pixels.', default: 16 }),
    jitter: real(0, 1, {
      description: 'How much the scale, position and slant of each glyph vary.',
      default: 0.3,
      label: 'amount of random scale, shift and slant',
    }),
    noise: real(0, 1, {
      label: 'pixel noise',
      description: 'The standard deviation of the pixel noise.',
      default: 0.1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const s = knobs['size'] as number
    const j = knobs['jitter'] as number
    const noise = knobs['noise'] as number
    const digits = Array.from(randint(foldIn(k, 0), [n], 0, 10).data)
    const params = uniform(foldIn(k, 1), [n, 4], { low: -1, high: 1 }).data
    const clean = new Float64Array(n * s * s)
    for (let i = 0; i < n; i++) {
      const glyph = FONT[digits[i] ?? 0] as readonly string[]
      const scale = ((0.75 * s) / 7) * (1 + 0.2 * j * (params[4 * i] ?? 0))
      const ox = s / 2 + 0.15 * s * j * (params[4 * i + 1] ?? 0)
      const oy = s / 2 + 0.15 * s * j * (params[4 * i + 2] ?? 0)
      const slant = 0.3 * j * (params[4 * i + 3] ?? 0)
      for (let y = 0; y < s; y++)
        for (let x = 0; x < s; x++) {
          let ink = 0
          for (const dx of [0.25, 0.75])
            for (const dy of [0.25, 0.75]) {
              const gy = (y + dy - oy) / scale + 3.5
              const gx = (x + dx - ox) / scale + 2.5 + slant * (gy - 3.5)
              const r = Math.floor(gy)
              const c = Math.floor(gx)
              if (r >= 0 && r < 7 && c >= 0 && c < 5 && glyph[r]?.[c] === '1') ink++
            }
          clean[i * s * s + y * s + x] = ink / 4
        }
    }
    const z = normal(foldIn(k, 2), [n * s * s]).data
    const noisy = Float64Array.from(clean, (v, i) => Math.min(1, Math.max(0, v + noise * (z[i] ?? 0))))
    return {
      data: images(fromStorage(noisy, [n, s, s, 1], 'float64'), {
        labels: digits,
        classes: Array.from({ length: 10 }, (_, d) => String(d)),
        meta: stamp('digitGlyphs', digitGlyphs.description, knobs),
      }),
      truth: { latents: { clean: fromStorage(clean, [n, s, s, 1], 'float64') }, noiseScale: noise },
    }
  },
}

/**
 * MacKay's bars and stripes: `side` by `side` binary images in which every row is all on or all off (bars) or every
 * column is (stripes). With `all`, the dataset is every distinct pattern (`2^(side+1) - 2` of them, the blank and full
 * images once each); otherwise `n` uniform draws. Labels are the orientation; the truth has the exact log density
 * (uniform over the patterns) and the on bars of each image.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { barsAndStripes } from 'mdml-engine/datasets'
 * generate(barsAndStripes, { knobs: { side: 3, all: true } }).data.images.shape // => [14, 3, 3, 1]
 * ```
 */
export const barsAndStripes: DatasetGenerator<ImageData> = {
  name: 'barsAndStripes',
  kind: 'image',
  description: 'Binary images whose rows (bars) or columns (stripes) are each all on or all off.',
  knobs: dict({
    side: int(2, 12, { label: 'image side', description: 'The side of each square binary image.', default: 4 }),
    all: bool({
      description: 'Generate every distinct pattern once, instead of n random ones.',
      default: true,
      label: 'every distinct pattern',
    }),
    n: int(1, 100_000, {
      label: 'images',
      description: 'The number of images when not generating all patterns.',
      default: 100,
    }),
  }),
  generate: (k, knobs) => {
    const side = knobs['side'] as number
    const patterns: { orientation: number; bits: number }[] = []
    const full = (1 << side) - 1
    for (let b = 0; b <= full; b++) patterns.push({ orientation: 0, bits: b })
    for (let b = 1; b < full; b++) patterns.push({ orientation: 1, bits: b })
    const chosen =
      knobs['all'] === true
        ? patterns
        : Array.from(
            randint(k, [knobs['n'] as number], 0, patterns.length).data,
            (i) => patterns[i] as { orientation: number; bits: number },
          )
    const m = chosen.length
    const px = new Float64Array(m * side * side)
    for (const [i, p] of chosen.entries())
      for (let r = 0; r < side; r++)
        for (let c = 0; c < side; c++) {
          const line = p.orientation === 0 ? r : c
          px[i * side * side + r * side + c] = (p.bits >> line) & 1
        }
    const logP = -Math.log(patterns.length)
    return {
      data: images(fromStorage(px, [m, side, side, 1], 'float64'), {
        labels: chosen.map((p) => p.orientation),
        classes: ['bars', 'stripes'],
        meta: stamp('barsAndStripes', barsAndStripes.description, knobs),
      }),
      truth: {
        latents: { bits: chosen.map((p) => p.bits) },
        extra: { patterns: patterns.length, logProbability: logP },
      },
    }
  },
}

/**
 * Geometric scenes for edge and corner detectors: rectangles and triangles of distinct grey levels drawn over each
 * other on a background, with the exact edge map (pixels whose region differs from a 4-neighbour's) and the visible
 * polygon corners as truth, and the region of every pixel as masks.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { geometricScenes } from 'mdml-engine/datasets'
 * generate(geometricScenes, { knobs: { n: 2, size: 32 } }).data.masks?.shape // => [2, 32, 32]
 * ```
 */
export const geometricScenes: DatasetGenerator<ImageData> = {
  name: 'geometricScenes',
  kind: 'image',
  description: 'Overlapping rectangles and triangles with exact edges and corners.',
  knobs: dict({
    n: int(1, 10_000, { label: 'images', description: 'The number of scenes.', default: 20 }),
    size: int(16, 512, { label: 'image side', description: 'The side of each square image in pixels.', default: 64 }),
    polygons: int(1, 8, { label: 'polygons', description: 'Shapes per scene.', default: 3 }),
    noise: real(0, 1, {
      label: 'pixel noise',
      description: 'The standard deviation of the pixel noise.',
      default: 0.02,
    }),
    kinds: choice(['mixed', 'rectangles', 'triangles'], {
      label: 'shapes',
      description: 'Rectangles, triangles or a mix.',
      default: 'mixed',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const s = knobs['size'] as number
    const np = knobs['polygons'] as number
    const noise = knobs['noise'] as number
    const pixels = new Float64Array(n * s * s)
    const regions = new Int32Array(n * s * s)
    const edges = new Uint8Array(n * s * s)
    const corners: number[][][] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      const u = uniform(foldIn(ki, 0), [np, 7]).data
      const polys: [number, number][][] = []
      const levels: number[] = []
      for (let p = 0; p < np; p++) {
        const kind =
          knobs['kinds'] === 'rectangles' ? 0 : knobs['kinds'] === 'triangles' ? 1 : (u[7 * p] ?? 0) < 0.5 ? 0 : 1
        const pts: [number, number][] = []
        if (kind === 0) {
          const x0 = s * 0.1 + s * 0.5 * (u[7 * p + 1] ?? 0)
          const y0 = s * 0.1 + s * 0.5 * (u[7 * p + 2] ?? 0)
          const w = s * (0.15 + 0.3 * (u[7 * p + 3] ?? 0))
          const h = s * (0.15 + 0.3 * (u[7 * p + 4] ?? 0))
          pts.push([x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h])
        } else {
          for (let v = 0; v < 3; v++)
            pts.push([s * (0.05 + 0.9 * (u[7 * p + 1 + 2 * v] ?? 0)), s * (0.05 + 0.9 * (u[7 * p + 2 + 2 * v] ?? 0))])
        }
        polys.push(pts.map(([x, y]) => [Math.round(x) + 0.5, Math.round(y) + 0.5] as [number, number]))
        levels.push(0.25 + (0.75 * (p + 1)) / np)
      }
      const shapesHere = polys.map(polygon)
      for (let y = 0; y < s; y++)
        for (let x = 0; x < s; x++) {
          let region = 0
          for (const [p, sh] of shapesHere.entries()) if (sh.inside(x + 0.5, y + 0.5)) region = p + 1
          regions[i * s * s + y * s + x] = region
          pixels[i * s * s + y * s + x] = region === 0 ? 0.1 : (levels[region - 1] ?? 0)
        }
      for (let y = 0; y < s; y++)
        for (let x = 0; x < s; x++) {
          const r = regions[i * s * s + y * s + x]
          const differs =
            (x > 0 && regions[i * s * s + y * s + x - 1] !== r) ||
            (x + 1 < s && regions[i * s * s + y * s + x + 1] !== r) ||
            (y > 0 && regions[i * s * s + (y - 1) * s + x] !== r) ||
            (y + 1 < s && regions[i * s * s + (y + 1) * s + x] !== r)
          edges[i * s * s + y * s + x] = differs ? 1 : 0
        }
      // Corners that are not covered by a later (higher) polygon.
      const visible: number[][] = []
      for (const [p, pts] of polys.entries()) {
        for (const [x, y] of pts) {
          const covered = shapesHere.slice(p + 1).some((sh) => sh.inside(x, y))
          if (!covered && x >= 0 && y >= 0 && x < s && y < s) visible.push([x, y])
        }
      }
      corners.push(visible)
    }
    const z = normal(foldIn(k, 99), [n * s * s]).data
    const noisy = Float64Array.from(pixels, (v, i) => Math.min(1, Math.max(0, v + noise * (z[i] ?? 0))))
    return {
      data: images(fromStorage(noisy, [n, s, s, 1], 'float64'), {
        masks: fromStorage(regions, [n, s, s], 'int32'),
        meta: stamp('geometricScenes', geometricScenes.description, knobs),
      }),
      truth: {
        latents: {
          edges: fromStorage(edges, [n, s, s], 'bool'),
          corners,
          clean: fromStorage(pixels, [n, s, s, 1], 'float64'),
        },
        noiseScale: noise,
      },
    }
  },
}

const FASHION_CLASSES = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot',
]

/** Read an IDX file (MNIST format): magic number, dimensions, then unsigned bytes. */
function readIdx(bytes: Uint8Array): { shape: number[]; data: Uint8Array } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const magic = view.getUint32(0, false)
  if (magic >> 8 !== 0x08) throw new MdmlError('value', 'readIdx: not an unsigned-byte IDX file')
  const dims = magic & 0xff
  const shape = Array.from({ length: dims }, (_, d) => view.getUint32(4 + 4 * d, false))
  return { shape, data: bytes.subarray(4 + 4 * dims) }
}

/**
 * Parse gzipped IDX image and label files (the MNIST format) into an image dataset with pixels scaled to [0, 1].
 *
 * @param files - The gzipped images file and labels file.
 * @returns The images.
 * @throws MdmlError with code `value` for malformed files or mismatched counts.
 * @example
 * ```ts
 * import { parseIdxImages } from 'mdml-engine/datasets'
 * typeof parseIdxImages // => 'function'
 * ```
 */
export async function parseIdxImages(files: readonly Uint8Array[]): Promise<ImageData> {
  const [imgFile, labelFile] = files
  if (imgFile === undefined || labelFile === undefined)
    throw new MdmlError('value', 'parseIdxImages: needs images and labels')
  const img = readIdx(await gunzip(imgFile))
  const lab = readIdx(await gunzip(labelFile))
  const [n = 0, h = 0, w = 0] = img.shape
  if (lab.shape[0] !== n) throw new MdmlError('value', `parseIdxImages: ${n} images and ${lab.shape[0]} labels`)
  return images(
    fromStorage(
      Float64Array.from(img.data.subarray(0, n * h * w), (v) => v / 255),
      [n, h, w, 1],
      'float64',
    ),
    {
      labels: Array.from(lab.data.subarray(0, n)),
      classes: FASHION_CLASSES,
      meta: {
        name: 'fashionMnistTest',
        description: 'The 10000 test images of Fashion-MNIST: 28 by 28 grey Zalando article images in ten classes.',
        source: 'zalandoresearch/fashion-mnist t10k files',
        url: 'https://github.com/zalandoresearch/fashion-mnist',
        citation:
          'Xiao, Rasul and Vollgraf (2017), Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.',
        licence: 'MIT',
      },
    },
  )
}

/**
 * A small Fashion-MNIST (the 10000-image test split, 4.4 MB) as a pair of remote files pinned by their SHA-256.
 *
 * @example
 * ```ts
 * import { fashionMnistTest } from 'mdml-engine/datasets'
 * fashionMnistTest.kind // => 'remote-files'
 * ```
 */
export const fashionMnistTest: DatasetRef<ImageData> = {
  kind: 'remote-files',
  name: 'fashionMnistTest',
  files: [
    {
      url: 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
      sha256: '346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073',
      size: 4_422_102,
    },
    {
      url: 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
      sha256: '67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5',
      size: 5_148,
    },
  ],
  licence: 'MIT',
  parse: parseIdxImages,
}
