/**
 * A dependency-free animated GIF encoder: `quantise` builds a palette of up to 256 colours by median cut over the
 * frames' pixels, and `encodeGif` maps each frame to it, LZW-compresses the indices and writes the frame delays and
 * looping.
 */

import { WhiteboxError } from 'whitebox-engine/base'
import type { Pixels } from 'whitebox-engine/scene'

/** A palette: up to 256 colours as `[r, g, b]`. */
export type Palette = readonly (readonly [number, number, number])[]

/**
 * A palette of at most `size` colours for a set of images, by median cut over a 5-bit-per-channel histogram of their
 * pixels (each box split along its widest channel at the weighted median; colours are the boxes' weighted means).
 * Transparent pixels are ignored.
 *
 * @param images - The images (all their pixels are counted).
 * @param size - The most colours (default 256).
 * @returns The palette, at least one colour.
 * @example
 * ```ts
 * import { quantise } from 'whitebox-render/animation'
 * const px = { width: 2, height: 1, rgba: new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 255]) }
 * quantise([px]).length // => 2
 * ```
 */
export function quantise(images: readonly Pixels[], size = 256): Palette {
  const counts = new Float64Array(1 << 15)
  // The colours in each bin are summed, so a palette colour is the true mean of its pixels.
  const sums = [new Float64Array(1 << 15), new Float64Array(1 << 15), new Float64Array(1 << 15)] as const
  for (const im of images) {
    const d = im.rgba
    for (let i = 0; i < d.length; i += 4) {
      if ((d[i + 3] as number) < 128) continue
      const [r, g, b] = [d[i] as number, d[i + 1] as number, d[i + 2] as number]
      const bin = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
      counts[bin] = (counts[bin] as number) + 1
      sums[0][bin] = (sums[0][bin] as number) + r
      sums[1][bin] = (sums[1][bin] as number) + g
      sums[2][bin] = (sums[2][bin] as number) + b
    }
  }
  const bins: number[] = []
  for (let b = 0; b < counts.length; b++) if ((counts[b] as number) > 0) bins.push(b)
  if (bins.length === 0) return [[0, 0, 0]]
  const channel = (b: number, c: number): number => (b >> (10 - 5 * c)) & 31
  type Box = number[]
  const boxes: Box[] = [bins]
  const spread = (box: Box): [number, number] => {
    let best = 0
    let widest = -1
    for (let c = 0; c < 3; c++) {
      let lo = 31
      let hi = 0
      for (const b of box) {
        const v = channel(b, c)
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
      if (hi - lo > widest) {
        widest = hi - lo
        best = c
      }
    }
    return [best, widest]
  }
  while (boxes.length < size) {
    // Split the box with the most pixels among those that can still split.
    let pick = -1
    let most = 0
    for (let k = 0; k < boxes.length; k++) {
      const box = boxes[k] as Box
      if (box.length < 2) continue
      const weight = box.reduce((a, b) => a + (counts[b] as number), 0)
      if (weight > most) {
        most = weight
        pick = k
      }
    }
    if (pick < 0) break
    const box = boxes[pick] as Box
    const [c] = spread(box)
    box.sort((a, b) => channel(a, c) - channel(b, c))
    let acc = 0
    let cut = 1
    for (let k = 0; k < box.length - 1; k++) {
      acc += counts[box[k] as number] as number
      if (acc >= most / 2) {
        cut = k + 1
        break
      }
      cut = k + 1
    }
    boxes.splice(pick, 1, box.slice(0, cut), box.slice(cut))
  }
  return boxes.map((box) => {
    let w = 0
    const sum = [0, 0, 0]
    for (const b of box) {
      w += counts[b] as number
      for (let c = 0; c < 3; c++) sum[c] = (sum[c] as number) + (sums[c as 0 | 1 | 2][b] as number)
    }
    return [Math.round((sum[0] as number) / w), Math.round((sum[1] as number) / w), Math.round((sum[2] as number) / w)]
  })
}

/** Nearest palette index for every 5-bit-per-channel bin, computed lazily. */
function indexer(palette: Palette): (r: number, g: number, b: number) => number {
  const cache = new Int16Array(1 << 15).fill(-1)
  return (r, g, b) => {
    const bin = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
    const known = cache[bin] as number
    if (known >= 0) return known
    const [cr, cg, cb] = [(r >> 3) * 8 + 4, (g >> 3) * 8 + 4, (b >> 3) * 8 + 4]
    let best = 0
    let bestD = Number.POSITIVE_INFINITY
    palette.forEach(([pr, pg, pb], i) => {
      const d = (pr - cr) ** 2 + (pg - cg) ** 2 + (pb - cb) ** 2
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    cache[bin] = best
    return best
  }
}

/** GIF's LZW: the image's palette indices as variable-width codes, packed least significant bit first. */
function lzw(indices: Uint8Array, minCodeSize: number): Uint8Array {
  const out: number[] = []
  let acc = 0
  let bits = 0
  const emit = (code: number, size: number): void => {
    acc |= code << bits
    bits += size
    while (bits >= 8) {
      out.push(acc & 0xff)
      acc >>>= 8
      bits -= 8
    }
  }
  const clear = 1 << minCodeSize
  const end = clear + 1
  // The dictionary maps (prefix code, next index) to a code; a generation stamp empties it without clearing memory.
  const keys = new Uint32Array(4096 * 256)
  const codes = new Uint16Array(4096 * 256)
  let generation = 1
  let size = minCodeSize + 1
  let next = end + 1
  emit(clear, size)
  if (indices.length === 0) {
    emit(end, size)
    if (bits > 0) out.push(acc & 0xff)
    return Uint8Array.from(out)
  }
  let prefix = indices[0] as number
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i] as number
    const key = (prefix << 8) | k
    if (keys[key] === generation) {
      prefix = codes[key] as number
      continue
    }
    emit(prefix, size)
    if (next === 4096) {
      emit(clear, size)
      generation++
      size = minCodeSize + 1
      next = end + 1
    } else {
      if (next >= 1 << size) size++
      keys[key] = generation
      codes[key] = next++
    }
    prefix = k
  }
  emit(prefix, size)
  emit(end, size)
  if (bits > 0) out.push(acc & 0xff)
  return Uint8Array.from(out)
}

/** Options of {@link encodeGif}. */
export interface GifOptions {
  /** Frames per second (default 10); GIF delays are whole hundredths of a second. */
  readonly fps?: number
  /** Times to play: 0 (the default) loops forever. */
  readonly loop?: number
  /** The palette (default {@link quantise} of every frame). */
  readonly palette?: Palette
}

/**
 * Encode frames as an animated GIF: one global palette (median cut over all frames, or the one given), each frame's
 * pixels mapped to their nearest palette colour and LZW-compressed, a frame delay from `fps`, and a loop count. Frames
 * smaller than the first are placed at its top left; transparent pixels take the nearest colour (GIF transparency is
 * not used).
 *
 * @param frames - The frames (the first sets the size).
 * @param options - Frames per second, loop count and palette.
 * @returns The GIF file's bytes.
 * @throws WhiteboxError with code `value` for no frames, and `shape` for a frame larger than the first.
 * @example
 * ```ts
 * import { encodeGif } from 'whitebox-render/animation'
 * const red = { width: 1, height: 1, rgba: new Uint8ClampedArray([255, 0, 0, 255]) }
 * const gif = encodeGif([red, red], { fps: 5 })
 * new TextDecoder().decode(gif.slice(0, 6)) // => 'GIF89a'
 * gif[gif.length - 1] // => 59
 * ```
 */
export function encodeGif(frames: readonly Pixels[], options: GifOptions = {}): Uint8Array {
  const first = frames[0]
  if (first === undefined) throw new WhiteboxError('value', 'encodeGif: need at least one frame')
  for (const f of frames)
    if (f.width > first.width || f.height > first.height)
      throw new WhiteboxError('shape', 'encodeGif: every frame must fit inside the first')
  const palette = options.palette ?? quantise(frames)
  const bitsPer = Math.max(1, Math.ceil(Math.log2(Math.max(2, palette.length))))
  const tableSize = 1 << bitsPer
  const nearest = indexer(palette)
  const delay = Math.max(2, Math.round(100 / (options.fps ?? 10)))
  const parts: Uint8Array[] = []
  const u16 = (v: number): number[] => [v & 0xff, (v >> 8) & 0xff]
  parts.push(new TextEncoder().encode('GIF89a'))
  parts.push(Uint8Array.from([...u16(first.width), ...u16(first.height), 0xf0 | (bitsPer - 1), 0, 0]))
  const table = new Uint8Array(3 * tableSize)
  palette.forEach(([r, g, b], i) => {
    table.set([r, g, b], 3 * i)
  })
  parts.push(table)
  if (frames.length > 1) {
    const loop = options.loop ?? 0
    parts.push(
      Uint8Array.from([0x21, 0xff, 0x0b, ...new TextEncoder().encode('NETSCAPE2.0'), 0x03, 0x01, ...u16(loop), 0x00]),
    )
  }
  const minCodeSize = Math.max(2, bitsPer)
  for (const f of frames) {
    parts.push(Uint8Array.from([0x21, 0xf9, 0x04, 0x04, ...u16(delay), 0x00, 0x00]))
    parts.push(Uint8Array.from([0x2c, 0, 0, 0, 0, ...u16(f.width), ...u16(f.height), 0x00]))
    const indices = new Uint8Array(f.width * f.height)
    const d = f.rgba
    for (let i = 0; i < indices.length; i++)
      indices[i] = nearest(d[4 * i] as number, d[4 * i + 1] as number, d[4 * i + 2] as number)
    const data = lzw(indices, minCodeSize)
    const blocks: number[] = [minCodeSize]
    for (let i = 0; i < data.length; i += 255) {
      const chunk = data.subarray(i, i + 255)
      blocks.push(chunk.length, ...chunk)
    }
    blocks.push(0)
    parts.push(Uint8Array.from(blocks))
  }
  parts.push(Uint8Array.from([0x3b]))
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0))
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return out
}
