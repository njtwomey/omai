import type { Pixels } from 'mdml-engine/scene'
import { describe, expect, it } from 'vitest'
import { encodeGif, quantise } from './gif.ts'

/** A minimal GIF decoder for the tests: the global palette and each frame's palette indices. */
function decodeGif(gif: Uint8Array): { width: number; height: number; palette: number[][]; frames: number[][] } {
  const u16 = (i: number): number => (gif[i] ?? 0) | ((gif[i + 1] ?? 0) << 8)
  const width = u16(6)
  const height = u16(8)
  const packed = gif[10] ?? 0
  const size = 1 << ((packed & 7) + 1)
  const palette = Array.from({ length: size }, (_, i) => [
    gif[13 + 3 * i] ?? 0,
    gif[14 + 3 * i] ?? 0,
    gif[15 + 3 * i] ?? 0,
  ])
  let p = 13 + 3 * size
  const frames: number[][] = []
  while (p < gif.length && gif[p] !== 0x3b) {
    if (gif[p] === 0x21) {
      p += 2
      while ((gif[p] ?? 0) !== 0) p += (gif[p] ?? 0) + 1
      p++
      continue
    }
    // Image descriptor, then the LZW data in sub-blocks.
    const w = u16(p + 5)
    const h = u16(p + 7)
    p += 10
    const min = gif[p++] ?? 2
    const data: number[] = []
    while ((gif[p] ?? 0) !== 0) {
      const n = gif[p] ?? 0
      data.push(...gif.subarray(p + 1, p + 1 + n))
      p += n + 1
    }
    p++
    const clear = 1 << min
    const end = clear + 1
    let codeSize = min + 1
    let dict: number[][] = []
    const reset = (): void => {
      dict = Array.from({ length: clear + 2 }, (_, i) => [i])
      codeSize = min + 1
    }
    reset()
    const out: number[] = []
    let bit = 0
    let prev: number[] | null = null
    for (;;) {
      let code = 0
      for (let k = 0; k < codeSize; k++, bit++) code |= (((data[bit >> 3] ?? 0) >> (bit & 7)) & 1) << k
      if (code === clear) {
        reset()
        prev = null
        continue
      }
      if (code === end) break
      let entry: number[]
      if (code < dict.length) entry = dict[code] as number[]
      else if (prev !== null) entry = [...prev, prev[0] as number]
      else throw new Error('bad code')
      out.push(...entry)
      if (prev !== null && dict.length < 4096) dict.push([...prev, entry[0] as number])
      if (dict.length === 1 << codeSize && codeSize < 12) codeSize++
      prev = entry
    }
    expect(out.length).toBe(w * h)
    frames.push(out)
  }
  return { width, height, palette, frames }
}

const image = (w: number, h: number, f: (x: number, y: number) => [number, number, number]): Pixels => {
  const rgba = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      rgba.set([...f(x, y), 255], 4 * (y * w + x))
    }
  return { width: w, height: h, rgba }
}

describe('gif', () => {
  it('finds the exact colours of an image with few of them', () => {
    const im = image(4, 4, (x) => (x < 2 ? [8, 8, 8] : [248, 0, 0]))
    expect([...quantise([im])].sort()).toEqual([
      [248, 0, 0],
      [8, 8, 8],
    ])
  })

  it('round-trips frames through LZW with a loop and the right size', () => {
    const colours: [number, number, number][] = [
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255],
      [250, 250, 250],
      [10, 10, 10],
    ]
    const frames = [0, 1, 2].map((t) => image(37, 23, (x, y) => colours[(x + 2 * y + t) % 5] ?? [0, 0, 0]))
    const gif = encodeGif(frames, { fps: 4 })
    const back = decodeGif(gif)
    expect([back.width, back.height, back.frames.length]).toEqual([37, 23, 3])
    // Every pixel maps back to (the palette's version of) its colour.
    back.frames.forEach((indices, t) => {
      indices.forEach((k, i) => {
        const want = colours[((i % 37) + 2 * Math.floor(i / 37) + t) % 5] ?? [0, 0, 0]
        const got = back.palette[k] ?? []
        for (let c = 0; c < 3; c++) expect(Math.abs((got[c] ?? 0) - (want[c] ?? 0))).toBeLessThanOrEqual(4)
      })
    })
    // The loop extension and a delay of 25 hundredths.
    const text = new TextDecoder('latin1').decode(gif)
    expect(text.includes('NETSCAPE2.0')).toBe(true)
    const gce = gif.indexOf(0xf9)
    expect((gif[gce + 3] ?? 0) | ((gif[gce + 4] ?? 0) << 8)).toBe(25)
  })

  it('handles big frames that overflow the 4096-code dictionary', () => {
    const big = image(160, 120, (x, y) => [(x * 13 + y * 7) % 256, (x * y) % 256, (x ^ y) % 256])
    const back = decodeGif(encodeGif([big]))
    expect(back.frames[0]?.length).toBe(160 * 120)
    expect(back.palette.length).toBe(256)
  })

  it('refuses no frames and frames larger than the first', () => {
    expect(() => encodeGif([])).toThrow(/at least one/)
    expect(() => encodeGif([image(2, 2, () => [0, 0, 0]), image(3, 2, () => [0, 0, 0])])).toThrow(/fit inside/)
  })
})
