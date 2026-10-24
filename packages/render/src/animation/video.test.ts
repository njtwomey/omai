import { describe, expect, it } from 'vitest'
import { canEncodeVideo, encodeWebm, muxWebm } from './video.ts'

interface El {
  readonly id: number
  readonly start: number
  readonly end: number
}

/** Read the EBML elements in `[from, to)` of a buffer. */
function elements(b: Uint8Array, from: number, to: number): El[] {
  const out: El[] = []
  let p = from
  while (p < to) {
    const first = b[p] ?? 0
    const idLen = Math.clz32(first) - 23
    let id = 0
    for (let k = 0; k < idLen; k++) id = id * 256 + (b[p + k] ?? 0)
    p += idLen
    const s0 = b[p] ?? 0
    const sizeLen = Math.clz32(s0) - 23
    let size = s0 & ((1 << (8 - sizeLen)) - 1)
    for (let k = 1; k < sizeLen; k++) size = size * 256 + (b[p + k] ?? 0)
    p += sizeLen
    out.push({ id, start: p, end: p + size })
    p += size
  }
  return out
}

describe('webm', () => {
  const chunk = (time: number, key: boolean, n: number) => ({ data: new Uint8Array(n).fill(time % 251), time, key })

  it('writes the EBML header, one track and a cluster per key frame with relative block times', () => {
    const webm = muxWebm([chunk(0, true, 300), chunk(100, false, 20), chunk(200, true, 40), chunk(300, false, 10)], {
      width: 320,
      height: 240,
    })
    const top = elements(webm, 0, webm.length)
    expect(top.map((e) => e.id)).toEqual([0x1a45dfa3, 0x18538067])
    const docType = elements(webm, top[0]?.start ?? 0, top[0]?.end ?? 0).find((e) => e.id === 0x4282)
    expect(new TextDecoder().decode(webm.subarray(docType?.start, docType?.end))).toBe('webm')
    const segment = elements(webm, top[1]?.start ?? 0, top[1]?.end ?? 0)
    expect(segment.map((e) => e.id)).toEqual([0x1549a966, 0x1654ae6b, 0x1f43b675, 0x1f43b675])
    const second = segment[3] as El
    const parts = elements(webm, second.start, second.end)
    expect(webm[parts[0]?.start ?? 0]).toBe(200)
    const blocks = parts.filter((e) => e.id === 0xa3)
    expect(blocks.map((e) => [webm[e.start + 2], webm[e.start + 3], e.end - e.start - 4])).toEqual([
      [0, 0x80, 40],
      [100, 0, 10],
    ])
    expect(() => muxWebm([], { width: 1, height: 1 })).toThrow()
  })

  it('says it cannot encode in Node, and refuses there', async () => {
    expect(canEncodeVideo()).toBe(false)
    await expect(encodeWebm([{ width: 2, height: 2, rgba: new Uint8ClampedArray(16) }])).rejects.toThrow(
      /neither WebCodecs nor MediaRecorder/,
    )
  })
})
