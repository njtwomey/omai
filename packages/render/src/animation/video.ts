/**
 * WebM video export: `muxWebm`, a pure muxer that packs encoded VP8 or VP9 frames into a WebM file (EBML header, track,
 * clusters of simple blocks), and `encodeWebm`, which encodes frames in the browser with WebCodecs or, failing that,
 * records a canvas with a `MediaRecorder`; `canEncodeVideo` says whether either is available.
 */

import { OmaiError } from 'omai-engine/base'
import type { Pixels } from 'omai-engine/scene'

/** One encoded video frame: its bytes, its time in milliseconds and whether it is a key frame. */
export interface VideoChunk {
  readonly data: Uint8Array
  readonly time: number
  readonly key: boolean
}

/** Options of {@link muxWebm}. */
export interface WebmOptions {
  readonly width: number
  readonly height: number
  /** The codec of the chunks (default `V_VP8`). */
  readonly codec?: 'V_VP8' | 'V_VP9'
  /** The duration in milliseconds (default the last chunk's time plus the mean frame interval). */
  readonly duration?: number
}

function concat(parts: readonly Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0))
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return out
}

/** An EBML element: its id (with its length marker, as written), its size as a variable-length integer, its payload. */
function element(id: number, ...payload: readonly Uint8Array[]): Uint8Array {
  const body = concat(payload)
  const idBytes: number[] = []
  for (let v = id; v > 0; v = Math.floor(v / 256)) idBytes.unshift(v & 0xff)
  // The size as a variable-length integer: the shortest of 1 to 8 bytes (all-ones values are reserved).
  const n = body.length
  let len = 1
  while (len < 8 && n >= 2 ** (7 * len) - 1) len++
  const size: number[] = []
  for (let k = 0, v = n; k < len; k++, v = Math.floor(v / 256)) size.unshift(v & 0xff)
  size[0] = (size[0] ?? 0) | (1 << (8 - len))
  return concat([Uint8Array.from(idBytes), Uint8Array.from(size), body])
}

const uint = (v: number): Uint8Array => {
  const out: number[] = []
  for (let x = Math.max(0, Math.floor(v)); x > 0 || out.length === 0; x = Math.floor(x / 256)) out.unshift(x & 0xff)
  return Uint8Array.from(out)
}
const float = (v: number): Uint8Array => {
  const b = new Uint8Array(8)
  new DataView(b.buffer).setFloat64(0, v)
  return b
}
const str = (s: string): Uint8Array => new TextEncoder().encode(s)

/**
 * Mux encoded VP8 or VP9 frames into a WebM file: the EBML header, segment information (millisecond timecodes and the
 * duration), one video track, and clusters of simple blocks, a new cluster at each key frame (and at least every 30
 * seconds, as block times are 16-bit offsets). Pure: the chunks come from WebCodecs' `VideoEncoder` in the browser, or
 * from anywhere else.
 *
 * @param chunks - The encoded frames in order; the first should be a key frame.
 * @param options - Frame size, codec and duration.
 * @returns The file's bytes.
 * @throws OmaiError with code `value` for no chunks or a non-positive size.
 * @example
 * ```ts
 * import { muxWebm } from 'omai-render/animation'
 * const webm = muxWebm([{ data: new Uint8Array([1, 2, 3]), time: 0, key: true }], { width: 2, height: 2 })
 * [...webm.slice(0, 4)] // => [26, 69, 223, 163]
 * ```
 */
export function muxWebm(chunks: readonly VideoChunk[], options: WebmOptions): Uint8Array {
  if (chunks.length === 0) throw new OmaiError('value', 'muxWebm: need at least one frame')
  if (!(options.width > 0 && options.height > 0)) throw new OmaiError('value', 'muxWebm: the size must be positive')
  const last = chunks[chunks.length - 1] as VideoChunk
  const step = chunks.length > 1 ? last.time / (chunks.length - 1) : 100
  const duration = options.duration ?? last.time + step
  const header = element(
    0x1a45dfa3,
    element(0x4286, uint(1)),
    element(0x42f7, uint(1)),
    element(0x42f2, uint(4)),
    element(0x42f3, uint(8)),
    element(0x4282, str('webm')),
    element(0x4287, uint(2)),
    element(0x4285, uint(2)),
  )
  const info = element(
    0x1549a966,
    element(0x2ad7b1, uint(1_000_000)),
    element(0x4d80, str('omai-render')),
    element(0x5741, str('omai-render')),
    element(0x4489, float(duration)),
  )
  const tracks = element(
    0x1654ae6b,
    element(
      0xae,
      element(0xd7, uint(1)),
      element(0x73c5, uint(1)),
      element(0x83, uint(1)),
      element(0x86, str(options.codec ?? 'V_VP8')),
      element(0xe0, element(0xb0, uint(options.width)), element(0xba, uint(options.height))),
    ),
  )
  const clusters: Uint8Array[] = []
  let start = 0
  let blocks: Uint8Array[] = []
  const flush = (): void => {
    if (blocks.length > 0) clusters.push(element(0x1f43b675, element(0xe7, uint(start)), ...blocks))
    blocks = []
  }
  chunks.forEach((c, i) => {
    const t = Math.round(c.time)
    if (i === 0 || c.key || t - start > 30_000) {
      flush()
      start = t
    }
    const rel = t - start
    blocks.push(element(0xa3, Uint8Array.from([0x81, (rel >> 8) & 0xff, rel & 0xff, c.key ? 0x80 : 0x00]), c.data))
  })
  flush()
  return concat([header, element(0x18538067, info, tracks, ...clusters)])
}

/** Options of {@link encodeWebm}. */
export interface EncodeVideoOptions {
  /** Frames per second (default 10). */
  readonly fps?: number
  /** Bits per second (default 2.5 million). */
  readonly bitrate?: number
  /** A key frame every this many frames (default 30). */
  readonly keyInterval?: number
}

/**
 * Whether this platform can encode video for {@link encodeWebm}: WebCodecs, or failing that, a `MediaRecorder` that
 * writes WebM (never in Node).
 *
 * @returns Whether `encodeWebm` can run.
 * @index function Whether this browser can encode video: `canEncodeVideo()`.
 */
export function canEncodeVideo(): boolean {
  return (
    typeof VideoEncoder !== 'undefined' ||
    (typeof MediaRecorder !== 'undefined' &&
      typeof document !== 'undefined' &&
      MediaRecorder.isTypeSupported('video/webm'))
  )
}

async function withWebCodecs(frames: readonly Pixels[], options: EncodeVideoOptions): Promise<Uint8Array> {
  const first = frames[0] as Pixels
  // VP8 wants even sizes: the frame is cropped by at most a pixel.
  const width = first.width - (first.width % 2)
  const height = first.height - (first.height % 2)
  const fps = options.fps ?? 10
  const config: VideoEncoderConfig = {
    codec: 'vp8',
    width,
    height,
    bitrate: options.bitrate ?? 2_500_000,
    framerate: fps,
  }
  const support = await VideoEncoder.isConfigSupported(config)
  if (support.supported !== true) throw new OmaiError('not-implemented', 'encodeWebm: VP8 is not supported here')
  const chunks: VideoChunk[] = []
  let failure: unknown
  const encoder = new VideoEncoder({
    output: (chunk) => {
      const data = new Uint8Array(chunk.byteLength)
      chunk.copyTo(data)
      chunks.push({ data, time: chunk.timestamp / 1000, key: chunk.type === 'key' })
    },
    error: (e) => {
      failure = e
    },
  })
  encoder.configure(config)
  const every = options.keyInterval ?? 30
  frames.forEach((f, i) => {
    const rgba = new Uint8Array(width * height * 4)
    for (let y = 0; y < height; y++)
      rgba.set(f.rgba.subarray(y * f.width * 4, y * f.width * 4 + width * 4), y * width * 4)
    const frame = new VideoFrame(rgba, {
      format: 'RGBA',
      codedWidth: width,
      codedHeight: height,
      timestamp: Math.round((i * 1e6) / fps),
      duration: Math.round(1e6 / fps),
    })
    encoder.encode(frame, { keyFrame: i % every === 0 })
    frame.close()
  })
  await encoder.flush()
  encoder.close()
  if (failure !== undefined) throw failure
  return muxWebm(chunks, { width, height, duration: (frames.length * 1000) / fps })
}

async function withRecorder(frames: readonly Pixels[], options: EncodeVideoOptions): Promise<Uint8Array> {
  const first = frames[0] as Pixels
  const canvas = document.createElement('canvas')
  canvas.width = first.width
  canvas.height = first.height
  const ctx = canvas.getContext('2d')
  if (ctx === null) throw new OmaiError('not-implemented', 'encodeWebm: no 2-D canvas')
  const fps = options.fps ?? 10
  const stream = canvas.captureStream(fps)
  const recorder = new MediaRecorder(stream, {
    mimeType: 'video/webm',
    videoBitsPerSecond: options.bitrate ?? 2_500_000,
  })
  const parts: Blob[] = []
  recorder.ondataavailable = (e) => parts.push(e.data)
  const stopped = new Promise<void>((resolve) => {
    recorder.onstop = () => resolve()
  })
  const draw = (f: Pixels): void => {
    ctx.putImageData(new ImageData(new Uint8ClampedArray(f.rgba), f.width, f.height), 0, 0)
  }
  draw(first)
  recorder.start()
  // A recorder records in real time: each frame is held for its interval.
  for (const f of frames) {
    draw(f)
    await new Promise((resolve) => setTimeout(resolve, 1000 / fps))
  }
  recorder.stop()
  await stopped
  return new Uint8Array(await new Blob(parts).arrayBuffer())
}

/**
 * Encode frames as a WebM video in the browser: with WebCodecs (`VideoEncoder`, VP8) as fast as the encoder runs,
 * muxed by {@link muxWebm}; without it, by drawing the frames to a canvas recorded by a `MediaRecorder`, in real time.
 *
 * @param frames - The frames (the first sets the size; VP8 crops odd sizes by a pixel).
 * @param options - Frames per second, bitrate and key-frame interval.
 * @returns A promise of the file's bytes.
 * @throws OmaiError with code `value` for no frames, and `not-implemented` where neither encoder exists (see
 * {@link canEncodeVideo}).
 *
 * @index function Pixel frames to WebM bytes in a browser: `await encodeWebm(frames, { fps: 30 })`.
 */
export async function encodeWebm(frames: readonly Pixels[], options: EncodeVideoOptions = {}): Promise<Uint8Array> {
  if (frames.length === 0) throw new OmaiError('value', 'encodeWebm: need at least one frame')
  if (typeof VideoEncoder !== 'undefined') {
    try {
      return await withWebCodecs(frames, options)
    } catch (e) {
      if (!(typeof MediaRecorder !== 'undefined' && typeof document !== 'undefined')) throw e
    }
  }
  if (typeof MediaRecorder !== 'undefined' && typeof document !== 'undefined') return withRecorder(frames, options)
  throw new OmaiError('not-implemented', 'encodeWebm: this platform has neither WebCodecs nor MediaRecorder')
}
