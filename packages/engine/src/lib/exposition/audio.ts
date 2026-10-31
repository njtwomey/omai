/**
 * The exposition record of an audio clip (`audio`): mono samples with their rate, an optional spectrogram, markers, and
 * the map from tape steps to times in the clip, so that a view can lock a playhead to the session's step cursor (the
 * frame of an analysis, the window of a transform) and a step can be found from a time clicked in the picture.
 */

import { MdmlError } from '../base/index.ts'

/** A magnitude spectrogram: `magnitude[f][k]` at `frequencies[f]` Hz and `times[k]` seconds. */
export interface Spectrogram {
  readonly times: readonly number[]
  readonly frequencies: readonly number[]
  readonly magnitude: readonly (readonly number[])[]
}

/** The time in the clip a tape step stands at. */
export interface AudioFrame {
  readonly step: number
  /** Seconds from the start of the clip. */
  readonly time: number
}

/** A mono audio clip. */
export interface AudioClip {
  readonly kind: 'audio'
  /** Samples per second. */
  readonly sampleRate: number
  /** The samples, in [-1, 1]. */
  readonly samples: readonly number[]
  /** Its spectrogram (a view computes one from the samples when absent). */
  readonly spectrogram?: Spectrogram
  /** The time of each recorded step, increasing in both; between two frames the playhead moves linearly. */
  readonly frames?: readonly AudioFrame[]
  /** Labelled times (an onset, a detected pitch change). */
  readonly markers?: readonly { readonly time: number; readonly label: string }[]
  readonly caption?: string
}

/**
 * An audio clip, with the sample rate, the spectrogram's shape and the order of the frames checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `value` for a rate that is not positive or frames out of order; `shape` when the spectrogram's
 *   rows or columns do not match its axes.
 * @example
 * ```ts
 * import { audioClip } from 'mdml-engine/exposition'
 * audioClip({ sampleRate: 4, samples: [0, 1, 0, -1] }).samples.length // => 4
 * ```
 */
export function audioClip(record: Omit<AudioClip, 'kind'>): AudioClip {
  if (!(record.sampleRate > 0)) throw new MdmlError('value', 'audioClip: the sample rate must be positive')
  const s = record.spectrogram
  if (s !== undefined) {
    if (s.magnitude.length !== s.frequencies.length)
      throw new MdmlError(
        'shape',
        `audioClip: spectrogram has ${s.magnitude.length} rows, expected ${s.frequencies.length}`,
      )
    for (const [f, row] of s.magnitude.entries())
      if (row.length !== s.times.length)
        throw new MdmlError(
          'shape',
          `audioClip: spectrogram row ${f} has ${row.length} entries, expected ${s.times.length}`,
        )
  }
  const frames = record.frames ?? []
  for (let k = 1; k < frames.length; k++)
    if ((frames[k]?.step ?? 0) <= (frames[k - 1]?.step ?? 0) || (frames[k]?.time ?? 0) < (frames[k - 1]?.time ?? 0))
      throw new MdmlError('value', `audioClip: frames[${k}] is out of order`)
  return { kind: 'audio', ...record }
}

/**
 * The length of a clip in seconds.
 *
 * @param clip - The clip.
 * @returns The duration.
 * @example
 * ```ts
 * import { audioClip, audioDuration } from 'mdml-engine/exposition'
 * audioDuration(audioClip({ sampleRate: 8, samples: new Array(16).fill(0) })) // => 2
 * ```
 */
export function audioDuration(clip: AudioClip): number {
  return clip.samples.length / clip.sampleRate
}

/**
 * Where the playhead stands in the clip at a tape step: the time of the latest frame at or before it, moving linearly
 * toward the next frame's; clamped to the first and last frame. Without frames the step is the time in seconds.
 *
 * @param clip - The clip.
 * @param step - The tape step.
 * @returns Seconds from the start.
 * @example
 * ```ts
 * import { audioClip, playheadAt } from 'mdml-engine/exposition'
 * const c = audioClip({ sampleRate: 1, samples: [0, 0, 0, 0], frames: [{ step: 0, time: 0 }, { step: 10, time: 2 }] })
 * playheadAt(c, 5) // => 1
 * playheadAt(c, 99) // => 2
 * ```
 */
export function playheadAt(clip: AudioClip, step: number): number {
  const f = clip.frames
  if (f === undefined || f.length === 0) return Math.max(0, Math.min(step, audioDuration(clip)))
  const first = f[0] as AudioFrame
  const last = f[f.length - 1] as AudioFrame
  if (step <= first.step) return first.time
  if (step >= last.step) return last.time
  const j = f.findIndex((x) => x.step > step)
  const a = f[j - 1] as AudioFrame
  const b = f[j] as AudioFrame
  return a.time + ((step - a.step) / (b.step - a.step)) * (b.time - a.time)
}

/**
 * The tape step whose playhead is nearest a time in the clip (the inverse of {@link playheadAt}, rounded to a whole
 * step): what clicking the picture, or the audio playing, moves the step cursor to.
 *
 * @param clip - The clip.
 * @param time - Seconds from the start.
 * @returns The step.
 * @example
 * ```ts
 * import { audioClip, stepAt } from 'mdml-engine/exposition'
 * const c = audioClip({ sampleRate: 1, samples: [0, 0, 0, 0], frames: [{ step: 0, time: 0 }, { step: 10, time: 2 }] })
 * stepAt(c, 1.01) // => 5
 * stepAt(c, -3) // => 0
 * ```
 */
export function stepAt(clip: AudioClip, time: number): number {
  const f = clip.frames
  if (f === undefined || f.length === 0) return Math.round(Math.max(0, Math.min(time, audioDuration(clip))))
  const first = f[0] as AudioFrame
  const last = f[f.length - 1] as AudioFrame
  if (time <= first.time) return first.step
  if (time >= last.time) return last.step
  const j = f.findIndex((x) => x.time > time)
  const a = f[j - 1] as AudioFrame
  const b = f[j] as AudioFrame
  return Math.round(a.step + ((time - a.time) / (b.time - a.time)) * (b.step - a.step))
}
