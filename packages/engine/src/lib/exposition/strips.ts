/**
 * The exposition record of a position-aligned strip (`strip`): tokens (or positions) along x with per-position values
 * beneath them in tracks (probabilities, an attention row, gates, the states of a trellis), tags and marks, and a
 * chosen path through the tracks. Sequence models, transformers, codes and rankings are read this way.
 */

import { MdmlError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** One track: a value at each position, drawn as a row of cells beneath the tokens. */
export interface StripTrack {
  readonly name: string
  readonly values: readonly number[]
}

/** One row of tags: a short categorical label at each position (a part of speech, an entity, a code symbol). */
export interface StripTags {
  readonly name: string
  readonly labels: readonly string[]
}

/**
 * Positions with tokens, tracks of values, rows of tags and marks. `path` picks one track per position (a Viterbi path
 * through a trellis's states, a beam's choice) and is drawn as a line through the chosen cells; `marks` highlight
 * positions (the attended token, the one being predicted, an error).
 */
export interface Strip {
  readonly kind: 'strip'
  /** The token at each position. */
  readonly tokens: readonly string[]
  readonly tracks: readonly StripTrack[]
  readonly tags?: readonly StripTags[]
  /** Positions to highlight. */
  readonly marks?: readonly number[]
  /** The index of the chosen track at each position. */
  readonly path?: readonly number[]
  /** The values mapped to the colour ramp (default [0, 1] for values within it, else the data's span). */
  readonly range?: readonly [number, number]
  /** What the tracks hold (`probability`, `attention`), for the colour bar and the tooltips. */
  readonly valueName?: string
}

/**
 * A strip, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a track, a row of tags or the path does not have one entry per token; `value` when a
 *   mark or a path entry is not a position or a track.
 * @example
 * ```ts
 * import { stripRecord } from 'mdml-engine/exposition'
 * const r = stripRecord({
 *   tokens: ['the', 'cat', 'sat'],
 *   tracks: [{ name: 'noun', values: [0.1, 0.9, 0.2] }, { name: 'verb', values: [0.1, 0.05, 0.8] }],
 *   path: [0, 0, 1],
 *   marks: [2],
 * })
 * r.tracks.length // => 2
 * ```
 */
export function stripRecord(record: Omit<Strip, 'kind'>): Strip {
  const n = record.tokens.length
  for (const [k, t] of record.tracks.entries()) sameLength('strip', `tracks[${k}].values`, t.values.length, n)
  for (const [k, t] of (record.tags ?? []).entries()) sameLength('strip', `tags[${k}].labels`, t.labels.length, n)
  if (record.path !== undefined) {
    sameLength('strip', 'path', record.path.length, n)
    for (const k of record.path)
      if (!(Number.isInteger(k) && k >= 0 && k < record.tracks.length))
        throw new MdmlError('value', `strip: path holds ${k}, not a track`)
  }
  for (const m of record.marks ?? [])
    if (!(Number.isInteger(m) && m >= 0 && m < n))
      throw new MdmlError('value', `strip: marks holds ${m}, not a position`)
  return { kind: 'strip', ...record }
}
