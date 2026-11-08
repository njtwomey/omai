/**
 * The exposition record of a position-aligned strip (`strip`): tokens (or positions) along x with per-position values
 * beneath them in tracks (probabilities, an attention row, gates, the states of a trellis), tags and marks, and a
 * chosen path through the tracks. Sequence models, transformers, codes and rankings are read this way, and so is the
 * run-length posterior of a changepoint detector over time (`runLengthStrip`).
 */

import { OmaiError } from '../base/index.ts'
import { sameLength, seal } from './_check.ts'

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
 * @throws OmaiError `shape` when a track, a row of tags or the path does not have one entry per token; `value` when a
 *   mark or a path entry is not a position or a track.
 * @example
 * ```ts
 * import { stripRecord } from 'omai-engine/exposition'
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
        throw new OmaiError('value', `strip: path holds ${k}, not a track`)
  }
  for (const m of record.marks ?? [])
    if (!(Number.isInteger(m) && m >= 0 && m < n))
      throw new OmaiError('value', `strip: marks holds ${m}, not a position`)
  return seal({ kind: 'strip', ...record })
}

/** What {@link runLengthStrip} reads: the run-length posterior of each step, and what to lay over it. */
export interface RunLengthStripInput {
  /** `runLength[t][r]` = p(r_t = r | y_1..y_t): the chance that the current segment holds r observations (entry 0 is 0). */
  readonly runLength: readonly ArrayLike<number>[]
  /** The most probable run length per step, drawn as the path through the tracks (default the arg-max of each row). */
  readonly mapRunLength?: readonly number[]
  /** The steps at which a segment truly starts, marked on the strip (when the data carries its truth). */
  readonly changepoints?: readonly number[]
  /** Show run lengths up to this one (default all of them); longer MAP run lengths are drawn on the last track. */
  readonly maxRunLength?: number
}

/**
 * The run-length posterior of a changepoint detector over time as a strip: one position per step, one track per run
 * length 1, 2, 3, ... holding the probability of that run length at that step, the most probable run length as the
 * path through the tracks (a segment that continues climbs a diagonal, a changepoint drops it back to the first
 * track) and the true changepoints, when known, as marks.
 *
 * @param input - The posterior rows, the MAP run lengths, the true changepoints and a cap on the run length shown.
 * @returns The strip.
 * @throws OmaiError `shape` for a MAP run length per step that does not match the rows; `value` for a changepoint that
 *   is not a step.
 * @example
 * ```ts
 * import { runLengthStrip } from 'omai-engine/exposition'
 * const s = runLengthStrip({ runLength: [[0, 1], [0, 0.1, 0.9], [0, 0.8, 0.1, 0.1]], changepoints: [2] })
 * [s.tracks.length, s.path, s.marks] // => [3, [0, 1, 0], [2]]
 * ```
 */
export function runLengthStrip(input: RunLengthStripInput): Strip {
  const steps = input.runLength.length
  const widest = input.runLength.reduce((m, row) => Math.max(m, row.length - 1), 0)
  const shown = Math.max(1, Math.min(input.maxRunLength ?? widest, widest))
  const map =
    input.mapRunLength ??
    input.runLength.map((row) => {
      let best = 1
      for (let r = 1; r < row.length; r++) if ((row[r] ?? 0) > (row[best] ?? 0)) best = r
      return best
    })
  sameLength('runLengthStrip', 'mapRunLength', map.length, steps)
  return stripRecord({
    tokens: input.runLength.map((_, t) => String(t)),
    tracks: Array.from({ length: shown }, (_, k) => ({
      name: `run length ${k + 1}`,
      values: input.runLength.map((row) => row[k + 1] ?? 0),
    })),
    path: map.map((r) => Math.min(Math.max(r, 1), shown) - 1),
    ...(input.changepoints === undefined ? {} : { marks: [...input.changepoints] }),
    range: [0, 1],
    valueName: 'probability',
  })
}
