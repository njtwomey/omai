/**
 * The headless view of a `Transcript` record: the tokens as spans in turns, each with its weight on a chosen track
 * normalised to a highlight level, the emitted token flagged, and the heaviest tokens listed in words (so the
 * highlight is never the only way to read it). The React view draws these spans as text.
 */

import { maxOf } from 'mdml-engine/base'
import { type Transcript, turnsOf } from 'mdml-engine/exposition'
import { type Scene, scene, text } from 'mdml-engine/scene'

/** One token as drawn. */
export interface TranscriptSpan {
  readonly text: string
  /** The raw weight on the track (0 without one). */
  readonly weight: number
  /** The weight as a share of the track's largest, in [0, 1]: the strength of the highlight. */
  readonly level: number
  /** Whether this is the token the viewed step emitted. */
  readonly emitted: boolean
  /** What a tooltip says: the token, its weight and whether it was emitted. */
  readonly title: string
}

/** A run of spans of one role. */
export interface TranscriptTurn {
  readonly role: string | undefined
  readonly spans: readonly TranscriptSpan[]
}

/**
 * The turns of a transcript with the highlight of one track.
 *
 * @param record - The record.
 * @param track - The track to show: an index into the record's tracks (default the first; none highlights nothing).
 * @returns The turns, each a run of spans.
 * @example
 * ```ts
 * import { transcript } from 'mdml-engine/exposition'
 * import { transcriptTurns } from 'mdml-render/views'
 * const r = transcript({ tokens: [{ text: 'a', role: 'user' }, { text: 'b' }], tracks: [{ name: 'h', weights: [1, 3] }], emitted: 1 })
 * transcriptTurns(r)[1]?.spans[0]?.level // => 1
 * transcriptTurns(r)[0]?.spans[0]?.level // => 0.3333333333333333
 * ```
 */
export function transcriptTurns(record: Transcript, track = 0): TranscriptTurn[] {
  const weights = record.tracks?.[track]?.weights
  const top = Math.max(0, maxOf(weights ?? []))
  return turnsOf(record).map((turn) => ({
    role: turn.role,
    spans: record.tokens.slice(turn.from, turn.to).map((token, k): TranscriptSpan => {
      const i = turn.from + k
      const weight = weights?.[i] ?? 0
      const emitted = record.emitted === i
      return {
        text: token.text,
        weight,
        level: top > 0 ? weight / top : 0,
        emitted,
        title: `${JSON.stringify(token.text.trim() === '' ? token.text : token.text.trim())}${weights === undefined ? '' : `, ${record.tracks?.[track]?.name ?? 'weight'} ${Number(weight.toPrecision(3))}`}${emitted ? ', emitted at this step' : ''}`,
      }
    }),
  }))
}

/**
 * The tokens with the most weight on a track, heaviest first, as plain words.
 *
 * @param record - The record.
 * @param track - The track (default the first).
 * @param count - How many (default 3).
 * @returns The tokens (trimmed), their index and weight; none when the track has no weight.
 * @example
 * ```ts
 * import { transcript } from 'mdml-engine/exposition'
 * import { heaviestTokens } from 'mdml-render/views'
 * const r = transcript({ tokens: [{ text: 'a' }, { text: ' b' }, { text: ' c' }], tracks: [{ name: 'h', weights: [1, 3, 2] }] })
 * heaviestTokens(r, 0, 2).map((t) => t.text) // => ['b', 'c']
 * ```
 */
export function heaviestTokens(
  record: Transcript,
  track = 0,
  count = 3,
): { text: string; index: number; weight: number }[] {
  const weights = record.tracks?.[track]?.weights ?? []
  return weights
    .map((weight, index) => ({ text: (record.tokens[index]?.text ?? '').trim(), index, weight }))
    .filter((t) => t.weight > 0)
    .sort((a, b) => b.weight - a.weight || a.index - b.index)
    .slice(0, count)
}

/**
 * A transcript as a still scene (for slides and export): the turns as lines of text, the emitted token marked with an
 * arrow, wrapped at about sixty characters.
 *
 * @param record - The record.
 * @param track - The track whose heaviest token is named in the caption (default the first).
 * @returns The scene.
 * @example
 * ```ts
 * import { transcript } from 'mdml-engine/exposition'
 * import { transcriptScene } from 'mdml-render/views'
 * transcriptScene(transcript({ tokens: [{ text: 'Hi', role: 'user' }] })).kind // => 'scene'
 * ```
 */
export function transcriptScene(record: Transcript, track = 0): Scene {
  const lines: string[] = []
  for (const turn of transcriptTurns(record, track)) {
    const words = turn.spans.map((s) => `${s.emitted ? '▸' : ''}${s.text}`).join('')
    const prefix = turn.role === undefined ? '' : `${turn.role}: `
    let line = prefix
    for (const word of words.split(/(?<= )/)) {
      if (line.length + word.length > 60 && line.trim() !== prefix.trim()) {
        lines.push(line.trimEnd())
        line = ' '.repeat(prefix.length)
      }
      line += word
    }
    lines.push(line.trimEnd())
  }
  const height = 16 * lines.length + 16
  return scene(
    520,
    Math.max(height, 40),
    lines.map((l, i) => text(8, 16 + 16 * i, l, { size: 12, fill: 'fg' })),
    { title: `Transcript of ${record.tokens.length} tokens` },
  )
}
