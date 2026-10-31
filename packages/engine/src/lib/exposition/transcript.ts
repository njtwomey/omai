/**
 * The exposition record of a transcript (`transcript`): the tokens of a prompt and of what a model has emitted so far,
 * in turns by role (user, assistant, tool call, tool result), each token with a weight per named track (the attention
 * of a head, a saliency, a probability), and the index of the token the viewed step emitted. A session probes it at
 * each step, so the transcript grows token by token and the highlight moves with the cursor.
 */

import { MdmlError } from '../base/index.ts'

/** One token of a transcript. */
export interface TranscriptToken {
  /** The text of the token, with its own leading space when it has one. */
  readonly text: string
  /** Who said it (`user`, `assistant`, `tool`, `system`); consecutive tokens of one role form a turn. */
  readonly role?: string
}

/** A named weight per token (one head's attention to each token, say), not necessarily normalised. */
export interface TranscriptTrack {
  readonly name: string
  /** One weight per token, at least zero. */
  readonly weights: readonly number[]
}

/** The tokens of a conversation or a generation, with highlights. */
export interface Transcript {
  readonly kind: 'transcript'
  readonly tokens: readonly TranscriptToken[]
  /** The weights the highlight can show; the first is shown by default. */
  readonly tracks?: readonly TranscriptTrack[]
  /** The index of the token emitted at this step (marked in the text). */
  readonly emitted?: number
  /** What the weights are (`attention of the emitted token`), for the legend. */
  readonly weightName?: string
  readonly caption?: string
}

/**
 * A transcript, with the weights and the emitted index checked against the tokens.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a track does not have one weight per token or `emitted` is not a token; `value` when
 *   a weight is negative or not finite.
 * @example
 * ```ts
 * import { transcript } from 'mdml-engine/exposition'
 * const t = transcript({ tokens: [{ text: 'Hi' }, { text: ' there' }], tracks: [{ name: 'head', weights: [0.3, 0.7] }], emitted: 1 })
 * t.emitted // => 1
 * ```
 */
export function transcript(record: Omit<Transcript, 'kind'>): Transcript {
  const n = record.tokens.length
  for (const [k, t] of (record.tracks ?? []).entries()) {
    if (t.weights.length !== n)
      throw new MdmlError('shape', `transcript: tracks[${k}] has ${t.weights.length} weights, expected ${n}`)
    for (const [i, w] of t.weights.entries())
      if (!(Number.isFinite(w) && w >= 0))
        throw new MdmlError('value', `transcript: tracks[${k}].weights[${i}] is not a weight`)
  }
  if (record.emitted !== undefined && !(Number.isInteger(record.emitted) && record.emitted >= 0 && record.emitted < n))
    throw new MdmlError('shape', `transcript: emitted ${record.emitted} is not one of the ${n} tokens`)
  return { kind: 'transcript', ...record }
}

/**
 * The turns of a transcript: runs of consecutive tokens with one role, as `[first, end)` ranges of token indices.
 *
 * @param record - The record.
 * @returns The turns in order, each with its role (`undefined` for tokens without one).
 * @example
 * ```ts
 * import { transcript, turnsOf } from 'mdml-engine/exposition'
 * const t = transcript({ tokens: [{ text: 'a', role: 'user' }, { text: 'b', role: 'user' }, { text: 'c', role: 'assistant' }] })
 * turnsOf(t) // => [{ role: 'user', from: 0, to: 2 }, { role: 'assistant', from: 2, to: 3 }]
 * ```
 */
export function turnsOf(record: Transcript): { role: string | undefined; from: number; to: number }[] {
  const turns: { role: string | undefined; from: number; to: number }[] = []
  for (const [i, t] of record.tokens.entries()) {
    const last = turns[turns.length - 1]
    if (last !== undefined && last.role === t.role) last.to = i + 1
    else turns.push({ role: t.role, from: i, to: i + 1 })
  }
  return turns
}
