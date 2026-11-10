/**
 * Metrics for sequence labelling at the level of entities (seqeval, the conlleval script): the entity spans a tag
 * sequence in the BIO, IOB1, BIOES or BILOU scheme encodes, the violations of the scheme (an inside tag that continues
 * nothing), and entity-level precision, recall and F1 in which a predicted entity counts only if its type and both ends
 * match a true one, micro, macro and weighted over types, with the per-type scores; and the share of sequences labelled
 * entirely right.
 */

import { OmaiError } from '../base/index.ts'
import { defineMetric, type MetricFunction, metricResult } from './metric.ts'

/** The tagging scheme: `bio` reads B, I, E and S prefixes by conlleval's rules (IOB1 and IOB2 included); `bilou` reads L as E and U as S. */
export type TagScheme = 'bio' | 'bilou'

/** An entity in a sequence: its type and the half-open range of tokens. */
export interface EntitySpan {
  readonly type: string
  readonly start: number
  readonly end: number
}

const split = (tag: string, scheme: TagScheme): [prefix: string, type: string] => {
  if (tag === 'O') return ['O', '']
  const cut = tag.indexOf('-')
  let prefix = cut < 0 ? tag : tag.slice(0, cut)
  const type = cut < 0 ? '' : tag.slice(cut + 1)
  if (scheme === 'bilou') prefix = prefix === 'L' ? 'E' : prefix === 'U' ? 'S' : prefix
  return [prefix, type]
}

const chunkEnd = (prev: string, tag: string, prevType: string, type: string): boolean => {
  if (prev === 'E' || prev === 'S') return true
  if (prev === 'B' && (tag === 'B' || tag === 'S' || tag === 'O')) return true
  if (prev === 'I' && (tag === 'B' || tag === 'S' || tag === 'O')) return true
  return prev !== 'O' && prevType !== type
}

const chunkStart = (prev: string, tag: string, prevType: string, type: string): boolean => {
  if (tag === 'B' || tag === 'S') return true
  if (prev === 'E' && (tag === 'E' || tag === 'I')) return true
  if (prev === 'S' && (tag === 'E' || tag === 'I')) return true
  if (prev === 'O' && (tag === 'E' || tag === 'I')) return true
  return tag !== 'O' && prevType !== type
}

/**
 * The entities a tag sequence encodes, by conlleval's rules as seqeval implements them: an entity starts at a `B` or
 * `S` tag, at an `I` or `E` tag that follows `O` or an entity of another type (so IOB1 and a stray `I` start one), and
 * ends where the next tag starts a new one or is `O`.
 *
 * @param tags - The tags of one sequence, like `B-PER`, `I-PER`, `O`.
 * @param scheme - `bio` (default; also IOB1 and BIOES) or `bilou`.
 * @returns The entities in order.
 * @concept sequence-labelling-metrics
 * @example
 * ```ts
 * import { entitySpans } from 'omai-engine/metrics'
 * entitySpans(['B-PER', 'I-PER', 'O', 'B-LOC']) // => [{ type: 'PER', start: 0, end: 2 }, { type: 'LOC', start: 3, end: 4 }]
 * ```
 */
export function entitySpans(tags: readonly string[], scheme: TagScheme = 'bio'): EntitySpan[] {
  const out: EntitySpan[] = []
  let prevPrefix = 'O'
  let prevType = ''
  let begin = 0
  for (let i = 0; i <= tags.length; i++) {
    const [prefix, type] = i === tags.length ? (['O', ''] as [string, string]) : split(tags[i] as string, scheme)
    if (chunkEnd(prevPrefix, prefix, prevType, type) && prevPrefix !== 'O')
      out.push({ type: prevType, start: begin, end: i })
    if (chunkStart(prevPrefix, prefix, prevType, type)) begin = i
    prevPrefix = prefix
    prevType = type
  }
  return out
}

/**
 * The positions where a BIO sequence breaks the scheme: an `I-X` that does not continue a `B-X` or `I-X` just before it
 * (an entity that starts without a `B`, as a tagger may produce).
 *
 * @param tags - The tags of one sequence.
 * @returns The indices of the violating tags.
 * @concept sequence-labelling-metrics
 * @example
 * ```ts
 * import { bioViolations } from 'omai-engine/metrics'
 * bioViolations(['O', 'I-PER', 'B-PER', 'I-LOC']) // => [1, 3]
 * ```
 */
export function bioViolations(tags: readonly string[]): number[] {
  const out: number[] = []
  for (const [i, t] of tags.entries()) {
    if (!t.startsWith('I-')) continue
    const prev = tags[i - 1] ?? 'O'
    const type = t.slice(2)
    if (!((prev.startsWith('B-') || prev.startsWith('I-')) && prev.slice(2) === type)) out.push(i)
  }
  return out
}

/** Precision, recall, F1 and the number of true entities. */
export interface PrfScores {
  readonly precision: number
  readonly recall: number
  readonly f1: number
  readonly support: number
}

/** Entity-level scores: micro-averaged, the macro and weighted averages over types, and each type. */
export interface EntityScores extends PrfScores {
  readonly macro: PrfScores
  readonly weighted: PrfScores
  readonly perType: Readonly<Record<string, PrfScores>>
}

const prf = (tp: number, predicted: number, support: number): PrfScores => {
  const precision = predicted === 0 ? 0 : tp / predicted
  const recall = support === 0 ? 0 : tp / support
  return {
    precision,
    recall,
    f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
    support,
  }
}

/**
 * Entity-level precision, recall and F1 (seqeval's `classification_report`): an entity of the predicted sequence is
 * correct only if a true entity has the same type and the same first and last token. The headline scores are micro
 * averages over all entities; the macro average is the mean over types, the weighted one weights types by support.
 *
 * @param yTrue - The true tag sequences.
 * @param yPred - The predicted tag sequences, of the same lengths.
 * @param scheme - `bio` (default) or `bilou`.
 * @returns The scores.
 * @throws OmaiError with code `shape` for sequences of different counts or lengths.
 * @concept sequence-labelling-metrics
 * @example
 * ```ts
 * import { entityScores } from 'omai-engine/metrics'
 * const t = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
 * const p = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
 * entityScores(t, p).f1 // => 0.5
 * ```
 */
export function entityScores(
  yTrue: readonly (readonly string[])[],
  yPred: readonly (readonly string[])[],
  scheme: TagScheme = 'bio',
): EntityScores {
  if (yTrue.length !== yPred.length) throw new OmaiError('shape', 'entityScores: different numbers of sequences')
  const tp = new Map<string, number>()
  const predicted = new Map<string, number>()
  const support = new Map<string, number>()
  const bump = (m: Map<string, number>, k: string): void => {
    m.set(k, (m.get(k) ?? 0) + 1)
  }
  for (const [i, t] of yTrue.entries()) {
    const p = yPred[i] as readonly string[]
    if (t.length !== p.length) throw new OmaiError('shape', `entityScores: sequence ${i} has different lengths`)
    const truth = new Set(entitySpans(t, scheme).map((e) => `${e.type}|${e.start}|${e.end}`))
    for (const e of entitySpans(t, scheme)) bump(support, e.type)
    for (const e of entitySpans(p, scheme)) {
      bump(predicted, e.type)
      if (truth.has(`${e.type}|${e.start}|${e.end}`)) bump(tp, e.type)
    }
  }
  const types = [...new Set([...support.keys(), ...predicted.keys()])].sort()
  const perType = Object.fromEntries(
    types.map((k) => [k, prf(tp.get(k) ?? 0, predicted.get(k) ?? 0, support.get(k) ?? 0)]),
  )
  const sum = (m: Map<string, number>): number => [...m.values()].reduce((a, b) => a + b, 0)
  const micro = prf(sum(tp), sum(predicted), sum(support))
  const rows = Object.values(perType)
  const mean = (f: (r: PrfScores) => number, w: (r: PrfScores) => number): number => {
    const total = rows.reduce((a, r) => a + w(r), 0)
    return total === 0 ? 0 : rows.reduce((a, r) => a + f(r) * w(r), 0) / total
  }
  const average = (w: (r: PrfScores) => number): PrfScores => ({
    precision: mean((r) => r.precision, w),
    recall: mean((r) => r.recall, w),
    f1: mean((r) => r.f1, w),
    support: micro.support,
  })
  return { ...micro, macro: average(() => 1), weighted: average((r) => r.support), perType }
}

/**
 * Entity-level F1 as a metric (micro average), with the precision, recall and counts in the breakdown.
 *
 * @param yTrue - The true tag sequences.
 * @param yPred - The predicted tag sequences.
 * @param scheme - `bio` (default) or `bilou`.
 * @returns The metric result.
 * @concept sequence-labelling-metrics
 * @example
 * ```ts
 * import { entityF1 } from 'omai-engine/metrics'
 * entityF1([['B-PER', 'O']], [['B-PER', 'O']]).estimate.value // => 1
 * ```
 */
export const entityF1: MetricFunction<[readonly (readonly string[])[], readonly (readonly string[])[], TagScheme?]> =
  defineMetric(
    { name: 'entityF1', direction: 'higher', range: [0, 1], inputs: 'labels', category: 'sequence labelling' },
    (yTrue, yPred, scheme = 'bio') => {
      const s = entityScores(yTrue, yPred, scheme)
      return metricResult('entityF1', s.f1, s.support, { precision: s.precision, recall: s.recall, support: s.support })
    },
  )

/**
 * Sequence accuracy (sklearn_crfsuite.metrics.sequence_accuracy_score): the share of sequences whose every label is
 * right. Stricter than the token accuracy of flattened sequences, which `accuracyScore` computes.
 *
 * @param yTrue - The true label sequences.
 * @param yPred - The predicted label sequences, of the same lengths.
 * @returns The metric result: the share, with the number of correct sequences in the breakdown.
 * @throws OmaiError with code `shape` for sequences of different counts or lengths, `value` for no sequences.
 * @concept sequence-labelling-metrics
 * @index function Share of label sequences predicted entirely right (sklearn-crfsuite sequence_accuracy_score).
 * @example
 * ```ts
 * import { sequenceAccuracyScore } from 'omai-engine/metrics'
 * sequenceAccuracyScore([['A', 'B'], ['A', 'A']], [['A', 'B'], ['A', 'B']]).estimate.value // => 0.5
 * ```
 */
export const sequenceAccuracyScore: MetricFunction<
  [readonly (readonly (string | number)[])[], readonly (readonly (string | number)[])[]]
> = defineMetric(
  {
    name: 'sequenceAccuracyScore',
    direction: 'higher',
    range: [0, 1],
    inputs: 'labels',
    category: 'sequence labelling',
  },
  (yTrue, yPred) => {
    const name = 'sequenceAccuracyScore'
    if (yTrue.length !== yPred.length) throw new OmaiError('shape', `${name}: different numbers of sequences`)
    if (yTrue.length === 0) throw new OmaiError('value', `${name}: no sequences`)
    let correct = 0
    for (const [i, t] of yTrue.entries()) {
      const p = yPred[i] as readonly (string | number)[]
      if (t.length !== p.length) throw new OmaiError('shape', `${name}: sequence ${i} has different lengths`)
      if (t.every((label, j) => label === p[j])) correct++
    }
    return metricResult(name, correct / yTrue.length, yTrue.length, { correct })
  },
)
