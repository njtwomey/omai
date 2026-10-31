/**
 * Sequential covering, the oldest way to learn rules: find one good rule, remove the rows it covers, repeat. CN2 (Clark
 * and Niblett 1989) searches each rule by a beam over conjunctions of literals, scores it with entropy or Laplace
 * accuracy and keeps it only if a likelihood-ratio test calls it significant; ordered CN2 learns a decision list, each
 * rule deciding for the majority class of the rows still in play, and unordered CN2 (Clark and Boswell 1991) learns the
 * rules of one class at a time over all rows and lets the rules that fire vote. PRISM (Cendrowska 1987) grows each rule
 * greedily on precision until it is pure. The covering loop is a traced algorithm: one rule per step.
 */

import { type Json, MdmlError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import {
  conflicts,
  conjunctionMask,
  countClasses,
  dropRows,
  keepRows,
  type LiteralPool,
  literalPool,
  type RuleData,
  ruleData,
} from './_table.ts'
import {
  argmaxOf,
  coverageOf,
  type Literal,
  literalText,
  predictCounts,
  type Rule,
  type RuleHeuristic,
  type RuleModel,
  ruleQuality,
  ruleSignificance,
} from './rule.ts'

/** Which covering learner. */
export type CoveringMode = 'ordered' | 'unordered' | 'prism'

/** Options of the covering learners. */
export interface CoveringOptions {
  /** `ordered` CN2 (a decision list), `unordered` CN2 (voting rules per class) or `prism` (default `ordered`). */
  readonly mode?: CoveringMode
  /** The rule quality the beam maximises (default `entropy` for ordered, `laplace` for unordered; ignored by PRISM). */
  readonly heuristic?: RuleHeuristic
  /** The beam width of the rule search (default 5; 1 is greedy). */
  readonly beamWidth?: number
  /** The most literals in a rule (default 6). */
  readonly maxLength?: number
  /** A rule is kept only if its likelihood-ratio p-value is at most this (default 0.05; 1 keeps every rule). */
  readonly alpha?: number
  /** The fewest rows a rule must cover (default 2). */
  readonly minCovered?: number
  /** The most rules (default 100). */
  readonly maxRules?: number
  /** The m of the m-estimate heuristic (default 2). */
  readonly m?: number
  /** The most thresholds per numeric feature (default 24). */
  readonly maxThresholds?: number
  /** Indices of features whose values are category codes, tested by equality (default none). */
  readonly nominal?: readonly number[]
}

/** The state of the covering loop. */
export type CoveringState = {
  readonly rules: readonly Rule[]
  /** The rows still in play. */
  readonly remaining: readonly number[]
  /** The class index being covered (unordered and PRISM; 0 for ordered). */
  readonly target: number
  readonly done: boolean
  /** The last rule found, with the number of rows in play after each of its literals. */
  readonly last: { readonly literals: readonly Literal[]; readonly trail: readonly number[] } | null
}

const COVERING_SCHEMA: TraceSchema = channels({
  metrics: {
    rules: { description: 'rules learned so far' },
    remaining: { description: 'rows still in play' },
    coverage: { description: 'rows the last rule covers' },
    precision: { description: 'the last rule’s share of rows of its class' },
    accuracy: { description: 'training accuracy of the rules so far', direction: 'higher' },
    covered: { description: 'training rows covered by at least one rule' },
  },
  messages: {
    rule: {
      description: 'a rule found',
      payload: { literals: 'the literals in the order added', trail: 'rows covered after each' },
    },
  },
})

type Found = {
  readonly lits: readonly number[]
  readonly rows: Int32Array
  readonly counts: number[]
  readonly label: number
}

/** CN2's beam search for the best significant rule over the rows in play. */
function beamRule(
  rd: RuleData,
  pool: LiteralPool,
  rows: Int32Array,
  target: number | null,
  o: Required<Pick<CoveringOptions, 'heuristic' | 'beamWidth' | 'maxLength' | 'alpha' | 'minCovered' | 'm'>>,
): Found | null {
  const prior = countClasses(rd, rows)
  const total = rows.length
  let star: { lits: readonly number[]; rows: Int32Array }[] = [{ lits: [], rows }]
  let best: (Found & { quality: number }) | null = null
  const seen = new Set<string>()
  for (let depth = 0; depth < o.maxLength && star.length > 0; depth++) {
    const next: { lits: readonly number[]; rows: Int32Array; quality: number }[] = []
    for (const e of star) {
      for (let l = 0; l < pool.literals.length; l++) {
        if (conflicts(pool, e.lits, l)) continue
        const kept = keepRows(pool, e.rows, l)
        if (kept.length < o.minCovered || kept.length === e.rows.length) continue
        const lits = [...e.lits, l]
        const id = [...lits].sort((a, b) => a - b).join(',')
        if (seen.has(id)) continue
        seen.add(id)
        const counts = countClasses(rd, kept)
        const label = target ?? argmaxOf(counts)
        const quality = ruleQuality(o.heuristic, counts, label, prior, o.m)
        next.push({ lits, rows: kept, quality })
        const better = (counts[label] ?? 0) / kept.length > (prior[label] ?? 0) / Math.max(1, total)
        if (
          better &&
          ruleSignificance(counts, prior) <= o.alpha &&
          (best === null || quality > best.quality || (quality === best.quality && kept.length > best.rows.length))
        ) {
          best = { lits, rows: kept, counts, label, quality }
        }
      }
    }
    next.sort((a, b) => b.quality - a.quality || b.rows.length - a.rows.length)
    star = next.slice(0, o.beamWidth)
  }
  return best
}

/** PRISM's greedy rule: add the literal of the highest precision for the target class until the rule is pure. */
function prismRule(
  rd: RuleData,
  pool: LiteralPool,
  rows: Int32Array,
  target: number,
  maxLength: number,
  minCovered: number,
): Found | null {
  let lits: number[] = []
  let cur = rows
  let counts = countClasses(rd, cur)
  while ((counts[target] ?? 0) < cur.length && lits.length < maxLength) {
    let bestL = -1
    let bestP = -1
    let bestPrecision = -1
    let bestRows = cur
    for (let l = 0; l < pool.literals.length; l++) {
      if (conflicts(pool, lits, l)) continue
      const kept = keepRows(pool, cur, l)
      if (kept.length === 0 || kept.length === cur.length) continue
      const p = countClasses(rd, kept)[target] ?? 0
      if (p === 0) continue
      const precision = p / kept.length
      if (precision > bestPrecision || (precision === bestPrecision && p > bestP)) {
        bestL = l
        bestP = p
        bestPrecision = precision
        bestRows = kept
      }
    }
    if (bestL < 0) break
    lits = [...lits, bestL]
    cur = bestRows
    counts = countClasses(rd, cur)
  }
  if (lits.length === 0 || (counts[target] ?? 0) === 0 || cur.length < minCovered) return null
  return { lits, rows: cur, counts, label: target }
}

function modelOf(rd: RuleData, s: CoveringState, mode: CoveringMode): RuleModel {
  const all = countClasses(
    rd,
    Int32Array.from({ length: rd.n }, (_, i) => i),
  )
  const rest = countClasses(rd, s.remaining)
  return {
    kind: mode === 'ordered' ? 'list' : 'set',
    classes: rd.classes,
    nFeatures: rd.d,
    rules: [...s.rules],
    defaultCounts: mode === 'ordered' && s.remaining.length > 0 ? rest : all,
  }
}

/**
 * The model a covering state defines: the rules so far as a decision list (ordered CN2) or a voting set, with the
 * default rule the majority of the rows left in play (a list) or of the whole training set (a set).
 *
 * @param data - The training data.
 * @param s - The state.
 * @param options - The learner (`mode`) and `nominal` features the state was learned with.
 * @returns The model.
 * @example
 * ```ts
 * import { coveringAlgorithm, coveringModel } from 'mdml-engine/rules'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = { x: [[0], [0], [1], [1]], y: [0, 0, 1, 1] }
 * const { state } = runAlgorithm(coveringAlgorithm(data, { alpha: 1, minCovered: 1 }), { maxSteps: 10 })
 * coveringModel(data, state, {}).rules.length // => 1
 * ```
 */
export function coveringModel(data: LearnData, s: CoveringState, options: CoveringOptions = {}): RuleModel {
  return modelOf(ruleData(data, options.nominal ?? [], 'covering'), s, options.mode ?? 'ordered')
}

/**
 * Sequential covering (CN2 ordered or unordered, or PRISM) as a traced algorithm. Each step finds one rule: for
 * ordered CN2 the significant conjunction of the best quality over the rows still in play, which then decides for the
 * majority class of the rows it covers and removes them; for unordered CN2 and PRISM the best rule for the class being
 * covered, removing only the rows of that class it covers (the next class starts again from all rows). The run ends when
 * no significant rule is left.
 *
 * @param data - Features and class labels.
 * @param options - The learner, heuristic, beam width, rule length, significance level and limits.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a beam width, length or alpha out of range.
 * @example
 * ```ts
 * import { coveringAlgorithm } from 'mdml-engine/rules'
 * coveringAlgorithm({ x: [[0], [1]], y: [0, 1] }, { mode: 'prism' }).name // => 'prism'
 * ```
 */
export function coveringAlgorithm(data: LearnData, options: CoveringOptions = {}): Algorithm<CoveringState> {
  const rd = ruleData(data, options.nominal ?? [], 'covering')
  const mode = options.mode ?? 'ordered'
  const o = {
    heuristic: options.heuristic ?? (mode === 'ordered' ? 'entropy' : 'laplace'),
    beamWidth: options.beamWidth ?? 5,
    maxLength: options.maxLength ?? 6,
    alpha: options.alpha ?? 0.05,
    minCovered: options.minCovered ?? 2,
    m: options.m ?? 2,
  } as const
  const maxRules = options.maxRules ?? 100
  if (!(Number.isInteger(o.beamWidth) && o.beamWidth >= 1))
    throw new MdmlError('value', 'covering: beamWidth must be a positive integer')
  if (!(Number.isInteger(o.maxLength) && o.maxLength >= 1))
    throw new MdmlError('value', 'covering: maxLength must be a positive integer')
  if (!(o.alpha > 0 && o.alpha <= 1)) throw new MdmlError('value', 'covering: alpha must be in (0, 1]')
  const pool = literalPool(rd, options.maxThresholds ?? 24)
  const everything = Int32Array.from({ length: rd.n }, (_, i) => i)
  return {
    name: mode === 'prism' ? 'prism' : mode === 'ordered' ? 'cn2' : 'cn2-unordered',
    config: { mode, ...o, maxRules },
    schema: COVERING_SCHEMA,
    init: () => ({ rules: [], remaining: Array.from(everything), target: 0, done: rd.n === 0, last: null }),
    step: (s) => {
      let target = s.target
      let remaining: Int32Array = Int32Array.from(s.remaining)
      for (;;) {
        if (s.rules.length >= maxRules) return { ...s, done: true }
        if (mode === 'ordered') {
          const counts = countClasses(rd, remaining)
          if (remaining.length === 0 || counts.filter((c) => c > 0).length <= 1) return { ...s, done: true }
        } else {
          if (target >= rd.k) return { ...s, target, done: true }
          const hasPositive = remaining.some((r) => rd.y[r] === target)
          if (!hasPositive) {
            target++
            remaining = everything
            continue
          }
        }
        const found =
          mode === 'prism'
            ? prismRule(rd, pool, remaining, target, o.maxLength, 1)
            : beamRule(rd, pool, remaining, mode === 'ordered' ? null : target, o)
        if (found === null) {
          if (mode === 'ordered') return { ...s, done: true }
          target++
          remaining = everything
          continue
        }
        const mask = conjunctionMask(rd, pool, found.lits)
        const full = new Int32Array(rd.n)
        let m = 0
        for (let i = 0; i < rd.n; i++) if (mask[i] === 1) full[m++] = i
        const counts = mode === 'ordered' ? found.counts : countClasses(rd, full.subarray(0, m))
        const rule: Rule = { literals: found.lits.map((l) => pool.literals[l] as Literal), label: found.label, counts }
        let rows = remaining
        const trail: number[] = []
        for (const l of found.lits) {
          rows = keepRows(pool, rows, l)
          trail.push(rows.length)
        }
        const next =
          mode === 'ordered'
            ? dropRows(remaining, mask)
            : Int32Array.from(Array.from(remaining).filter((r) => !(mask[r] === 1 && rd.y[r] === target)))
        return {
          rules: [...s.rules, rule],
          remaining: Array.from(next),
          target,
          done: false,
          last: { literals: rule.literals, trail },
        }
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => {
      const model = modelOf(rd, s, mode)
      const counts = predictCounts(model, rd.x)
      let right = 0
      const covered = new Uint8Array(rd.n)
      for (const r of s.rules) {
        const m = coverageOf(r.literals, rd.x, rd.d)
        for (let i = 0; i < rd.n; i++) if (m[i] === 1) covered[i] = 1
      }
      for (let i = 0; i < rd.n; i++) if (argmaxOf(counts.subarray(i * rd.k, (i + 1) * rd.k)) === rd.y[i]) right++
      const last = s.rules[s.rules.length - 1]
      const n = last === undefined ? 0 : last.counts.reduce((a, v) => a + v, 0)
      return {
        metrics: {
          rules: s.rules.length,
          remaining: s.remaining.length,
          coverage: n,
          precision: n === 0 || last === undefined ? 0 : (last.counts[last.label] ?? 0) / n,
          accuracy: right / Math.max(1, rd.n),
          covered: covered.reduce((a, v) => a + v, 0),
        },
        messages:
          s.last === null
            ? []
            : [
                {
                  kind: 'rule',
                  payload: { literals: s.last.literals.map((l) => literalText(l)), trail: [...s.last.trail] } as Json,
                },
              ],
      }
    },
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as CoveringState,
  }
}

/**
 * Learn rules by sequential covering.
 *
 * @param data - Features and class labels.
 * @param options - The learner and its limits, and tracing.
 * @returns The rule model.
 * @throws MdmlError with code `value` for options out of range.
 * @example
 * ```ts
 * import { fitCovering, modelText } from 'mdml-engine/rules'
 * const data = { x: [[0], [1], [2], [3], [4], [5]], y: [0, 0, 0, 1, 1, 1] }
 * modelText(fitCovering(data, { alpha: 1, minCovered: 1 })) // => 'IF x0 <= 2.5 THEN 0\nELSE 1'
 * ```
 */
export function fitCovering(
  data: LearnData,
  options: CoveringOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): RuleModel {
  const { tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const { state } = runAlgorithm(coveringAlgorithm(data, rest), { ...run, maxSteps: 100_000 })
  return coveringModel(data, state, rest)
}
