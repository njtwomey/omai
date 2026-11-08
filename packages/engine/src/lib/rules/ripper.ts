/**
 * RIPPER (Cohen 1995) and its predecessor IREP (Fürnkranz and Widmer 1994). Classes are taken in order of increasing
 * frequency and a rule set is learned for each against all the rest: a rule is grown on two thirds of the rows by
 * FOIL's information gain, pruned on the other third by deleting final literals while (p - n) / (p + n) rises, and added
 * unless its error on the pruning set reaches one half or the rule set's description length exceeds the shortest seen by
 * more than 64 bits (the MDL stopping rule). RIPPER then optimises the set: each rule in turn is replaced by a rule grown
 * from scratch and revised by growing it further, keeping whichever of the three gives the shortest description length
 * of the whole set, and rules that no longer pay for themselves are deleted. The last class is the default. A step adds
 * one rule or revisits one rule.
 */

import { type Json, OmaiError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { foilGain } from '../logic/ilp/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { foldIn, permutation } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { conflicts, countClasses, type LiteralPool, literalPool, type RuleData, ruleData } from './_table.ts'
import { coverageOf, type Literal, literalText, type Rule, type RuleModel } from './rule.ts'

/** Options of RIPPER. */
export interface RipperOptions {
  /** `ripper` (grow, prune, MDL stop, optimise) or `irep` (grow and prune with the 50% error stop only; default `ripper`). */
  readonly mode?: 'ripper' | 'irep'
  /** Optimisation passes (default 2; ignored by `irep`). */
  readonly optimisations?: number
  /** The share of rows used to grow a rule, the rest prune it (default 2/3). */
  readonly growFraction?: number
  /** The most literals in a rule (default 8). */
  readonly maxLength?: number
  /** Bits by which the description length may exceed its minimum before rule addition stops (default 64). */
  readonly slack?: number
  /** The most thresholds per numeric feature (default 24). */
  readonly maxThresholds?: number
  /** Indices of features whose values are category codes (default none). */
  readonly nominal?: readonly number[]
}

/** The state of RIPPER. */
export type RipperState = {
  /** Class indices from the least to the most frequent. */
  readonly order: readonly number[]
  /** The class being learned (an index into `order`). */
  readonly ci: number
  readonly phase: 'build' | 'optimise' | 'done'
  /** `false` before the first optimisation, `true` for the rebuild after it. */
  readonly rebuilt: boolean
  /** Rules of all classes so far, in class order (each rule's `label` is its class index). */
  readonly rules: readonly Rule[]
  /** Where the current class's rules begin in `rules`. */
  readonly start: number
  /** Rows not covered by an earlier class's rules. */
  readonly inPlay: readonly number[]
  readonly minDL: number
  readonly pass: number
  /** The rule being revisited, an index into the current class's rules. */
  readonly index: number
  /** The description length (bits) of the current class's rule set. */
  readonly dl: number
  /** What the last step did. */
  readonly event: { readonly kind: string; readonly detail: string } | null
}

const RIPPER_SCHEMA: TraceSchema = channels({
  metrics: {
    rules: { description: 'rules in the rule set so far' },
    descriptionLength: {
      description: 'description length of the current class’s rule set in bits',
      unit: 'bits',
      direction: 'lower',
    },
    minDescriptionLength: {
      description: 'the shortest description length seen for this class',
      unit: 'bits',
      direction: 'lower',
    },
    inPlay: { description: 'rows not covered by an earlier class’s rules' },
  },
  messages: {
    step: {
      description: 'what the step did: a rule grown, a stop, a revision',
      payload: { kind: 'event kind', detail: 'text' },
    },
  },
})

const log2 = Math.log2

/** Rissanen's universal code length for an integer k >= 1, in bits. */
function log2Star(k: number): number {
  let bits = Math.log2(2.865064)
  for (let t = Math.log2(k); t > 0; t = Math.log2(t)) bits += t
  return bits
}

/** log2 of the binomial coefficient C(n, k). */
function logChoose2(n: number, k: number): number {
  if (k < 0 || k > n) return 0
  return (item(lgamma(n + 1)) - item(lgamma(k + 1)) - item(lgamma(n - k + 1))) / Math.LN2
}

/**
 * The description length in bits of a rule set for one class over the rows in play, after Cohen: each rule costs half
 * of the code for its length plus the choice of its `k` literals among the `nLiterals` available, and the exceptions
 * cost the choice of which covered rows are false positives and which uncovered rows are false negatives.
 *
 * @param ruleLengths - The number of literals of each rule.
 * @param nLiterals - The number of candidate literals.
 * @param covered - Rows in play the set covers.
 * @param falsePositives - Covered rows not of the class.
 * @param uncovered - Rows in play the set does not cover.
 * @param falseNegatives - Uncovered rows of the class.
 * @returns The description length.
 * @example
 * ```ts
 * import { ruleSetDescriptionLength } from 'omai-engine/rules'
 * ruleSetDescriptionLength([], 10, 0, 0, 10, 4) > ruleSetDescriptionLength([2], 10, 4, 0, 6, 0) // => true
 * ```
 */
export function ruleSetDescriptionLength(
  ruleLengths: readonly number[],
  nLiterals: number,
  covered: number,
  falsePositives: number,
  uncovered: number,
  falseNegatives: number,
): number {
  let bits = 0
  for (const k of ruleLengths) bits += 0.5 * (log2Star(Math.max(1, k)) + logChoose2(nLiterals, k))
  bits +=
    log2(covered + 1) +
    log2(uncovered + 1) +
    logChoose2(covered, falsePositives) +
    logChoose2(uncovered, falseNegatives)
  return bits
}

type Ctx = { readonly rd: RuleData; readonly pool: LiteralPool; readonly o: Required<Omit<RipperOptions, 'nominal'>> }

/** Description length of a set of rules (as literal lists) for class `c` over some rows. */
function setDL(ctx: Ctx, lits: readonly (readonly Literal[])[], c: number, rows: readonly number[]): number {
  const covered = new Uint8Array(ctx.rd.n)
  for (const l of lits) {
    const m = coverageOf(l, ctx.rd.x, ctx.rd.d)
    for (let i = 0; i < ctx.rd.n; i++) if (m[i] === 1) covered[i] = 1
  }
  let cov = 0
  let fp = 0
  let unc = 0
  let fn = 0
  for (const r of rows) {
    const isC = ctx.rd.y[r] === c
    if (covered[r] === 1) {
      cov++
      if (!isC) fp++
    } else {
      unc++
      if (isC) fn++
    }
  }
  return ruleSetDescriptionLength(
    lits.map((l) => l.length),
    ctx.pool.literals.length,
    cov,
    fp,
    unc,
    fn,
  )
}

function maskOf(ctx: Ctx, lits: readonly Literal[]): Uint8Array {
  return coverageOf(lits, ctx.rd.x, ctx.rd.d)
}

function countPN(ctx: Ctx, mask: Uint8Array, rows: readonly number[], c: number): [number, number] {
  let p = 0
  let n = 0
  for (const r of rows) {
    if (mask[r] !== 1) continue
    if (ctx.rd.y[r] === c) p++
    else n++
  }
  return [p, n]
}

/** Grow a rule from `start` on the grow rows by FOIL gain until it covers no negative or nothing gains. */
function growRule(ctx: Ctx, grow: readonly number[], c: number, start: readonly Literal[]): Literal[] {
  const lits = [...start]
  const idx: number[] = []
  for (const l of start) {
    const found = ctx.pool.literals.findIndex((p) => p.feature === l.feature && p.op === l.op && p.value === l.value)
    idx.push(found)
  }
  let cur = grow.filter((r) => maskOf(ctx, lits)[r] === 1)
  while (lits.length < ctx.o.maxLength) {
    let p0 = 0
    let n0 = 0
    for (const r of cur)
      if (ctx.rd.y[r] === c) p0++
      else n0++
    if (n0 === 0 || p0 === 0) break
    let best = -1
    let bestGain = 1e-12
    for (let l = 0; l < ctx.pool.literals.length; l++) {
      if (
        conflicts(
          ctx.pool,
          idx.filter((i) => i >= 0),
          l,
        )
      )
        continue
      const mask = ctx.pool.masks[l] as Uint8Array
      let p1 = 0
      let n1 = 0
      for (const r of cur) {
        if (mask[r] !== 1) continue
        if (ctx.rd.y[r] === c) p1++
        else n1++
      }
      const gain = foilGain(p0, n0, p1, n1, p1)
      if (gain > bestGain) {
        bestGain = gain
        best = l
      }
    }
    if (best < 0) break
    lits.push(ctx.pool.literals[best] as Literal)
    idx.push(best)
    const mask = ctx.pool.masks[best] as Uint8Array
    cur = cur.filter((r) => mask[r] === 1)
  }
  return lits
}

/** Prune final literals while (p - n) / (p + n) on the prune rows does not fall; returns the pruned rule and its error. */
function pruneRule(
  ctx: Ctx,
  lits: readonly Literal[],
  prune: readonly number[],
  c: number,
): { lits: Literal[]; error: number } {
  let bestK = lits.length
  let bestV = Number.NEGATIVE_INFINITY
  let bestErr = 1
  for (let k = lits.length; k >= 1; k--) {
    const [p, n] = countPN(ctx, maskOf(ctx, lits.slice(0, k)), prune, c)
    const v = p + n === 0 ? 0 : (p - n) / (p + n)
    if (v >= bestV) {
      bestV = v
      bestK = k
      bestErr = p + n === 0 ? 1 : n / (p + n)
    }
  }
  return { lits: lits.slice(0, bestK), error: bestErr }
}

function split(
  ctx: Ctx,
  rows: readonly number[],
  c: number,
  key: ReturnType<typeof foldIn>,
): { grow: number[]; prune: number[] } {
  const grow: number[] = []
  const prune: number[] = []
  for (const isC of [true, false]) {
    const group = rows.filter((r) => (ctx.rd.y[r] === c) === isC)
    const order = permutation(foldIn(key, isC ? 1 : 2), group.length).data
    const cut = Math.max(Math.min(group.length, 1), Math.round(ctx.o.growFraction * group.length))
    group.forEach((_, i) => {
      const r = group[order[i] as number] as number
      if (i < cut) grow.push(r)
      else prune.push(r)
    })
  }
  return { grow, prune }
}

function classRules(s: RipperState): Rule[] {
  return s.rules.slice(s.start) as Rule[]
}

/** The model a RIPPER state defines: an ordered list whose counts are recomputed in sequence over the training set. */
function modelOfRipper(rd: RuleData, s: RipperState): RuleModel {
  const left = new Uint8Array(rd.n).fill(1)
  const rules: Rule[] = s.rules.map((r) => {
    const m = coverageOf(r.literals, rd.x, rd.d)
    const counts = new Array<number>(rd.k).fill(0)
    for (let i = 0; i < rd.n; i++) {
      if (m[i] === 1 && left[i] === 1) {
        counts[rd.y[i] as number] = (counts[rd.y[i] as number] ?? 0) + 1
        left[i] = 0
      }
    }
    return { literals: r.literals, label: r.label, counts }
  })
  const rest = new Array<number>(rd.k).fill(0)
  for (let i = 0; i < rd.n; i++) if (left[i] === 1) rest[rd.y[i] as number] = (rest[rd.y[i] as number] ?? 0) + 1
  const total = countClasses(
    rd,
    Int32Array.from({ length: rd.n }, (_, i) => i),
  )
  const last = s.order[s.order.length - 1] ?? 0
  const defaults = rest.some((v) => v > 0) ? rest : total.map((_, c) => (c === last ? 1 : 0))
  return { kind: 'list', classes: rd.classes, nFeatures: rd.d, rules, defaultCounts: defaults }
}

/**
 * RIPPER or IREP as a traced algorithm, one rule per step. The build phase of each class adds a grown and pruned rule
 * per step until the 50% error or MDL stop; the optimisation phase then revisits one rule per step (keep, replace by a
 * freshly grown rule, or revise by growing it further, whichever gives the set the shortest description length); a final
 * build adds rules for the positives the optimised set misses, and a last pass deletes rules whose removal shortens the
 * description. The randomness is the split of the rows into growing and pruning sets, from the step's key.
 *
 * @param data - Features and class labels.
 * @param options - Mode, optimisation passes, split, length and slack.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a grow fraction outside (0, 1).
 * @example
 * ```ts
 * import { ripperAlgorithm } from 'omai-engine/rules'
 * ripperAlgorithm({ x: [[0], [1]], y: [0, 1] }).name // => 'ripper'
 * ```
 */
export function ripperAlgorithm(data: LearnData, options: RipperOptions = {}): Algorithm<RipperState> {
  const rd = ruleData(data, options.nominal ?? [], 'ripper')
  const mode = options.mode ?? 'ripper'
  const o = {
    mode,
    optimisations: mode === 'irep' ? 0 : (options.optimisations ?? 2),
    growFraction: options.growFraction ?? 2 / 3,
    maxLength: options.maxLength ?? 8,
    slack: options.slack ?? 64,
    maxThresholds: options.maxThresholds ?? 24,
  }
  if (!(o.growFraction > 0 && o.growFraction < 1))
    throw new OmaiError('value', 'ripper: growFraction must be in (0, 1)')
  const ctx: Ctx = { rd, pool: literalPool(rd, o.maxThresholds), o }
  const counts = countClasses(
    rd,
    Int32Array.from({ length: rd.n }, (_, i) => i),
  )
  const order = counts.map((_, c) => c).sort((a, b) => (counts[a] ?? 0) - (counts[b] ?? 0) || a - b)
  const all = Array.from({ length: rd.n }, (_, i) => i)
  const startClass = (s: RipperState, ci: number, rules: readonly Rule[], inPlay: readonly number[]): RipperState => {
    const c = order[ci] as number
    const dl = setDL(ctx, [], c, inPlay)
    const phase: RipperState['phase'] = ci >= order.length - 1 ? 'done' : 'build'
    return { ...s, ci, phase, rebuilt: false, rules, start: rules.length, inPlay, minDL: dl, pass: 0, index: 0, dl }
  }
  return {
    name: mode,
    config: { ...o },
    schema: RIPPER_SCHEMA,
    init: () => {
      const s0: RipperState = {
        order,
        ci: 0,
        phase: 'build',
        rebuilt: false,
        rules: [],
        start: 0,
        inPlay: all,
        minDL: 0,
        pass: 0,
        index: 0,
        dl: 0,
        event: null,
      }
      return order.length < 2 || rd.n === 0 ? { ...s0, phase: 'done' } : startClass(s0, 0, [], all)
    },
    step: (s, stepCtx) => {
      const c = order[s.ci] as number
      const mine = classRules(s)
      const uncoveredRows = (excluding: number): number[] => {
        const m = new Uint8Array(rd.n)
        mine.forEach((r, i) => {
          if (i === excluding) return
          const cm = maskOf(ctx, r.literals)
          for (let j = 0; j < rd.n; j++) if (cm[j] === 1) m[j] = 1
        })
        return s.inPlay.filter((r) => m[r] !== 1)
      }
      const finishClass = (rules: readonly Rule[], event: { kind: string; detail: string }): RipperState => {
        const mask = new Uint8Array(rd.n)
        for (const r of rules.slice(s.start)) {
          const cm = maskOf(ctx, r.literals)
          for (let j = 0; j < rd.n; j++) if (cm[j] === 1) mask[j] = 1
        }
        const next = s.inPlay.filter((r) => mask[r] !== 1)
        return { ...startClass({ ...s, event }, s.ci + 1, rules, next), event }
      }
      if (s.phase === 'build') {
        const rows = uncoveredRows(-1)
        const key = foldIn(stepCtx.key, s.rules.length + 1)
        const hasPositive = rows.some((r) => rd.y[r] === c)
        if (!hasPositive) {
          return o.optimisations > 0 && !s.rebuilt
            ? {
                ...s,
                phase: 'optimise',
                pass: 0,
                index: 0,
                event: { kind: 'stop', detail: 'no positives left; optimising' },
              }
            : finishClass(o.optimisations > 0 ? prunedByDL(ctx, s, c) : s.rules, {
                kind: 'stop',
                detail: 'no positives left',
              })
        }
        const { grow, prune } = split(ctx, rows, c, key)
        const grown = growRule(ctx, grow, c, [])
        const pruned = pruneRule(ctx, grown, prune.length > 0 ? prune : grow, c)
        const [pc] = countPN(ctx, maskOf(ctx, pruned.lits), rows, c)
        const dl = setDL(ctx, [...mine.map((r) => r.literals), pruned.lits], c, s.inPlay)
        const stop =
          pruned.lits.length === 0 || pc === 0
            ? 'no useful rule'
            : pruned.error >= 0.5
              ? 'pruning error reached 50%'
              : mode === 'ripper' && dl > s.minDL + o.slack
                ? 'description length exceeded the minimum by the slack'
                : null
        if (stop !== null) {
          return o.optimisations > 0 && !s.rebuilt
            ? { ...s, phase: 'optimise', pass: 0, index: 0, event: { kind: 'stop', detail: `${stop}; optimising` } }
            : finishClass(o.optimisations > 0 ? prunedByDL(ctx, s, c) : s.rules, { kind: 'stop', detail: stop })
        }
        const rule: Rule = { literals: pruned.lits, label: c, counts: [] }
        return {
          ...s,
          rules: [...s.rules, rule],
          dl,
          minDL: Math.min(s.minDL, dl),
          event: { kind: 'rule', detail: pruned.lits.map((l) => literalText(l)).join(' AND ') },
        }
      }
      if (s.phase === 'optimise') {
        if (mine.length === 0 || s.pass >= o.optimisations) {
          return {
            ...s,
            phase: 'build',
            rebuilt: true,
            event: { kind: 'rebuild', detail: 'adding rules for positives still uncovered' },
          }
        }
        const i = s.index
        const rows = uncoveredRows(i)
        const { grow, prune } = split(ctx, rows, c, foldIn(stepCtx.key, 1000 + s.pass * 100 + i))
        const pr = prune.length > 0 ? prune : grow
        const replacement = pruneRule(ctx, growRule(ctx, grow, c, []), pr, c).lits
        const revision = pruneRule(ctx, growRule(ctx, grow, c, (mine[i] as Rule).literals), pr, c).lits
        const candidates: [string, readonly Literal[]][] = [
          ['keep', (mine[i] as Rule).literals],
          ['replacement', replacement],
          ['revision', revision],
        ]
        let best = 0
        let bestDL = Number.POSITIVE_INFINITY
        candidates.forEach(([, lits], ci2) => {
          if (lits.length === 0) return
          const sets = mine.map((r, j) => (j === i ? lits : r.literals))
          const dl = setDL(ctx, sets, c, s.inPlay)
          if (dl < bestDL - 1e-9) {
            bestDL = dl
            best = ci2
          }
        })
        const chosen = candidates[best] as [string, readonly Literal[]]
        const rulesNow = s.rules.map((r, j) => (j === s.start + i ? { ...r, literals: [...chosen[1]] } : r))
        const nextIndex = i + 1 >= mine.length ? 0 : i + 1
        const nextPass = i + 1 >= mine.length ? s.pass + 1 : s.pass
        return {
          ...s,
          rules: rulesNow,
          index: nextIndex,
          pass: nextPass,
          dl: bestDL,
          minDL: Math.min(s.minDL, bestDL),
          event: { kind: 'revision', detail: `rule ${i + 1}: ${chosen[0]}` },
        }
      }
      return s
    },
    done: (s) => (s.phase === 'done' ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        rules: s.rules.length,
        descriptionLength: s.dl,
        minDescriptionLength: s.minDL,
        inPlay: s.inPlay.length,
      },
      messages: s.event === null ? [] : [{ kind: 'step', payload: s.event as unknown as Json }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as RipperState,
  }
}

/** The state's rules with those of the current class that do not shorten the description length deleted. */
function prunedByDL(ctx: Ctx, s: RipperState, c: number): Rule[] {
  const mine = classRules(s)
  let keep = mine.map((_, i) => i)
  for (let i = mine.length - 1; i >= 0; i--) {
    const without = keep.filter((k) => k !== i)
    const a = setDL(
      ctx,
      keep.map((k) => (mine[k] as Rule).literals),
      c,
      s.inPlay,
    )
    const b = setDL(
      ctx,
      without.map((k) => (mine[k] as Rule).literals),
      c,
      s.inPlay,
    )
    if (b < a) keep = without
  }
  return [...s.rules.slice(0, s.start), ...keep.map((k) => mine[k] as Rule)]
}

/**
 * The model of a RIPPER state: the rules as an ordered list (least frequent class first), their counts recomputed in
 * sequence over the training set, and the rows no rule covers as the default.
 *
 * @param data - The training data.
 * @param s - The state.
 * @param options - The `nominal` features the state was learned with.
 * @returns The model.
 * @example
 * ```ts
 * import { ripperAlgorithm, ripperModel } from 'omai-engine/rules'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const data = { x: [[0], [1], [2], [3], [4], [5], [6], [7]], y: [0, 0, 0, 1, 1, 1, 1, 1] }
 * const { state } = runAlgorithm(ripperAlgorithm(data), { maxSteps: 50, seed: 1 })
 * ripperModel(data, state).rules.length > 0 // => true
 * ```
 */
export function ripperModel(data: LearnData, s: RipperState, options: Pick<RipperOptions, 'nominal'> = {}): RuleModel {
  return modelOfRipper(ruleData(data, options.nominal ?? [], 'ripper'), s)
}

/**
 * Learn a rule list with RIPPER (or IREP).
 *
 * @param data - Features and class labels.
 * @param options - The options of {@link ripperAlgorithm}, a seed (for the grow and prune splits) and tracing.
 * @returns The rule model.
 * @throws OmaiError with code `value` for a grow fraction outside (0, 1).
 * @example
 * ```ts
 * import { fitRipper, modelText } from 'omai-engine/rules'
 * const data = { x: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], y: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] }
 * modelText(fitRipper(data, { seed: 3 })).split('\n').at(-1) // => 'ELSE 1'
 * ```
 */
export function fitRipper(
  data: LearnData,
  options: RipperOptions & { readonly seed?: number } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>
    > = {},
): RuleModel {
  const { seed, tracer, policy, parent, runId, ...rest } = options
  const run: Partial<RunOptions> = {
    ...(seed === undefined ? {} : { seed }),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  }
  const { state } = runAlgorithm(ripperAlgorithm(data, rest), { ...run, maxSteps: 10_000 })
  return ripperModel(data, state, rest)
}
