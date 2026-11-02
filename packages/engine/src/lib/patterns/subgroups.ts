/**
 * Subgroup discovery: finding conjunctions of conditions on columns whose rows have an unusual target, ranked by a
 * quality function. The quality functions of pysubgroup (the Klösgen family with weighted relative accuracy as its
 * centre, the chi-squared statistic, lift, and the mean shift of a numeric target), the selectors of nominal and
 * binned numeric columns, search by beam or exhaustively over the refinement lattice of `search`, and the two
 * covering variants that return a diverse set of subgroups by reweighting examples: CN2-SD (Lavrač et al. 2004) and
 * Apriori-SD (Kavšek and Lavrač 2006), each a traced algorithm whose step is one rule.
 *
 * @index function Subgroup discovery with quality functions and beam search.
 */

import { type Json, MdmlError } from '../base/index.ts'
import {
  conjunctions,
  type RefinementProblem,
  refinementBeamSearch,
  refinementExhaustive,
  type Selector,
} from '../search/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'

/**
 * The (possibly weighted) counts a binary quality function sees: the subgroup's size and positives, and the whole
 * data's size and positives.
 */
export interface SubgroupCounts {
  /** Rows (or their total weight) in the subgroup. */
  readonly size: number
  /** Positive rows (or their weight) in the subgroup. */
  readonly positives: number
  /** Rows in all the data. */
  readonly total: number
  /** Positive rows in all the data. */
  readonly totalPositives: number
}

/** The quality functions for a binary target. */
export type SubgroupMeasure = 'wracc' | 'klosgen' | 'relative-accuracy' | 'lift' | 'chi-squared'

/**
 * A quality function of a binary target, pysubgroup's. `wracc` is `(n/N)(p/n - P/N)` (the `StandardQF` with a = 1);
 * `klosgen` the same with `sqrt(n/N)` (a = 0.5), favouring smaller, purer subgroups; `relative-accuracy` the plain
 * difference `p/n - P/N` (a = 0); `lift` the ratio `(p/n)/(P/N)`; `chi-squared` the Pearson statistic of the 2 by 2
 * table, signed positive when the subgroup's target rate is above the data's and negative below.
 *
 * @param measure - Which function.
 * @param a - The exponent of the size factor of the Klösgen family (default from the measure).
 * @returns A function of the counts.
 * @example
 * ```ts
 * import { subgroupQuality } from 'mdml-engine/patterns'
 * subgroupQuality('wracc')({ size: 20, positives: 15, total: 100, totalPositives: 30 }) // => 0.09
 * ```
 */
export function subgroupQuality(measure: SubgroupMeasure, a?: number): (counts: SubgroupCounts) => number {
  return ({ size: n, positives: p, total: N, totalPositives: P }) => {
    if (n <= 0) return measure === 'lift' ? 0 : Number.NEGATIVE_INFINITY
    const rate = p / n
    const base = P / N
    switch (measure) {
      case 'wracc':
        return (n / N) ** (a ?? 1) * (rate - base)
      case 'klosgen':
        return (n / N) ** (a ?? 0.5) * (rate - base)
      case 'relative-accuracy':
        return rate - base
      case 'lift':
        return base === 0 ? 0 : rate / base
      case 'chi-squared': {
        const m = N - n
        if (m <= 0 || base <= 0 || base >= 1) return 0
        const q = P - p
        const chi =
          (p - n * base) ** 2 / (n * base) +
          (n - p - n * (1 - base)) ** 2 / (n * (1 - base)) +
          (q - m * base) ** 2 / (m * base) +
          (m - q - m * (1 - base)) ** 2 / (m * (1 - base))
        return rate >= base ? chi : -chi
      }
    }
  }
}

/**
 * The counts of a coverage against a binary target, with optional example weights.
 *
 * @param coverage - 0/1 per row: in the subgroup or not.
 * @param positive - 0/1 per row: the target.
 * @param weights - Row weights (default 1 each).
 * @returns The counts.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @example
 * ```ts
 * import { subgroupCounts } from 'mdml-engine/patterns'
 * subgroupCounts([1, 1, 0, 0], [1, 0, 1, 0]) // => { size: 2, positives: 1, total: 4, totalPositives: 2 }
 * ```
 */
export function subgroupCounts(
  coverage: ArrayLike<number>,
  positive: ArrayLike<number>,
  weights?: ArrayLike<number>,
): SubgroupCounts {
  const n = positive.length
  if (coverage.length !== n || (weights !== undefined && weights.length !== n)) {
    throw new MdmlError('shape', 'subgroupCounts: lengths differ')
  }
  let size = 0
  let positives = 0
  let total = 0
  let totalPositives = 0
  for (let i = 0; i < n; i++) {
    const w = weights?.[i] ?? 1
    const y = (positive[i] ?? 0) > 0 ? 1 : 0
    total += w
    totalPositives += w * y
    if ((coverage[i] ?? 0) !== 0) {
      size += w
      positives += w * y
    }
  }
  return { size, positives, total, totalPositives }
}

/** A subgroup found: the conjunction, the rows it covers and its statistics. */
export interface Subgroup {
  /** The conjunction as text (`a=x & b=p`). */
  readonly description: string
  /** Indices of the selectors it combines. */
  readonly selectors: readonly number[]
  readonly size: number
  /** Positive rows (or, for a numeric target, the subgroup mean). */
  readonly positives: number
  readonly quality: number
  /** 0/1 per row. */
  readonly coverage: readonly number[]
}

/** Options of {@link columnSelectors}. */
export interface ColumnSelectorOptions {
  /** Columns to treat as nominal even if numeric (strings always are). */
  readonly nominal?: readonly string[]
  /** Equal-frequency bins for a numeric column: `bins - 1` thresholds, each giving `<=` and `>` (default 4). */
  readonly bins?: number
  /** Columns to leave out. */
  readonly ignore?: readonly string[]
}

/**
 * The selectors of a table, pysubgroup's `create_selectors`: `column=value` for each value of a nominal column, and
 * `column<=t` and `column>t` at equal-frequency thresholds for a numeric one. Selectors on the same column are not
 * combined by the searches.
 *
 * @param columns - Columns by name, each a list of strings or numbers.
 * @param options - Nominal columns, bins and columns to ignore.
 * @returns The selectors, with 0/1 coverages, in column order.
 * @example
 * ```ts
 * import { columnSelectors } from 'mdml-engine/patterns'
 * columnSelectors({ colour: ['r', 'g', 'r'], size: [1, 2, 3, 4].slice(0, 3) }, { bins: 2 }).map((s) => s.name) // => ['colour=g', 'colour=r', 'size<=1.5', 'size>1.5']
 * ```
 */
export function columnSelectors(
  columns: Readonly<Record<string, ArrayLike<number | string>>>,
  options: ColumnSelectorOptions = {},
): Selector[] {
  const bins = options.bins ?? 4
  const out: Selector[] = []
  for (const [name, col] of Object.entries(columns)) {
    if (options.ignore?.includes(name) === true) continue
    const values = Array.from(col)
    if (values.some((v) => typeof v === 'string') || options.nominal?.includes(name) === true) {
      const levels = [...new Set(values.map(String))].sort()
      for (const v of levels) {
        out.push({ name: `${name}=${v}`, attribute: name, covers: values.map((x) => (String(x) === v ? 1 : 0)) })
      }
    } else {
      const nums = (values as number[]).slice().sort((a, b) => a - b)
      const distinct = [...new Set(nums)]
      const cuts = new Set<number>()
      for (let b = 1; b < bins; b++) {
        const v = nums[Math.min(nums.length - 1, Math.floor((b * nums.length) / bins))] as number
        const i = distinct.indexOf(v)
        if (i > 0) cuts.add(((distinct[i - 1] as number) + v) / 2)
      }
      for (const t of [...cuts].sort((a, b) => a - b)) {
        const f = String(Number(t.toPrecision(4)))
        out.push({ name: `${name}<=${f}`, attribute: name, covers: (values as number[]).map((x) => (x <= t ? 1 : 0)) })
        out.push({ name: `${name}>${f}`, attribute: name, covers: (values as number[]).map((x) => (x > t ? 1 : 0)) })
      }
    }
  }
  return out
}

/** The target of a subgroup search: binary labels, or numeric values whose mean shift is sought. */
export type SubgroupTarget =
  | { readonly kind: 'binary'; readonly positive: ArrayLike<number> }
  | { readonly kind: 'numeric'; readonly values: ArrayLike<number> }

/** Options of {@link discoverSubgroups}. */
export interface SubgroupOptions {
  /** The quality function for a binary target (default `wracc`). */
  readonly measure?: SubgroupMeasure
  /** Exponent of the size factor in the Klösgen family, and in the numeric mean shift (default 1). */
  readonly a?: number
  /** The number of subgroups to return (default 10). */
  readonly k?: number
  /** The most selectors in a description (default 3). */
  readonly depth?: number
  /** `beam` with this width (default 20), or `exhaustive` search with optimistic-estimate pruning for WRAcc. */
  readonly search?: 'beam' | 'exhaustive'
  readonly width?: number
  /** The smallest subgroup in rows (default 1). */
  readonly minSize?: number
}

function numericQuality(values: ArrayLike<number>, a: number): (coverage: Uint8Array) => number {
  const N = values.length
  let mean = 0
  for (let i = 0; i < N; i++) mean += values[i] ?? 0
  mean /= Math.max(1, N)
  return (c) => {
    let n = 0
    let s = 0
    for (let i = 0; i < N; i++) {
      if (c[i] === 1) {
        n++
        s += values[i] ?? 0
      }
    }
    return n === 0 ? Number.NEGATIVE_INFINITY : n ** a * (s / n - mean)
  }
}

/**
 * The refinement problem of a subgroup search for `search`'s beam, best-first and exhaustive searches: conjunctions of
 * the selectors, a quality of the chosen measure, a minimum size, and for weighted relative accuracy pysubgroup's
 * optimistic estimate (the quality of the subgroup's positives alone), which prunes the exhaustive search.
 *
 * @param selectors - The selectors.
 * @param target - The binary or numeric target.
 * @param options - The measure, exponent and minimum size.
 * @param weights - Example weights (binary targets only).
 * @returns The problem.
 * @example
 * ```ts
 * import { subgroupProblem } from 'mdml-engine/patterns'
 * const sel = [{ name: 'a', covers: [1, 1, 0, 0] }]
 * subgroupProblem(sel, { kind: 'binary', positive: [1, 1, 0, 0] }).quality([0]) // => 0.25
 * ```
 */
export function subgroupProblem(
  selectors: readonly Selector[],
  target: SubgroupTarget,
  options: Pick<SubgroupOptions, 'measure' | 'a' | 'minSize'> = {},
  weights?: ArrayLike<number>,
): RefinementProblem<readonly number[]> {
  const minSize = options.minSize ?? 1
  if (target.kind === 'numeric') {
    return conjunctions(selectors, numericQuality(target.values, options.a ?? 1), { minSupport: minSize })
  }
  const measure = options.measure ?? 'wracc'
  const q = subgroupQuality(measure, options.a)
  const positive = target.positive
  const quality = (c: Uint8Array): number => q(subgroupCounts(c, positive, weights))
  if (measure !== 'wracc' || (options.a ?? 1) !== 1) return conjunctions(selectors, quality, { minSupport: minSize })
  const estimate = (c: Uint8Array): number => {
    const k = subgroupCounts(c, positive, weights)
    return (k.positives / k.total) * (1 - k.totalPositives / k.total)
  }
  return conjunctions(selectors, quality, { minSupport: minSize, optimisticEstimate: estimate })
}

/**
 * Subgroup discovery: the best conjunctions of selectors by a quality function, searched by beam or exhaustively with
 * optimistic-estimate pruning. For a binary target the default quality is weighted relative accuracy, and the result
 * agrees with pysubgroup's exhaustive search; for a numeric target it is `size^a * (subgroup mean - overall mean)`.
 *
 * @param selectors - The selectors (see {@link columnSelectors}).
 * @param target - The binary or numeric target.
 * @param options - The measure, number of results, depth, search kind and width, minimum size.
 * @returns The subgroups, best first.
 * @throws MdmlError with code `shape` when the target and selectors differ in length.
 * @example
 * ```ts
 * import { columnSelectors, discoverSubgroups } from 'mdml-engine/patterns'
 * const sel = columnSelectors({ a: ['x', 'x', 'y', 'y', 'y', 'x'], b: ['p', 'q', 'p', 'q', 'p', 'q'] })
 * const best = discoverSubgroups(sel, { kind: 'binary', positive: [1, 1, 0, 0, 0, 1] }, { k: 1, depth: 1 })
 * best[0]?.description // => 'a=x'
 * ```
 */
export function discoverSubgroups(
  selectors: readonly Selector[],
  target: SubgroupTarget,
  options: SubgroupOptions = {},
): Subgroup[] {
  const problem = subgroupProblem(selectors, target, options)
  const run = { k: options.k ?? 10, maxDepth: options.depth ?? 3 }
  const result =
    options.search === 'exhaustive'
      ? refinementExhaustive(problem, run)
      : refinementBeamSearch(problem, { ...run, width: options.width ?? 20 })
  const y = target.kind === 'binary' ? target.positive : target.values
  return result.top.map((t) => {
    const coverage = Array.from(t.coverage ?? problem.coverage?.(t.pattern) ?? [])
    return describe(problem, selectors, t.pattern, t.quality, coverage, y, target.kind)
  })
}

function describe(
  problem: RefinementProblem<readonly number[]>,
  _selectors: readonly Selector[],
  pattern: readonly number[],
  quality: number,
  coverage: readonly number[],
  y: ArrayLike<number>,
  kind: 'binary' | 'numeric',
): Subgroup {
  let size = 0
  let s = 0
  coverage.forEach((c, i) => {
    if (c !== 0) {
      size++
      s += kind === 'binary' ? ((y[i] ?? 0) > 0 ? 1 : 0) : (y[i] ?? 0)
    }
  })
  return {
    description: problem.label?.(pattern) ?? '',
    selectors: pattern,
    size,
    positives: kind === 'binary' ? s : size === 0 ? 0 : s / size,
    quality,
    coverage,
  }
}

// ------------------------------------------------------------------------------------------------ covering variants

/** How the weights of covered positives fall: multiplicatively `gamma^c` (CN2-SD) or additively `1/(c+1)` (Apriori-SD). */
export type WeightScheme = 'multiplicative' | 'additive'

/** Options of the covering subgroup algorithms. */
export interface CoveringSubgroupOptions {
  /** The most rules (default 10). */
  readonly rules?: number
  /** The most selectors in a rule (default 3). */
  readonly depth?: number
  /** The smallest subgroup in rows (default 1). */
  readonly minSize?: number
  /** CN2-SD: the beam width (default 10). */
  readonly width?: number
  /** CN2-SD: the multiplicative factor (default 0.5); the scheme is multiplicative unless `scheme` says otherwise. */
  readonly gamma?: number
  readonly scheme?: WeightScheme
  /** Apriori-SD: the smallest share of the positives a candidate covers (default 0.1). */
  readonly minSupport?: number
  /** Apriori-SD: the smallest confidence of a candidate (default 0). */
  readonly minConfidence?: number
}

/** The state of {@link cn2SdAlgorithm} and {@link aprioriSdAlgorithm}. */
export interface CoveringSubgroupState {
  /** Example weights now. */
  readonly weights: readonly number[]
  /** Times each example has been covered by a chosen rule. */
  readonly covered: readonly number[]
  /** The rules chosen, with statistics on the unweighted data. */
  readonly rules: readonly Omit<Subgroup, 'coverage'>[]
  /** The weighted quality of the last rule chosen. */
  readonly lastQuality: number
  /** Apriori-SD: the candidate selector patterns and the level being generated (empty for CN2-SD). */
  readonly candidates: readonly (readonly number[])[]
  readonly level: number
  readonly done: boolean
}

const COVERING_SCHEMA: TraceSchema = channels({
  metrics: {
    rules: { description: 'subgroup rules chosen so far' },
    quality: { description: 'weighted relative accuracy of the last rule chosen' },
    candidates: { description: 'candidate conjunctions (Apriori-SD)' },
    coveredPositives: { description: 'positive examples covered by at least one rule' },
  },
  messages: {
    rule: { description: 'a rule chosen', payload: { description: 'the conjunction', quality: 'weighted quality' } },
  },
})

function updateWeights(
  weights: number[],
  covered: number[],
  coverage: ArrayLike<number>,
  positive: ArrayLike<number>,
  scheme: WeightScheme,
  gamma: number,
): void {
  for (let i = 0; i < weights.length; i++) {
    if ((coverage[i] ?? 0) !== 0) {
      covered[i] = (covered[i] ?? 0) + 1
      if ((positive[i] ?? 0) > 0)
        weights[i] = scheme === 'multiplicative' ? gamma ** (covered[i] as number) : 1 / ((covered[i] as number) + 1)
    }
  }
}

function coverageOf(selectors: readonly Selector[], pattern: readonly number[], n: number): Uint8Array {
  const out = new Uint8Array(n).fill(1)
  for (const j of pattern) for (let i = 0; i < n; i++) if ((selectors[j]?.covers[i] ?? 0) === 0) out[i] = 0
  return out
}

function chosenRule(
  selectors: readonly Selector[],
  pattern: readonly number[],
  positive: ArrayLike<number>,
  quality: number,
): Omit<Subgroup, 'coverage'> {
  const n = positive.length
  const c = coverageOf(selectors, pattern, n)
  const k = subgroupCounts(c, positive)
  return {
    description: pattern.length === 0 ? 'true' : pattern.map((j) => selectors[j]?.name ?? '?').join(' & '),
    selectors: [...pattern],
    size: k.size,
    positives: k.positives,
    quality,
  }
}

function emptyCovering(n: number): CoveringSubgroupState {
  return {
    weights: new Array<number>(n).fill(1),
    covered: new Array<number>(n).fill(0),
    rules: [],
    lastQuality: 0,
    candidates: [],
    level: 0,
    done: false,
  }
}

function observeCovering(
  s: CoveringSubgroupState,
  positive: ArrayLike<number>,
): ReturnType<NonNullable<Algorithm<CoveringSubgroupState>['observe']>> {
  let cp = 0
  s.covered.forEach((c, i) => {
    if (c > 0 && (positive[i] ?? 0) > 0) cp++
  })
  const last = s.rules[s.rules.length - 1]
  return {
    metrics: { rules: s.rules.length, quality: s.lastQuality, candidates: s.candidates.length, coveredPositives: cp },
    messages:
      last === undefined ? [] : [{ kind: 'rule', payload: { description: last.description, quality: s.lastQuality } }],
  }
}

/**
 * CN2-SD (Lavrač, Kavšek, Flach and Todorovski 2004) as a traced algorithm, one rule per step: a beam search finds the
 * conjunction of the highest weighted relative accuracy under the current example weights, and the weights of the
 * positives it covers fall (`gamma^c` after c coverings, or `1/(c+1)`), so the next rule must find positives not yet
 * explained. The rules are subgroups with their unweighted size and positives.
 *
 * @param selectors - The selectors.
 * @param positive - The binary target, 0/1 per row.
 * @param options - Rule count, depth, beam width, weight scheme.
 * @returns The algorithm.
 * @throws MdmlError with code `shape` when a selector's length differs from the target's.
 * @example
 * ```ts
 * import { cn2SdAlgorithm } from 'mdml-engine/patterns'
 * cn2SdAlgorithm([{ name: 'a', covers: [1, 0] }], [1, 0]).name // => 'cn2-sd'
 * ```
 */
export function cn2SdAlgorithm(
  selectors: readonly Selector[],
  positive: ArrayLike<number>,
  options: CoveringSubgroupOptions = {},
): Algorithm<CoveringSubgroupState> {
  const n = positive.length
  if (selectors.some((s) => s.covers.length !== n))
    throw new MdmlError('shape', 'cn2Sd: selector lengths differ from the target')
  const maxRules = options.rules ?? 10
  const scheme = options.scheme ?? 'multiplicative'
  const gamma = options.gamma ?? 0.5
  return {
    name: 'cn2-sd',
    config: { rules: maxRules, depth: options.depth ?? 3, width: options.width ?? 10, scheme, gamma },
    schema: COVERING_SCHEMA,
    init: () => emptyCovering(n),
    step: (s) => {
      const problem = subgroupProblem(
        selectors,
        { kind: 'binary', positive },
        { measure: 'wracc', minSize: options.minSize ?? 1 },
        s.weights,
      )
      const top = refinementBeamSearch(problem, { k: 1, maxDepth: options.depth ?? 3, width: options.width ?? 10 })
        .top[0]
      if (top === undefined || !(top.quality > 1e-12)) return { ...s, done: true }
      const weights = [...s.weights]
      const covered = [...s.covered]
      updateWeights(weights, covered, coverageOf(selectors, top.pattern, n), positive, scheme, gamma)
      const rules = [...s.rules, chosenRule(selectors, top.pattern, positive, top.quality)]
      return { ...s, weights, covered, rules, lastQuality: top.quality, done: rules.length >= maxRules }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => observeCovering(s, positive),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as CoveringSubgroupState,
  }
}

/**
 * Apriori-SD (Kavšek and Lavrač 2006) as a traced algorithm. The first steps generate candidate conjunctions level by
 * level (a conjunction extends a frequent shorter one and must cover at least `minSupport` of the positives with at
 * least `minConfidence`); the later steps pick one rule each, the candidate of the highest weighted relative accuracy,
 * lowering the weights of the positives it covers to `1/(c+1)`.
 *
 * @param selectors - The selectors.
 * @param positive - The binary target, 0/1 per row.
 * @param options - Rule count, depth, minimum support and confidence.
 * @returns The algorithm.
 * @throws MdmlError with code `shape` when a selector's length differs from the target's.
 * @example
 * ```ts
 * import { aprioriSdAlgorithm } from 'mdml-engine/patterns'
 * aprioriSdAlgorithm([{ name: 'a', covers: [1, 0] }], [1, 0]).name // => 'apriori-sd'
 * ```
 */
export function aprioriSdAlgorithm(
  selectors: readonly Selector[],
  positive: ArrayLike<number>,
  options: CoveringSubgroupOptions = {},
): Algorithm<CoveringSubgroupState> {
  const n = positive.length
  if (selectors.some((s) => s.covers.length !== n))
    throw new MdmlError('shape', 'aprioriSd: selector lengths differ from the target')
  const maxRules = options.rules ?? 10
  const depth = options.depth ?? 3
  const totalPositives = Array.from(positive).filter((v) => v > 0).length
  const minPositives = Math.max(1, Math.ceil((options.minSupport ?? 0.1) * totalPositives - 1e-9))
  const minConfidence = options.minConfidence ?? 0
  const wracc = subgroupQuality('wracc')
  const ok = (pattern: readonly number[]): boolean => {
    const k = subgroupCounts(coverageOf(selectors, pattern, n), positive)
    return k.positives >= minPositives && k.size >= (options.minSize ?? 1) && k.positives / k.size >= minConfidence
  }
  return {
    name: 'apriori-sd',
    config: { rules: maxRules, depth, minSupport: options.minSupport ?? 0.1, minConfidence },
    schema: COVERING_SCHEMA,
    init: () => emptyCovering(n),
    step: (s) => {
      if (s.level < depth && !s.done) {
        // Generate the next level: extend each pattern of the last level by a later selector on a new attribute.
        const previous = s.level === 0 ? [[] as number[]] : s.candidates.filter((p) => p.length === s.level)
        const next: number[][] = []
        for (const p of previous) {
          const used = new Set(p.map((j) => selectors[j]?.attribute ?? `#${j}`))
          for (let j = (p[p.length - 1] ?? -1) + 1; j < selectors.length; j++) {
            if (used.has(selectors[j]?.attribute ?? `#${j}`)) continue
            const q = [...p, j]
            if (ok(q)) next.push(q)
          }
        }
        return { ...s, level: s.level + 1, candidates: [...s.candidates, ...next], done: s.done }
      }
      let best: { pattern: readonly number[]; quality: number } | undefined
      for (const p of s.candidates) {
        const k = subgroupCounts(coverageOf(selectors, p, n), positive, s.weights)
        const quality = wracc(k)
        if (s.rules.some((r) => r.selectors.join() === p.join())) continue
        if (best === undefined || quality > best.quality) best = { pattern: p, quality }
      }
      if (best === undefined || !(best.quality > 1e-12)) return { ...s, level: depth, done: true }
      const weights = [...s.weights]
      const covered = [...s.covered]
      updateWeights(weights, covered, coverageOf(selectors, best.pattern, n), positive, 'additive', 0)
      const rules = [...s.rules, chosenRule(selectors, best.pattern, positive, best.quality)]
      return {
        ...s,
        level: Math.max(s.level, depth),
        weights,
        covered,
        rules,
        lastQuality: best.quality,
        done: rules.length >= maxRules,
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => observeCovering(s, positive),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as CoveringSubgroupState,
  }
}

/** Tracing options for the covering subgroup runs. */
export type CoveringRunOptions = CoveringSubgroupOptions &
  Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>>

function runCovering(
  alg: Algorithm<CoveringSubgroupState>,
  run: Partial<RunOptions>,
): readonly Omit<Subgroup, 'coverage'>[] {
  return runAlgorithm(alg, { ...run, maxSteps: 10_000 }).state.rules
}

const RUN_KEYS = ['tracer', 'policy', 'parent', 'runId'] as const

function split(options: CoveringRunOptions): { opts: CoveringSubgroupOptions; run: Partial<RunOptions> } {
  const run: Record<string, unknown> = {}
  const opts: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(options)) {
    if ((RUN_KEYS as readonly string[]).includes(k)) run[k] = v
    else opts[k] = v
  }
  return { opts: opts as CoveringSubgroupOptions, run: run as Partial<RunOptions> }
}

/**
 * CN2-SD subgroups: a diverse set of rules by weighted covering with beam search.
 *
 * @param selectors - The selectors.
 * @param positive - The binary target.
 * @param options - Rule count, depth, beam width, weight scheme and tracing.
 * @returns The rules in the order chosen, with unweighted sizes and positives and the weighted quality at choice.
 * @throws MdmlError with code `shape` when a selector's length differs from the target's.
 * @example
 * ```ts
 * import { columnSelectors, cn2Sd } from 'mdml-engine/patterns'
 * const sel = columnSelectors({ a: ['x', 'x', 'x', 'y', 'y', 'y'], b: ['p', 'q', 'q', 'p', 'q', 'q'] })
 * cn2Sd(sel, [1, 1, 1, 0, 0, 0], { rules: 1, depth: 1 })[0]?.description // => 'a=x'
 * ```
 */
export function cn2Sd(
  selectors: readonly Selector[],
  positive: ArrayLike<number>,
  options: CoveringRunOptions = {},
): readonly Omit<Subgroup, 'coverage'>[] {
  const { opts, run } = split(options)
  return runCovering(cn2SdAlgorithm(selectors, positive, opts), run)
}

/**
 * Apriori-SD subgroups: candidates by level-wise generation, rules chosen by weighted covering.
 *
 * @param selectors - The selectors.
 * @param positive - The binary target.
 * @param options - Rule count, depth, minimum support and confidence, and tracing.
 * @returns The rules in the order chosen.
 * @throws MdmlError with code `shape` when a selector's length differs from the target's.
 * @example
 * ```ts
 * import { columnSelectors, aprioriSd } from 'mdml-engine/patterns'
 * const sel = columnSelectors({ a: ['x', 'x', 'x', 'y', 'y', 'y'], b: ['p', 'q', 'q', 'p', 'q', 'q'] })
 * aprioriSd(sel, [1, 1, 1, 0, 0, 0], { rules: 1, depth: 1, minSupport: 0.5 })[0]?.description // => 'a=x'
 * ```
 */
export function aprioriSd(
  selectors: readonly Selector[],
  positive: ArrayLike<number>,
  options: CoveringRunOptions = {},
): readonly Omit<Subgroup, 'coverage'>[] {
  const { opts, run } = split(options)
  return runCovering(aprioriSdAlgorithm(selectors, positive, opts), run)
}
