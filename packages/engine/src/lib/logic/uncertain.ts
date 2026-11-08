/**
 * Reasoning under uncertainty with the calculi of classical expert systems: MYCIN's certainty factors (combination of
 * evidence for a hypothesis, chaining through rules), Dempster-Shafer belief functions (mass functions on subsets of a
 * frame, belief and plausibility, Dempster's and Yager's rules of combination, the pignistic transform, evidence added
 * one item per step as a traced algorithm) and fuzzy inference (membership functions, t-norms and t-conorms, Mamdani and
 * Sugeno rule bases, and the defuzzifiers centroid, bisector and the means, smallest and largest of the maxima).
 */

import { type Json, maxOf, minOf, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type TraceSchema } from '../trace/index.ts'

// ------------------------------------------------------------------------------------------------ certainty factors

/**
 * MYCIN's combination of two certainty factors for the same hypothesis, in [-1, 1]: `a + b (1 - a)` when both support
 * it, `a + b (1 + a)` when both oppose it, and `(a + b) / (1 - min(|a|, |b|))` when they disagree. It is commutative,
 * associative, and 1 and -1 absorb.
 *
 * @param a - A certainty factor.
 * @param b - Another.
 * @returns The combined factor.
 * @throws OmaiError with code `value` for a factor outside [-1, 1], or for the undefined combination of 1 and -1.
 * @example
 * ```ts
 * import { combineCertainty } from 'omai-engine/logic'
 * [combineCertainty(0.6, 0.4), combineCertainty(-0.6, -0.4), combineCertainty(0.8, -0.4)] // => [0.76, -0.76, 0.4 / 0.6]
 * ```
 */
export function combineCertainty(a: number, b: number): number {
  if (!(Math.abs(a) <= 1 && Math.abs(b) <= 1)) throw new OmaiError('value', 'certainty factors lie in [-1, 1]')
  if (a > 0 && b > 0) return a + b * (1 - a)
  if (a < 0 && b < 0) return a + b * (1 + a)
  const d = 1 - Math.min(Math.abs(a), Math.abs(b))
  if (d === 0) throw new OmaiError('value', 'certainty factors 1 and -1 cannot be combined')
  return (a + b) / d
}

/** A certainty-factor rule: if all the conditions hold the conclusion gains `factor` times the weakest condition's certainty. */
export interface CertaintyRule {
  readonly conditions: readonly string[]
  readonly conclusion: string
  /** The rule's certainty factor, in [-1, 1]. */
  readonly factor: number
}

/**
 * Chain certainty factors through a rule base: a rule fires when the weakest of its conditions has certainty above 0.2
 * (MYCIN's threshold) and contributes `factor * min(conditions)` to its conclusion; contributions to one hypothesis are
 * combined with {@link combineCertainty}. Evidence is the initial certainty of some facts; the rule base is applied until
 * nothing changes.
 *
 * @param rules - The rules.
 * @param evidence - The certainty factor of observed facts.
 * @returns The certainty factor of every fact, evidence included.
 * @throws OmaiError with code `value` for a factor outside [-1, 1].
 * @example
 * ```ts
 * import { certaintyInference } from 'omai-engine/logic'
 * const rules = [
 *   { conditions: ['fever', 'cough'], conclusion: 'flu', factor: 0.8 },
 *   { conditions: ['aches'], conclusion: 'flu', factor: 0.5 },
 * ]
 * certaintyInference(rules, { fever: 0.9, cough: 0.6, aches: 1 }).flu // => 0.48 + 0.5 * (1 - 0.48)
 * ```
 */
export function certaintyInference(
  rules: readonly CertaintyRule[],
  evidence: Readonly<Record<string, number>>,
): Record<string, number> {
  for (const v of Object.values(evidence))
    if (!(Math.abs(v) <= 1)) throw new OmaiError('value', 'certainty factors lie in [-1, 1]')
  let cf: Record<string, number> = { ...evidence }
  for (let pass = 0; pass < 100; pass++) {
    const contributions = new Map<string, number[]>()
    for (const r of rules) {
      const weakest = minOf(r.conditions.map((c) => cf[c] ?? 0))
      if (weakest > 0.2)
        contributions.set(r.conclusion, [...(contributions.get(r.conclusion) ?? []), r.factor * weakest])
    }
    const next: Record<string, number> = { ...evidence }
    for (const [h, list] of contributions) {
      let v = evidence[h] ?? list[0] ?? 0
      for (const c of evidence[h] === undefined ? list.slice(1) : list) v = combineCertainty(v, c)
      next[h] = v
    }
    const same = Object.keys({ ...cf, ...next }).every((k) => Math.abs((cf[k] ?? 0) - (next[k] ?? 0)) < 1e-12)
    cf = next
    if (same) break
  }
  return cf
}

// ------------------------------------------------------------------------------------------------ Dempster-Shafer

/** A mass function on subsets of a frame of discernment: the subset (its sorted elements joined by `|`) and its mass. */
export type MassFunction = Readonly<Record<string, number>>

const subsetKey = (items: readonly string[]): string => [...new Set(items)].sort().join('|')
const subsetOf = (key: string): string[] => (key === '' ? [] : key.split('|'))
const intersect = (a: string, b: string): string => {
  const sb = new Set(subsetOf(b))
  return subsetKey(subsetOf(a).filter((x) => sb.has(x)))
}

/**
 * A mass function from subsets given as lists, checked to sum to one; the empty set has no mass (the closed-world
 * assumption). Mass on the whole frame is ignorance.
 *
 * @param frame - The hypotheses.
 * @param masses - Mass by subset, as lists of hypotheses.
 * @returns The mass function.
 * @throws OmaiError with code `value` for a subset outside the frame, a negative mass or masses that do not sum to one.
 * @example
 * ```ts
 * import { massFunction } from 'omai-engine/logic'
 * massFunction(['a', 'b'], [[['a'], 0.6], [['a', 'b'], 0.4]])['a'] // => 0.6
 * ```
 */
export function massFunction(
  frame: readonly string[],
  masses: readonly (readonly [readonly string[], number])[],
): MassFunction {
  const out: Record<string, number> = {}
  let total = 0
  for (const [items, m] of masses) {
    if (m < 0) throw new OmaiError('value', 'masses are non-negative')
    if (items.length === 0 || items.some((i) => !frame.includes(i)))
      throw new OmaiError('value', 'a focal set must be a non-empty subset of the frame')
    out[subsetKey(items)] = (out[subsetKey(items)] ?? 0) + m
    total += m
  }
  if (Math.abs(total - 1) > 1e-9) throw new OmaiError('value', `masses sum to ${total}, not 1`)
  return out
}

/**
 * Belief: the mass of the subsets contained in A.
 *
 * @param m - The mass function.
 * @param a - The hypothesis set.
 * @returns Bel(A).
 * @example
 * ```ts
 * import { belief, massFunction } from 'omai-engine/logic'
 * belief(massFunction(['a', 'b'], [[['a'], 0.6], [['a', 'b'], 0.4]]), ['a']) // => 0.6
 * ```
 */
export function belief(m: MassFunction, a: readonly string[]): number {
  const sa = new Set(a)
  let total = 0
  for (const [k, v] of Object.entries(m)) if (subsetOf(k).every((x) => sa.has(x))) total += v
  return total
}

/**
 * Plausibility: the mass of the subsets that meet A, which is one minus the belief in its complement.
 *
 * @param m - The mass function.
 * @param a - The hypothesis set.
 * @returns Pl(A).
 * @example
 * ```ts
 * import { massFunction, plausibility } from 'omai-engine/logic'
 * plausibility(massFunction(['a', 'b'], [[['a'], 0.6], [['a', 'b'], 0.4]]), ['b']) // => 0.4
 * ```
 */
export function plausibility(m: MassFunction, a: readonly string[]): number {
  const sa = new Set(a)
  let total = 0
  for (const [k, v] of Object.entries(m)) if (subsetOf(k).some((x) => sa.has(x))) total += v
  return total
}

/** The result of combining two bodies of evidence. */
export interface Combination {
  readonly mass: MassFunction
  /** The mass the two bodies assign to incompatible sets (Dempster's K). */
  readonly conflict: number
}

/**
 * Dempster's rule of combination: multiply masses of every pair of focal sets, assign the product to their intersection,
 * and renormalise by `1 - K` where K is the conflict (mass falling on the empty set). Total conflict (K = 1) is an error.
 *
 * @param a - A mass function.
 * @param b - An independent one.
 * @returns The combined masses and the conflict.
 * @throws OmaiError with code `value` when the evidence is totally conflicting.
 * @example
 * ```ts
 * import { combineDempster, massFunction } from 'omai-engine/logic'
 * const f = ['a', 'b', 'c']
 * const r = combineDempster(massFunction(f, [[['a'], 0.99], [['b'], 0.01]]), massFunction(f, [[['c'], 0.99], [['b'], 0.01]]))
 * [Math.round(r.mass['b'] * 1e9) / 1e9, r.conflict] // => [1, 0.9999]
 * ```
 */
export function combineDempster(a: MassFunction, b: MassFunction): Combination {
  const out: Record<string, number> = {}
  let conflict = 0
  for (const [ka, va] of Object.entries(a)) {
    for (const [kb, vb] of Object.entries(b)) {
      const k = intersect(ka, kb)
      if (k === '') conflict += va * vb
      else out[k] = (out[k] ?? 0) + va * vb
    }
  }
  if (conflict >= 1 - 1e-12) throw new OmaiError('value', 'the evidence is totally conflicting')
  for (const k of Object.keys(out)) out[k] = (out[k] as number) / (1 - conflict)
  return { mass: out, conflict }
}

/**
 * Yager's rule: like Dempster's, but the conflicting mass goes to the whole frame (ignorance) instead of being
 * renormalised away, so conflict weakens the conclusion rather than inflating the rest.
 *
 * @param frame - The frame of discernment.
 * @param a - A mass function.
 * @param b - Another.
 * @returns The combined masses and the conflict.
 * @example
 * ```ts
 * import { combineYager, massFunction } from 'omai-engine/logic'
 * const f = ['a', 'b', 'c']
 * combineYager(f, massFunction(f, [[['a'], 0.99], [['b'], 0.01]]), massFunction(f, [[['c'], 0.99], [['b'], 0.01]])).mass['a|b|c'] // => 0.9999
 * ```
 */
export function combineYager(frame: readonly string[], a: MassFunction, b: MassFunction): Combination {
  const out: Record<string, number> = {}
  let conflict = 0
  for (const [ka, va] of Object.entries(a)) {
    for (const [kb, vb] of Object.entries(b)) {
      const k = intersect(ka, kb)
      if (k === '') conflict += va * vb
      else out[k] = (out[k] ?? 0) + va * vb
    }
  }
  const whole = subsetKey(frame)
  out[whole] = (out[whole] ?? 0) + conflict
  return { mass: out, conflict }
}

/**
 * The pignistic transform (Smets): each focal set shares its mass equally among its elements, giving a probability
 * distribution to decide with.
 *
 * @param m - The mass function.
 * @returns Probability by hypothesis.
 * @example
 * ```ts
 * import { massFunction, pignistic } from 'omai-engine/logic'
 * pignistic(massFunction(['a', 'b'], [[['a'], 0.6], [['a', 'b'], 0.4]])) // => { a: 0.8, b: 0.2 }
 * ```
 */
export function pignistic(m: MassFunction): Record<string, number> {
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(m)) {
    const items = subsetOf(k)
    for (const x of items) out[x] = (out[x] ?? 0) + v / items.length
  }
  return out
}

/** The state of evidence combination: the mass so far and the intervals [belief, plausibility] of each hypothesis. */
export interface EvidenceState {
  readonly mass: MassFunction
  readonly conflict: number
  readonly totalConflict: number
  readonly used: number
  readonly intervals: Readonly<Record<string, readonly [number, number]>>
}

const EVIDENCE_SCHEMA: TraceSchema = channels({
  metrics: {
    conflict: { description: 'the conflict between the evidence just added and what was known' },
    totalConflict: { description: 'the conflict accumulated over all evidence, 1 - prod(1 - K)' },
    ignorance: { description: 'mass on the whole frame' },
  },
  messages: {
    interval: { description: 'belief and plausibility of each hypothesis', payload: { intervals: 'by hypothesis' } },
  },
})

/**
 * Evidence combination as a traced algorithm: each step combines one more body of evidence (Dempster's or Yager's rule)
 * into the running mass function and reports the belief interval of every hypothesis, which narrows as consistent
 * evidence arrives and, under Dempster's rule, jumps when the evidence conflicts.
 *
 * @param frame - The frame of discernment.
 * @param evidence - The mass functions to combine, in order.
 * @param rule - `dempster` (default) or `yager`.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { evidenceAlgorithm, massFunction } from 'omai-engine/logic'
 * evidenceAlgorithm(['a', 'b'], [massFunction(['a', 'b'], [[['a'], 1]])]).name // => 'dempster-shafer'
 * ```
 */
export function evidenceAlgorithm(
  frame: readonly string[],
  evidence: readonly MassFunction[],
  rule: 'dempster' | 'yager' = 'dempster',
): Algorithm<EvidenceState> {
  const whole = subsetKey(frame)
  const intervalsOf = (m: MassFunction): Record<string, readonly [number, number]> =>
    Object.fromEntries(frame.map((h) => [h, [belief(m, [h]), plausibility(m, [h])] as const]))
  return {
    name: 'dempster-shafer',
    config: { rule, hypotheses: [...frame] },
    schema: EVIDENCE_SCHEMA,
    init: () => ({
      mass: { [whole]: 1 },
      conflict: 0,
      totalConflict: 0,
      used: 0,
      intervals: intervalsOf({ [whole]: 1 }),
    }),
    step: (s) => {
      const next = evidence[s.used]
      if (next === undefined) return s
      const r = rule === 'dempster' ? combineDempster(s.mass, next) : combineYager(frame, s.mass, next)
      return {
        mass: r.mass,
        conflict: r.conflict,
        totalConflict: 1 - (1 - s.totalConflict) * (1 - r.conflict),
        used: s.used + 1,
        intervals: intervalsOf(r.mass),
      }
    },
    done: (s) => (s.used >= evidence.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { conflict: s.conflict, totalConflict: s.totalConflict, ignorance: s.mass[whole] ?? 0 },
      messages: [{ kind: 'interval', payload: { intervals: s.intervals } as unknown as Json }],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as EvidenceState,
  }
}

// ------------------------------------------------------------------------------------------------ fuzzy logic

/** A membership function on the reals. */
export type Membership = (x: number) => number

/**
 * A triangular membership function rising from `a` to its peak at `b` and falling to `c` (a = b or b = c gives a shoulder).
 *
 * @param a - Left foot.
 * @param b - Peak.
 * @param c - Right foot.
 * @returns The function.
 * @example
 * ```ts
 * import { triangular } from 'omai-engine/logic'
 * [triangular(0, 5, 10)(2.5), triangular(0, 5, 10)(5), triangular(0, 0, 5)(0)] // => [0.5, 1, 1]
 * ```
 */
export function triangular(a: number, b: number, c: number): Membership {
  return (x) => {
    if (x < a || x > c) return 0
    if (x === b) return 1
    return x < b ? (x - a) / (b - a) : (c - x) / (c - b)
  }
}

/**
 * A trapezoidal membership function: 0 below `a`, rising to 1 at `b`, flat to `c`, falling to 0 at `d`.
 *
 * @param a - Left foot.
 * @param b - Start of the plateau.
 * @param c - End of the plateau.
 * @param d - Right foot.
 * @returns The function.
 * @example
 * ```ts
 * import { trapezoid } from 'omai-engine/logic'
 * [trapezoid(0, 2, 4, 6)(1), trapezoid(0, 2, 4, 6)(3), trapezoid(0, 2, 4, 6)(5)] // => [0.5, 1, 0.5]
 * ```
 */
export function trapezoid(a: number, b: number, c: number, d: number): Membership {
  return (x) => {
    if (x < a || x > d) return 0
    if (x >= b && x <= c) return 1
    return x < b ? (x - a) / (b - a) : (d - x) / (d - c)
  }
}

/**
 * A Gaussian membership function `exp(-(x - mean)^2 / (2 sigma^2))`.
 *
 * @param mean - Centre.
 * @param sigma - Width.
 * @returns The function.
 * @example
 * ```ts
 * import { gaussianMembership } from 'omai-engine/logic'
 * gaussianMembership(0, 1)(1) // => Math.exp(-0.5)
 * ```
 */
export function gaussianMembership(mean: number, sigma: number): Membership {
  return (x) => Math.exp(-((x - mean) ** 2) / (2 * sigma * sigma))
}

/** A t-norm (fuzzy and) or its dual t-conorm (fuzzy or): the minimum, the product, or Łukasiewicz's. */
export type TNorm = 'min' | 'product' | 'lukasiewicz'

/**
 * The t-norm of two truth values.
 *
 * @param kind - Which t-norm.
 * @param a - A truth value in [0, 1].
 * @param b - Another.
 * @returns `min(a, b)`, `a b` or `max(0, a + b - 1)`.
 * @example
 * ```ts
 * import { tNorm } from 'omai-engine/logic'
 * [tNorm('min', 0.6, 0.7), tNorm('product', 0.6, 0.7), tNorm('lukasiewicz', 0.6, 0.7)].map((v) => Math.round(v * 1e9) / 1e9) // => [0.6, 0.42, 0.3]
 * ```
 */
export function tNorm(kind: TNorm, a: number, b: number): number {
  return kind === 'min' ? Math.min(a, b) : kind === 'product' ? a * b : Math.max(0, a + b - 1)
}

/**
 * The t-conorm dual to a t-norm: `max(a, b)`, `a + b - a b` or `min(1, a + b)`.
 *
 * @param kind - Which family.
 * @param a - A truth value in [0, 1].
 * @param b - Another.
 * @returns The fuzzy or.
 * @example
 * ```ts
 * import { tConorm } from 'omai-engine/logic'
 * [tConorm('min', 0.6, 0.7), tConorm('product', 0.6, 0.7), tConorm('lukasiewicz', 0.6, 0.7)].map((v) => Math.round(v * 1e9) / 1e9) // => [0.7, 0.88, 1]
 * ```
 */
export function tConorm(kind: TNorm, a: number, b: number): number {
  return kind === 'min' ? Math.max(a, b) : kind === 'product' ? a + b - a * b : Math.min(1, a + b)
}

/** A linguistic variable: its universe of discourse (the grid defuzzification works on) and its named fuzzy sets. */
export interface FuzzyVariable {
  readonly universe: readonly number[]
  readonly sets: Readonly<Record<string, Membership>>
}

/**
 * The equally spaced points of a universe.
 *
 * @param lo - First point.
 * @param hi - Last point.
 * @param n - The number of points (default 1001).
 * @returns The grid.
 * @example
 * ```ts
 * import { linspaceUniverse } from 'omai-engine/logic'
 * linspaceUniverse(0, 1, 5) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export function linspaceUniverse(lo: number, hi: number, n = 1001): number[] {
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

/** A fuzzy rule: if the (variable, set) conditions hold, joined by `and` or `or`, the output variable is the `consequent` set. */
export interface FuzzyRule {
  readonly conditions: readonly (readonly [variable: string, set: string])[]
  readonly connective: 'and' | 'or'
  readonly consequent: string
}

/** The defuzzifiers: centre of area, the point that halves the area, and the mean, smallest and largest of the maxima. */
export type Defuzzifier = 'centroid' | 'bisector' | 'mom' | 'som' | 'lom'

/**
 * Defuzzify an aggregated membership curve sampled on a grid. The centroid is the centre of area of the piecewise-linear
 * curve through the samples.
 *
 * @param universe - The grid points, ascending.
 * @param membership - The membership at each point.
 * @param method - The defuzzifier.
 * @returns The crisp value (the grid's midpoint for an all-zero curve).
 * @example
 * ```ts
 * import { defuzzify } from 'omai-engine/logic'
 * defuzzify([0, 1, 2, 3, 4], [0, 0.5, 1, 0.5, 0], 'centroid') // => 2
 * ```
 */
export function defuzzify(
  universe: readonly number[],
  membership: readonly number[],
  method: Defuzzifier = 'centroid',
): number {
  const n = universe.length
  const mid = ((universe[0] ?? 0) + (universe[n - 1] ?? 0)) / 2
  const total = membership.reduce((s, v) => s + v, 0)
  if (total === 0) return mid
  if (method === 'centroid') {
    // The centre of area of the piecewise-linear curve through the samples, exactly (as scikit-fuzzy computes it).
    let moment = 0
    let area = 0
    for (let i = 0; i + 1 < n; i++) {
      const x1 = universe[i] as number
      const dx = (universe[i + 1] as number) - x1
      const y1 = membership[i] ?? 0
      const y2 = membership[i + 1] ?? 0
      area += (dx * (y1 + y2)) / 2
      moment += dx * (x1 * y1 + (x1 * (y2 - y1)) / 2 + (dx * y1) / 2 + (dx * (y2 - y1)) / 3)
    }
    return area === 0 ? mid : moment / area
  }
  if (method === 'bisector') {
    let acc = 0
    for (let i = 0; i < n; i++) {
      acc += membership[i] ?? 0
      if (acc >= total / 2) return universe[i] as number
    }
    return mid
  }
  const top = maxOf(membership)
  const at = universe.filter((_, i) => Math.abs((membership[i] ?? 0) - top) < 1e-12)
  return method === 'som'
    ? (at[0] as number)
    : method === 'lom'
      ? (at[at.length - 1] as number)
      : at.reduce((s, x) => s + x, 0) / at.length
}

/** Options of fuzzy inference. */
export interface FuzzyOptions {
  /** The t-norm for `and` and the dual t-conorm for `or` (default `min`). */
  readonly norm?: TNorm
  /** The defuzzifier (default `centroid`). */
  readonly defuzzifier?: Defuzzifier
}

/**
 * Mamdani inference: the firing strength of each rule is the t-norm (`and`) or t-conorm (`or`) of the input memberships,
 * its output set is clipped at that strength, the clipped sets are aggregated by maximum and defuzzified.
 *
 * @param inputs - The input variables.
 * @param output - The output variable.
 * @param rules - The rule base.
 * @param values - The crisp input values by variable name.
 * @param options - The t-norm and the defuzzifier.
 * @returns The crisp output and each rule's firing strength.
 * @throws OmaiError with code `value` for a rule that names an unknown variable or set.
 * @example
 * ```ts
 * import { linspaceUniverse, mamdani, triangular } from 'omai-engine/logic'
 * const x = { universe: linspaceUniverse(0, 10, 11), sets: { high: triangular(5, 10, 10) } }
 * const y = { universe: linspaceUniverse(0, 10, 101), sets: { big: triangular(0, 5, 10) } }
 * Math.round(mamdani({ x }, y, [{ conditions: [['x', 'high']], connective: 'and', consequent: 'big' }], { x: 7.5 }).output * 1e9) / 1e9 // => 5
 * ```
 */
export function mamdani(
  inputs: Readonly<Record<string, FuzzyVariable>>,
  output: FuzzyVariable,
  rules: readonly FuzzyRule[],
  values: Readonly<Record<string, number>>,
  options: FuzzyOptions = {},
): { readonly output: number; readonly strengths: readonly number[]; readonly curve: readonly number[] } {
  const norm = options.norm ?? 'min'
  const strengths = rules.map((r) => {
    const degrees = r.conditions.map(([v, s]) => {
      const set = inputs[v]?.sets[s]
      if (set === undefined) throw new OmaiError('value', `unknown fuzzy set ${v}.${s}`)
      return set(values[v] ?? 0)
    })
    return degrees.reduce(
      (acc, d) => (r.connective === 'and' ? tNorm(norm, acc, d) : tConorm(norm, acc, d)),
      r.connective === 'and' ? 1 : 0,
    )
  })
  const curve = output.universe.map((x) =>
    rules.reduce((acc, r, i) => {
      const set = output.sets[r.consequent]
      if (set === undefined) throw new OmaiError('value', `unknown output set ${r.consequent}`)
      return Math.max(acc, tNorm(norm === 'product' ? 'product' : 'min', strengths[i] as number, set(x)))
    }, 0),
  )
  return { output: defuzzify(output.universe, curve, options.defuzzifier ?? 'centroid'), strengths, curve }
}

/** A Sugeno rule: if the conditions hold, the output is the constant, or the linear function `bias + sum coefficient * input`. */
export interface SugenoRule {
  readonly conditions: readonly (readonly [variable: string, set: string])[]
  readonly connective: 'and' | 'or'
  readonly constant?: number
  readonly linear?: { readonly bias: number; readonly coefficients: Readonly<Record<string, number>> }
}

/**
 * Sugeno (Takagi-Sugeno) inference: the weighted average of the rules' crisp outputs (constants or linear functions of
 * the inputs) by their firing strengths.
 *
 * @param inputs - The input variables.
 * @param rules - The rule base.
 * @param values - The crisp input values.
 * @param norm - The t-norm for `and` (default `min`).
 * @returns The output and each rule's firing strength.
 * @throws OmaiError with code `value` for an unknown set or a rule with no output.
 * @example
 * ```ts
 * import { linspaceUniverse, sugeno, triangular } from 'omai-engine/logic'
 * const x = { universe: linspaceUniverse(0, 10, 11), sets: { low: triangular(0, 0, 10), high: triangular(0, 10, 10) } }
 * const rules = [
 *   { conditions: [['x', 'low']], connective: 'and', constant: 0 },
 *   { conditions: [['x', 'high']], connective: 'and', constant: 10 },
 * ]
 * sugeno({ x }, rules, { x: 2.5 }).output // => 2.5
 * ```
 */
export function sugeno(
  inputs: Readonly<Record<string, FuzzyVariable>>,
  rules: readonly SugenoRule[],
  values: Readonly<Record<string, number>>,
  norm: TNorm = 'min',
): { readonly output: number; readonly strengths: readonly number[] } {
  const strengths = rules.map((r) => {
    const degrees = r.conditions.map(([v, s]) => {
      const set = inputs[v]?.sets[s]
      if (set === undefined) throw new OmaiError('value', `unknown fuzzy set ${v}.${s}`)
      return set(values[v] ?? 0)
    })
    return degrees.reduce(
      (acc, d) => (r.connective === 'and' ? tNorm(norm, acc, d) : tConorm(norm, acc, d)),
      r.connective === 'and' ? 1 : 0,
    )
  })
  let num = 0
  let den = 0
  rules.forEach((r, i) => {
    const z =
      r.constant ??
      (r.linear === undefined
        ? undefined
        : r.linear.bias + Object.entries(r.linear.coefficients).reduce((s, [k, c]) => s + c * (values[k] ?? 0), 0))
    if (z === undefined) throw new OmaiError('value', 'a Sugeno rule needs a constant or a linear output')
    num += (strengths[i] as number) * z
    den += strengths[i] as number
  })
  return { output: den === 0 ? 0 : num / den, strengths }
}

/**
 * Possibility and necessity of an event from a possibility distribution: `Pi(A) = max over A of pi`, `N(A) = 1 -
 * Pi(not A)`.
 *
 * @param distribution - Possibility of each state, with a maximum of 1.
 * @param event - The states of the event.
 * @returns The possibility and the necessity.
 * @throws OmaiError with code `value` for a distribution that is not normalised.
 * @example
 * ```ts
 * import { possibilityMeasures } from 'omai-engine/logic'
 * possibilityMeasures({ a: 1, b: 0.7, c: 0.2 }, ['a', 'b']) // => { possibility: 1, necessity: 0.8 }
 * ```
 */
export function possibilityMeasures(
  distribution: Readonly<Record<string, number>>,
  event: readonly string[],
): { readonly possibility: number; readonly necessity: number } {
  if (maxOf(Object.values(distribution)) !== 1) throw new OmaiError('value', 'a possibility distribution has maximum 1')
  const inside = new Set(event)
  const pi = Math.max(
    0,
    maxOf(
      Object.entries(distribution)
        .filter(([k]) => inside.has(k))
        .map(([, v]) => v),
    ),
  )
  const outside = Math.max(
    0,
    maxOf(
      Object.entries(distribution)
        .filter(([k]) => !inside.has(k))
        .map(([, v]) => v),
    ),
  )
  return { possibility: pi, necessity: 1 - outside }
}
