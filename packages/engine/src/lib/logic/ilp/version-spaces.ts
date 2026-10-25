/**
 * Version spaces for concept learning over attribute-value examples (Mitchell): conjunctive hypotheses and their
 * more-general-than order, Find-S, candidate elimination as a traced algorithm maintaining the specific and general
 * boundaries, enumeration of the version space, and classification of new instances by its vote.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../../trace/index.ts'

/**
 * A conjunctive hypothesis over attributes (Mitchell's representation): per attribute a required value, `'?'` (any
 * value) or `null` (no value: the hypothesis matches nothing).
 *
 * @concept version-spaces
 */
export type Hypothesis = readonly (string | null)[]

/** A training example: attribute values and whether it is positive. */
export interface LabelledExample {
  readonly values: readonly string[]
  readonly positive: boolean
}

/**
 * Whether a hypothesis classifies an instance as positive.
 *
 * @param h - The hypothesis.
 * @param x - The instance's attribute values.
 * @returns True when every constraint is met.
 * @example
 * ```ts
 * import { coversInstance } from 'mdml-engine/logic/ilp'
 * [coversInstance(['sunny', '?'], ['sunny', 'warm']), coversInstance(['sunny', null], ['sunny', 'warm'])] // => [true, false]
 * ```
 */
export function coversInstance(h: Hypothesis, x: readonly string[]): boolean {
  return h.every((c, i) => c === '?' || (c !== null && c === x[i]))
}

/**
 * The more-general-than-or-equal order: `h1 >= h2` when every instance `h2` covers is covered by `h1`.
 *
 * @param h1 - A hypothesis.
 * @param h2 - Another.
 * @returns True when `h1` is at least as general.
 * @example
 * ```ts
 * import { moreGeneral } from 'mdml-engine/logic/ilp'
 * [moreGeneral(['?', 'warm'], ['sunny', 'warm']), moreGeneral(['sunny', 'warm'], ['?', 'warm'])] // => [true, false]
 * ```
 */
export function moreGeneral(h1: Hypothesis, h2: Hypothesis): boolean {
  if (h2.some((c) => c === null)) return true
  return h1.every((c, i) => c === '?' || (c !== null && c === h2[i]))
}

function checkExamples(examples: readonly LabelledExample[], n: number, where: string): void {
  for (const e of examples) {
    if (e.values.length !== n) throw new MdmlError('value', `${where}: every example needs ${n} attribute values`)
  }
}

/**
 * Find-S: the most specific conjunctive hypothesis covering every positive example (negatives are ignored, as in
 * Mitchell's algorithm), starting from the empty hypothesis and generalising minimally.
 *
 * @param examples - The training examples.
 * @returns The hypothesis (all `null` when there are no positives).
 * @throws MdmlError with code `value` for examples of different lengths.
 * @concept version-spaces
 * @example
 * ```ts
 * import { findS } from 'mdml-engine/logic/ilp'
 * const examples = [
 *   { values: ['sunny', 'warm', 'normal'], positive: true },
 *   { values: ['sunny', 'warm', 'high'], positive: true },
 *   { values: ['rainy', 'cold', 'high'], positive: false },
 * ]
 * findS(examples) // => ['sunny', 'warm', '?']
 * ```
 */
export function findS(examples: readonly LabelledExample[]): (string | null)[] {
  const n = examples[0]?.values.length ?? 0
  checkExamples(examples, n, 'findS')
  const h: (string | null)[] = new Array(n).fill(null)
  for (const e of examples) {
    if (!e.positive) continue
    e.values.forEach((x, i) => {
      if (h[i] === null) h[i] = x
      else if (h[i] !== x) h[i] = '?'
    })
  }
  return h
}

/** The state of candidate elimination: the specific and general boundaries after each example. */
export interface VersionSpaceState {
  readonly specific: readonly Hypothesis[]
  readonly general: readonly Hypothesis[]
  readonly seen: number
  /** Whether the version space collapsed (no consistent hypothesis remains). */
  readonly collapsed: boolean
}

const VS_SCHEMA: TraceSchema = channels({
  metrics: {
    specific: { description: 'hypotheses on the specific boundary S' },
    general: { description: 'hypotheses on the general boundary G' },
  },
  messages: {
    example: {
      description: 'an example processed and the boundaries after it',
      payload: { example: 'values', positive: 'label', S: 'specific boundary', G: 'general boundary' },
    },
  },
})

const key = (h: Hypothesis): string => JSON.stringify(h)

function dedupe(hs: readonly Hypothesis[]): Hypothesis[] {
  const seen = new Set<string>()
  return hs.filter((h) => {
    const k = key(h)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}

/**
 * Candidate elimination (Mitchell) as a traced {@link Algorithm}: one training example per step. Positives generalise
 * the specific boundary minimally and remove general hypotheses that miss them; negatives specialise the general
 * boundary minimally (using each attribute's domain) and remove specific hypotheses that cover them; boundary
 * members not between the other boundary are discarded.
 *
 * @param examples - The training examples.
 * @param domains - The values of each attribute.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when examples and domains disagree in length.
 * @concept version-spaces
 * @example
 * ```ts
 * import { candidateEliminationAlgorithm } from 'mdml-engine/logic/ilp'
 * candidateEliminationAlgorithm([], [['a', 'b']]).name // => 'candidate-elimination'
 * ```
 */
export function candidateEliminationAlgorithm(
  examples: readonly LabelledExample[],
  domains: readonly (readonly string[])[],
): Algorithm<VersionSpaceState> {
  const n = domains.length
  checkExamples(examples, n, 'candidateElimination')
  return {
    name: 'candidate-elimination',
    config: { attributes: n, examples: examples.length },
    schema: VS_SCHEMA,
    init: () => ({
      specific: [new Array<string | null>(n).fill(null)],
      general: [new Array<string | null>(n).fill('?')],
      seen: 0,
      collapsed: false,
    }),
    step: (s) => {
      const e = examples[s.seen]
      if (e === undefined) return s
      const x = e.values
      let S = [...s.specific]
      let G = [...s.general]
      if (e.positive) {
        G = G.filter((g) => coversInstance(g, x))
        const next: Hypothesis[] = []
        for (const h of S) {
          if (coversInstance(h, x)) {
            next.push(h)
            continue
          }
          // The minimal generalisation of a conjunction that covers x is unique.
          const g = h.map((c, i) => (c === null ? (x[i] as string) : c === x[i] ? c : '?'))
          if (G.some((gg) => moreGeneral(gg, g))) next.push(g)
        }
        S = dedupe(next).filter((h) => !next.some((o) => key(o) !== key(h) && moreGeneral(h, o)))
      } else {
        S = S.filter((h) => !coversInstance(h, x))
        const next: Hypothesis[] = []
        for (const g of G) {
          if (!coversInstance(g, x)) {
            next.push(g)
            continue
          }
          for (let i = 0; i < n; i++) {
            if (g[i] !== '?') continue
            for (const v of domains[i] ?? []) {
              if (v === x[i]) continue
              const h = g.map((c, j) => (j === i ? v : c))
              if (S.some((ss) => moreGeneral(h, ss))) next.push(h)
            }
          }
        }
        const unique = dedupe(next)
        G = unique.filter((h) => !unique.some((o) => key(o) !== key(h) && moreGeneral(o, h)))
      }
      return { specific: S, general: G, seen: s.seen + 1, collapsed: S.length === 0 || G.length === 0 }
    },
    done: (s) => (s.seen >= examples.length || s.collapsed ? 'converged' : undefined),
    observe: (s) => {
      const e = examples[s.seen - 1]
      return {
        metrics: { specific: s.specific.length, general: s.general.length },
        messages:
          e === undefined
            ? []
            : [
                {
                  kind: 'example',
                  payload: {
                    example: [...e.values],
                    positive: e.positive,
                    S: s.specific,
                    G: s.general,
                  } as unknown as Json,
                },
              ],
      }
    },
  }
}

/** The boundaries of a version space. */
export interface VersionSpace {
  readonly specific: readonly Hypothesis[]
  readonly general: readonly Hypothesis[]
  readonly domains: readonly (readonly string[])[]
  readonly collapsed: boolean
}

/**
 * Run candidate elimination over the examples and return the version space's boundaries.
 *
 * @param examples - The training examples.
 * @param domains - The values of each attribute.
 * @returns The specific and general boundaries.
 * @throws MdmlError with code `value` when examples and domains disagree in length.
 * @concept version-spaces
 * @example
 * ```ts
 * import { candidateElimination } from 'mdml-engine/logic/ilp'
 * // Mitchell's EnjoySport (table 2.1).
 * const domains = [['sunny', 'cloudy', 'rainy'], ['warm', 'cold'], ['normal', 'high'], ['strong', 'weak'], ['warm', 'cool'], ['same', 'change']]
 * const vs = candidateElimination([
 *   { values: ['sunny', 'warm', 'normal', 'strong', 'warm', 'same'], positive: true },
 *   { values: ['sunny', 'warm', 'high', 'strong', 'warm', 'same'], positive: true },
 *   { values: ['rainy', 'cold', 'high', 'strong', 'warm', 'change'], positive: false },
 *   { values: ['sunny', 'warm', 'high', 'strong', 'cool', 'change'], positive: true },
 * ], domains)
 * vs.specific // => [['sunny', 'warm', '?', 'strong', '?', '?']]
 * vs.general // => [['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?']]
 * ```
 */
export function candidateElimination(
  examples: readonly LabelledExample[],
  domains: readonly (readonly string[])[],
): VersionSpace {
  const { state } = runAlgorithm(candidateEliminationAlgorithm(examples, domains), { maxSteps: examples.length + 1 })
  return { specific: state.specific, general: state.general, domains, collapsed: state.collapsed }
}

/**
 * Every hypothesis of a version space: those at least as general as some member of S and at most as general as some
 * member of G (enumerated over the attribute domains; exponential in the number of attributes).
 *
 * @param vs - The version space.
 * @returns The hypotheses.
 * @example
 * ```ts
 * import { versionSpaceMembers } from 'mdml-engine/logic/ilp'
 * versionSpaceMembers({ specific: [['a', 'x']], general: [['?', '?']], domains: [['a', 'b'], ['x', 'y']], collapsed: false }).length // => 4
 * ```
 */
export function versionSpaceMembers(vs: VersionSpace): Hypothesis[] {
  if (vs.collapsed) return []
  const out: Hypothesis[] = []
  const options = vs.domains.map((d) => ['?', ...d])
  const total = options.reduce((p, o) => p * o.length, 1)
  for (let m = 0; m < total; m++) {
    let k = m
    const h = options.map((o) => {
      const c = o[k % o.length] as string
      k = Math.floor(k / o.length)
      return c
    })
    if (vs.specific.some((s) => moreGeneral(h, s)) && vs.general.some((g) => moreGeneral(g, h))) out.push(h)
  }
  return out
}

/** The vote of a version space on an instance. */
export interface VersionSpaceVote {
  /** `positive` or `negative` when every member agrees, else `unknown`. */
  readonly label: 'positive' | 'negative' | 'unknown'
  readonly positive: number
  readonly negative: number
}

/**
 * Classify an instance by the version space: unanimous when all of S covers it (then every member does) or none of G
 * does (then no member does); otherwise the members vote.
 *
 * @param vs - The version space.
 * @param x - The instance.
 * @returns The label and the vote counts.
 * @example
 * ```ts
 * import { classifyByVersionSpace } from 'mdml-engine/logic/ilp'
 * const vs = { specific: [['a', 'x']], general: [['a', '?'], ['?', 'x']], domains: [['a', 'b'], ['x', 'y']], collapsed: false }
 * [classifyByVersionSpace(vs, ['a', 'x']).label, classifyByVersionSpace(vs, ['b', 'y']).label, classifyByVersionSpace(vs, ['a', 'y'])] // => ['positive', 'negative', { label: 'unknown', positive: 1, negative: 2 }]
 * ```
 */
export function classifyByVersionSpace(vs: VersionSpace, x: readonly string[]): VersionSpaceVote {
  const members = versionSpaceMembers(vs)
  const positive = members.filter((h) => coversInstance(h, x)).length
  const negative = members.length - positive
  if (members.length > 0 && negative === 0) return { label: 'positive', positive, negative }
  if (positive === 0) return { label: 'negative', positive, negative }
  return { label: 'unknown', positive, negative }
}
