/**
 * Probabilistic logic programs in the style of ProbLog (De Raedt et al. 2007): a program of independent probabilistic
 * facts `p::fact` and function-free definite rules defines a distribution over possible worlds (each probabilistic fact
 * in or out), and the success probability of a query is the probability of the worlds whose least model contains it.
 * The program is grounded by forward chaining, each atom gets the set of proofs (sets of probabilistic facts that
 * entail it, found to a fixed point so recursion and cycles are handled), and the probability of the disjunction of the
 * proofs is computed exactly by weighted model counting: Shannon expansion on the facts with memoisation. Evidence
 * (atoms observed true or false) gives conditional probabilities, and the facts' probabilities are learned from observed
 * interpretations by EM, one iteration per step.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { parseTerm } from './parse.ts'
import { type Term, termToString } from './terms.ts'
import { applySubstitution, matchTerm, type Substitution } from './unify.ts'

/** A probabilistic logic program. */
export interface ProbabilisticProgram {
  /** Probabilistic facts: `[probability, ground atom]`. */
  readonly facts: readonly (readonly [probability: number, atom: string])[]
  /** Certain facts. */
  readonly certain: readonly string[]
  /** Definite rules `head :- body1, body2` (function-free, every head variable in the body). */
  readonly rules: readonly string[]
}

/** A grounded program: the atoms, the probability of each probabilistic fact and the ground rule instances. */
export interface GroundProgram {
  readonly atoms: readonly string[]
  readonly probabilities: readonly number[]
  /** Names of the probabilistic facts, by index. */
  readonly factNames: readonly string[]
  /** Ground rule instances `[head, body atoms]`. */
  readonly clauses: readonly (readonly [head: string, body: readonly string[]])[]
}

function ruleParts(text: string): { head: Term; body: Term[] } {
  const t = parseTerm(text)
  if (t.kind === 'compound' && t.functor === ':-' && t.args.length === 2) {
    const body: Term[] = []
    const walk = (b: Term): void => {
      if (b.kind === 'compound' && b.functor === ',' && b.args.length === 2) {
        walk(b.args[0] as Term)
        walk(b.args[1] as Term)
      } else body.push(b)
    }
    walk(t.args[1] as Term)
    return { head: t.args[0] as Term, body }
  }
  return { head: t, body: [] }
}

/**
 * Ground a program by forward chaining over the atoms that can be derived at all (ignoring probabilities).
 *
 * @param program - The program.
 * @returns The ground program.
 * @throws MdmlError with code `value` for a probability outside [0, 1], a non-ground fact, or a head variable not in the body.
 * @example
 * ```ts
 * import { groundProbabilistic } from 'mdml-engine/logic'
 * groundProbabilistic({ facts: [[0.5, 'edge(a,b)']], certain: [], rules: ['path(X,Y) :- edge(X,Y)'] }).atoms // => ['edge(a,b)', 'path(a,b)']
 * ```
 */
export function groundProbabilistic(program: ProbabilisticProgram): GroundProgram {
  for (const [p] of program.facts)
    if (!(p >= 0 && p <= 1)) throw new MdmlError('value', `probability ${p} is not in [0, 1]`)
  const known = new Map<string, Term>()
  for (const [, a] of program.facts) known.set(termToString(parseTerm(a)), parseTerm(a))
  for (const a of program.certain) known.set(termToString(parseTerm(a)), parseTerm(a))
  const rules = program.rules.map(ruleParts)
  for (const r of rules) {
    const bodyVars = new Set(r.body.flatMap((b) => termToString(b).match(/[A-Z_][A-Za-z0-9_]*/g) ?? []))
    for (const v of termToString(r.head).match(/[A-Z_][A-Za-z0-9_]*/g) ?? [])
      if (!bodyVars.has(v)) throw new MdmlError('value', `rule head variable ${v} is not in the body`)
  }
  const clauses = new Map<string, readonly [string, readonly string[]]>()
  for (let changed = true; changed; ) {
    changed = false
    for (const r of rules) {
      let substs: Substitution[] = [new Map()]
      for (const b of r.body) {
        const next: Substitution[] = []
        for (const s of substs)
          for (const fact of known.values()) {
            const m = matchTerm(applySubstitution(b, s), fact, s)
            if (m !== null) next.push(m)
          }
        substs = next
      }
      for (const s of substs) {
        const head = termToString(applySubstitution(r.head, s))
        const body = r.body.map((b) => termToString(applySubstitution(b, s)))
        const key = `${head}<-${body.join(',')}`
        if (!clauses.has(key)) {
          clauses.set(key, [head, body])
          changed = true
        }
        if (!known.has(head)) {
          known.set(head, applySubstitution(r.head, s))
          changed = true
        }
      }
    }
  }
  return {
    atoms: [...known.keys()],
    probabilities: program.facts.map(([p]) => p),
    factNames: program.facts.map(([, a]) => termToString(parseTerm(a))),
    clauses: [...clauses.values()],
  }
}

/** A proof: the probabilistic facts (by index) whose joint presence entails an atom. */
type ProofSet = readonly number[]

const union = (a: ProofSet, b: ProofSet): ProofSet => [...new Set([...a, ...b])].sort((x, y) => x - y)
const subsetOf = (a: ProofSet, b: ProofSet): boolean => a.every((x) => b.includes(x))

/** Remove proofs that contain another proof (absorption), leaving the minimal ones. */
function minimal(proofs: readonly ProofSet[]): ProofSet[] {
  const unique = [...new Map(proofs.map((p) => [p.join(','), p])).values()].sort((a, b) => a.length - b.length)
  const out: ProofSet[] = []
  for (const p of unique) if (!out.some((q) => subsetOf(q, p))) out.push(p)
  return out
}

/**
 * The minimal proofs of every atom of a ground program, as sets of probabilistic facts, computed to a fixed point (so
 * cyclic definitions such as reachability on a cyclic graph are handled and no proof uses a fact twice).
 *
 * @param g - The ground program.
 * @param certainAtoms - The atoms that hold in every world.
 * @returns The proofs by atom.
 * @example
 * ```ts
 * import { groundProbabilistic, proofSets } from 'mdml-engine/logic'
 * const g = groundProbabilistic({ facts: [[0.5, 'a'], [0.5, 'b']], certain: [], rules: ['c :- a', 'c :- b'] })
 * proofSets(g, [])['c'] // => [[0], [1]]
 * ```
 */
export function proofSets(g: GroundProgram, certainAtoms: readonly string[]): Record<string, ProofSet[]> {
  const proofs = new Map<string, ProofSet[]>()
  for (const a of certainAtoms) proofs.set(a, [[]])
  for (const [i, name] of g.factNames.entries()) proofs.set(name, minimal([...(proofs.get(name) ?? []), [i]]))
  for (let changed = true; changed; ) {
    changed = false
    for (const [head, body] of g.clauses) {
      let combos: ProofSet[] = [[]]
      for (const b of body) {
        const ps = proofs.get(b)
        if (ps === undefined) {
          combos = []
          break
        }
        combos = combos.flatMap((c) => ps.map((p) => union(c, p)))
        combos = minimal(combos)
      }
      const before = proofs.get(head) ?? []
      const merged = minimal([...before, ...combos])
      if (merged.length !== before.length || merged.some((m, i) => m.join(',') !== before[i]?.join(','))) {
        proofs.set(head, merged)
        changed = true
      }
    }
  }
  return Object.fromEntries(proofs)
}

/**
 * The probability that at least one proof holds, by weighted model counting: Shannon expansion on the most frequent
 * fact, `P = p P(proofs with the fact true) + (1 - p) P(proofs without the fact)`, memoised on the remaining proofs.
 *
 * @param proofs - The proofs (sets of fact indices).
 * @param probabilities - The probability of each fact.
 * @returns The probability of the disjunction.
 * @example
 * ```ts
 * import { disjunctionProbability } from 'mdml-engine/logic'
 * disjunctionProbability([[0], [1]], [0.5, 0.5]) // => 0.75
 * ```
 */
export function disjunctionProbability(proofs: readonly ProofSet[], probabilities: readonly number[]): number {
  const memo = new Map<string, number>()
  const go = (ps: readonly ProofSet[]): number => {
    if (ps.length === 0) return 0
    if (ps.some((p) => p.length === 0)) return 1
    const key = ps
      .map((p) => p.join('.'))
      .sort()
      .join('|')
    const hit = memo.get(key)
    if (hit !== undefined) return hit
    const count = new Map<number, number>()
    for (const p of ps) for (const f of p) count.set(f, (count.get(f) ?? 0) + 1)
    const [v] = [...count.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0] as [number, number]
    const pv = probabilities[v] ?? 0
    const whenTrue = minimal(ps.map((p) => p.filter((f) => f !== v)))
    const whenFalse = ps.filter((p) => !p.includes(v))
    const value = pv * go(whenTrue) + (1 - pv) * go(whenFalse)
    memo.set(key, value)
    return value
  }
  return go(minimal(proofs))
}

/** The result of a query: its probability and the proofs behind it. */
export interface QueryResult {
  readonly probability: number
  /** Each minimal proof as the probabilistic facts it uses, with their joint probability. */
  readonly proofs: readonly { readonly facts: readonly string[]; readonly probability: number }[]
}

/**
 * The success probability of a query atom, with its proofs: each proof is a set of probabilistic facts that together
 * entail the query, with the probability that all of them hold. The query's probability is that of their disjunction.
 *
 * @param program - The program.
 * @param query - The ground atom.
 * @returns The probability and the proofs (none, with probability 0, if the atom cannot be derived).
 * @throws MdmlError with code `value` for an invalid program.
 * @example
 * ```ts
 * import { successProbability } from 'mdml-engine/logic'
 * const program = {
 *   facts: [[0.8, 'edge(a,b)'], [0.6, 'edge(b,c)'], [0.3, 'edge(a,c)']],
 *   certain: [],
 *   rules: ['path(X,Y) :- edge(X,Y)', 'path(X,Y) :- edge(X,Z), path(Z,Y)'],
 * }
 * Math.round(successProbability(program, 'path(a,c)').probability * 1e9) / 1e9 // => 0.636
 * ```
 */
export function successProbability(program: ProbabilisticProgram, query: string): QueryResult {
  const g = groundProbabilistic(program)
  const proofs =
    proofSets(
      g,
      program.certain.map((c) => termToString(parseTerm(c))),
    )[termToString(parseTerm(query))] ?? []
  return {
    probability: disjunctionProbability(proofs, g.probabilities),
    proofs: proofs.map((p) => ({
      facts: p.map((i) => g.factNames[i] as string),
      probability: p.reduce((acc, i) => acc * (g.probabilities[i] ?? 0), 1),
    })),
  }
}

/**
 * The probability of a query given evidence atoms observed true or false: `P(q, e) / P(e)`, where joint events with
 * false evidence are expanded by inclusion and exclusion over the false atoms.
 *
 * @param program - The program.
 * @param query - The ground query atom.
 * @param evidence - Observed atoms with their truth values.
 * @returns The conditional probability.
 * @throws MdmlError with code `value` when the evidence has probability zero.
 * @example
 * ```ts
 * import { conditionalProbability } from 'mdml-engine/logic'
 * const alarm = { facts: [[0.1, 'burglary'], [0.2, 'earthquake']], certain: [], rules: ['alarm :- burglary', 'alarm :- earthquake'] }
 * conditionalProbability(alarm, 'burglary', [['alarm', true]]) // => 0.1 / 0.28
 * ```
 */
export function conditionalProbability(
  program: ProbabilisticProgram,
  query: string,
  evidence: readonly (readonly [atom: string, truth: boolean])[],
): number {
  const g = groundProbabilistic(program)
  const sets = proofSets(
    g,
    program.certain.map((c) => termToString(parseTerm(c))),
  )
  const proofsOf = (a: string): ProofSet[] => sets[termToString(parseTerm(a))] ?? []
  const conj = (atoms: readonly string[]): ProofSet[] =>
    atoms.reduce<ProofSet[]>((acc, a) => minimal(acc.flatMap((c) => proofsOf(a).map((p) => union(c, p)))), [[]])
  const joint = (atoms: readonly string[], negated: readonly string[]): number => {
    let total = 0
    for (let mask = 0; mask < 1 << negated.length; mask++) {
      const chosen = negated.filter((_, i) => (mask >> i) & 1)
      const sign = chosen.length % 2 === 0 ? 1 : -1
      total += sign * disjunctionProbability(conj([...atoms, ...chosen]), g.probabilities)
    }
    return total
  }
  const positive = evidence.filter(([, t]) => t).map(([a]) => a)
  const negative = evidence.filter(([, t]) => !t).map(([a]) => a)
  const pe = joint(positive, negative)
  if (!(pe > 0)) throw new MdmlError('value', 'the evidence has probability zero')
  return joint([...positive, query], negative) / pe
}

/** An observed interpretation: truth values of some atoms. */
export type Observation = readonly (readonly [atom: string, truth: boolean])[]

/** The state of EM for the facts' probabilities. */
export interface ProbLogEmState {
  readonly probabilities: readonly number[]
  readonly logLikelihood: number
  readonly iteration: number
  readonly change: number
}

const EM_SCHEMA: TraceSchema = channels({
  metrics: {
    logLikelihood: { description: 'log likelihood of the observations', direction: 'higher' },
    change: { description: 'largest change in a fact’s probability in this iteration' },
  },
  tensors: { parameters: { role: 'param', description: 'the probabilistic facts’ probabilities', shape: ['facts'] } },
})

/**
 * EM for the probabilities of the facts from observed interpretations, one iteration per step. The E-step finds, for
 * each observation, the posterior probability that each probabilistic fact is true given the observed atoms
 * (`p P(obs | fact) / P(obs)`); the M-step sets each fact's probability to the average posterior. The log likelihood of
 * the observations never decreases.
 *
 * @param program - The program with initial probabilities.
 * @param observations - Observed interpretations (atoms with truth values).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for an invalid program.
 * @example
 * ```ts
 * import { probLogEmAlgorithm } from 'mdml-engine/logic'
 * probLogEmAlgorithm({ facts: [[0.5, 'heads']], certain: [], rules: [] }, [[['heads', true]]]).name // => 'problog-em'
 * ```
 */
export function probLogEmAlgorithm(
  program: ProbabilisticProgram,
  observations: readonly Observation[],
): Algorithm<ProbLogEmState> {
  const g = groundProbabilistic(program)
  const sets = proofSets(
    g,
    program.certain.map((c) => termToString(parseTerm(c))),
  )
  const proofsOf = (a: string): ProofSet[] => sets[termToString(parseTerm(a))] ?? []
  const probOf = (obs: Observation, probabilities: readonly number[]): number => {
    const positive = obs.filter(([, t]) => t).map(([a]) => a)
    const negative = obs.filter(([, t]) => !t).map(([a]) => a)
    let total = 0
    for (let mask = 0; mask < 1 << negative.length; mask++) {
      const chosen = negative.filter((_, i) => (mask >> i) & 1)
      const conj = [...positive, ...chosen].reduce<ProofSet[]>(
        (acc, a) => minimal(acc.flatMap((c) => proofsOf(a).map((p) => union(c, p)))),
        [[]],
      )
      total += (chosen.length % 2 === 0 ? 1 : -1) * disjunctionProbability(conj, probabilities)
    }
    return total
  }
  return {
    name: 'problog-em',
    config: { facts: [...g.factNames], observations: observations.length },
    schema: EM_SCHEMA,
    init: () => ({
      probabilities: [...g.probabilities],
      logLikelihood: observations.reduce((s, o) => s + Math.log(probOf(o, g.probabilities)), 0),
      iteration: 0,
      change: Number.POSITIVE_INFINITY,
    }),
    step: (s) => {
      const next = s.probabilities.map((p, f) => {
        let sum = 0
        for (const o of observations) {
          const pe = probOf(o, s.probabilities)
          const clamped = s.probabilities.map((q, i) => (i === f ? 1 : q))
          sum += (p * probOf(o, clamped)) / pe
        }
        return sum / observations.length
      })
      return {
        probabilities: next,
        logLikelihood: observations.reduce((acc, o) => acc + Math.log(probOf(o, next)), 0),
        iteration: s.iteration + 1,
        change: Math.max(0, ...next.map((p, i) => Math.abs(p - (s.probabilities[i] ?? 0)))),
      }
    },
    done: (s) => (s.change < 1e-10 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { logLikelihood: s.logLikelihood, change: Number.isFinite(s.change) ? s.change : 0 },
      tensors: { parameters: { role: 'param', value: s.probabilities } },
    }),
    serialise: (s) => ({ ...s, change: Number.isFinite(s.change) ? s.change : -1 }) as unknown as Json,
    deserialise: (j) => {
      const s = j as unknown as ProbLogEmState
      return { ...s, change: s.change < 0 ? Number.POSITIVE_INFINITY : s.change }
    },
  }
}

/**
 * Learn the probabilistic facts' probabilities from observations by EM.
 *
 * @param program - The program with initial probabilities.
 * @param observations - Observed interpretations.
 * @param options - The most iterations (default 200) and tracing.
 * @returns The learned probabilities (in fact order) and the log likelihood.
 * @throws MdmlError with code `value` for an invalid program.
 * @example
 * ```ts
 * import { learnProbabilities } from 'mdml-engine/logic'
 * const obs = [true, true, true, false].map((t) => [['heads', t]])
 * learnProbabilities({ facts: [[0.5, 'heads']], certain: [], rules: [] }, obs).probabilities // => [0.75]
 * ```
 */
export function learnProbabilities(
  program: ProbabilisticProgram,
  observations: readonly Observation[],
  options: { readonly maxIterations?: number } & Partial<
    Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>
  > = {},
): { readonly probabilities: readonly number[]; readonly logLikelihood: number; readonly iterations: number } {
  const { maxIterations, ...run } = options
  const { state } = runAlgorithm(probLogEmAlgorithm(program, observations), { ...run, maxSteps: maxIterations ?? 200 })
  return { probabilities: state.probabilities, logLikelihood: state.logLikelihood, iterations: state.iteration }
}
