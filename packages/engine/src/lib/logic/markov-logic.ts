/**
 * Markov logic networks (Richardson and Domingos 2006): weighted first-order formulas that, over a finite domain of
 * constants, define a Markov network with one binary variable per ground atom and one feature per grounding of each
 * formula, `P(x) = exp(sum_i w_i n_i(x)) / Z`, where `n_i(x)` counts the true groundings of formula `i`. This file
 * grounds a network (free variables are universally quantified, quantifiers are expanded over the domain, equality is
 * decided), computes the exact distribution by enumeration for tiny domains (the truth the samplers are tested
 * against), and infers by Gibbs sampling, by MC-SAT (slice sampling whose slice is sampled near-uniformly by SampleSAT,
 * or exactly for tiny slices), and by MAP through weighted MaxSAT. Samplers are stepped algorithms with the marginals
 * and the query probabilities on the tape. Weight learning is in `mln-learning.ts`.
 */

import { type Json, MdmlError, sigmoidOf } from '../base/index.ts'
import { categorical, type Key, uniformStream } from '../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  type Observation as TraceObservation,
  tensor,
} from '../trace/index.ts'
import { type FolFormula, parseFol } from './first-order.ts'
import { maxsat } from './maxsat.ts'
import { cnfClauses, type Formula, formulaVariables, tseitin } from './propositional.ts'
import { atom, compound, isPlainAtom, type Term, termEquals, termToString, termVariables } from './terms.ts'
import { applySubstitution, type Substitution } from './unify.ts'

// ------------------------------------------------------------------------------------------------ the network

/** A weighted first-order formula; an infinite weight makes it a hard constraint (a negative infinity forbids it). */
export interface WeightedFormula {
  /** The weight; `Infinity` for a hard formula. */
  readonly weight: number
  /** The formula in `parseFol` syntax: lower-case constants and predicates, upper-case free variables. */
  readonly formula: string
}

/** A predicate that appears in no formula but whose atoms belong to the world. */
export interface PredicateSpec {
  readonly name: string
  readonly arity: number
}

/** A Markov logic network: a finite domain of constants and weighted formulas over it. */
export interface MarkovLogicNetwork {
  /** The constants (lower-case identifiers). */
  readonly domain: readonly string[]
  readonly formulas: readonly WeightedFormula[]
  /** Predicates beyond those the formulas mention. */
  readonly predicates?: readonly PredicateSpec[]
}

/** A truth assignment to ground atoms by name; atoms not listed are false. */
export type World = Readonly<Record<string, boolean>>

/** One grounding of a first-order formula. */
export interface GroundFormula {
  /** The index of the first-order formula it instantiates. */
  readonly source: number
  readonly weight: number
  /** The propositional formula over ground atoms (named as `smokes(anna)`). */
  readonly formula: Formula
  /** The atom indices it mentions. */
  readonly atoms: readonly number[]
  /** Whether it holds in a world (an array of 0 and 1 by atom index). */
  readonly holds: (world: ArrayLike<number>) => boolean
}

/** A grounded network: the ground atoms, the ground formulas and the weights of the first-order formulas. */
export interface GroundMln {
  readonly network: MarkovLogicNetwork
  /** The ground atoms in a fixed order (predicates in order of appearance, arguments in domain order). */
  readonly atoms: readonly string[]
  readonly atomIndex: ReadonlyMap<string, number>
  readonly formulas: readonly GroundFormula[]
  /** The weight of each first-order formula. */
  readonly weights: readonly number[]
  /** For each atom, the ground formulas that mention it. */
  readonly occurrences: readonly (readonly number[])[]
}

const DOMAIN_SIZE_LIMIT = 4096

function atomKey(name: string, args: readonly string[]): string {
  return termToString(
    compound(
      name,
      args.map((a) => atom(a)),
    ),
  )
}

function freeVariables(f: FolFormula, bound: ReadonlySet<string>, out: Set<string>): void {
  switch (f.kind) {
    case 'atom':
      for (const v of termVariables(f.atom)) if (!bound.has(v.name)) out.add(v.name)
      return
    case 'const':
      return
    case 'not':
      freeVariables(f.arg, bound, out)
      return
    case 'and':
    case 'or':
      for (const a of f.args) freeVariables(a, bound, out)
      return
    case 'implies':
    case 'iff':
      freeVariables(f.left, bound, out)
      freeVariables(f.right, bound, out)
      return
    case 'forall':
    case 'exists':
      freeVariables(f.body, new Set([...bound, f.variable]), out)
  }
}

function collectPredicates(f: FolFormula, out: Map<string, number>): void {
  switch (f.kind) {
    case 'atom': {
      const t = f.atom
      const name = t.kind === 'compound' ? t.functor : t.kind === 'atom' ? t.name : null
      if (name === null) throw new MdmlError('value', 'a formula atom must be a predicate application')
      if (name === '=') return
      const arity = t.kind === 'compound' ? t.args.length : 0
      const known = out.get(name)
      if (known !== undefined && known !== arity)
        throw new MdmlError('value', `predicate ${name} is used with arities ${known} and ${arity}`)
      out.set(name, arity)
      return
    }
    case 'const':
      return
    case 'not':
      collectPredicates(f.arg, out)
      return
    case 'and':
    case 'or':
      for (const a of f.args) collectPredicates(a, out)
      return
    case 'implies':
    case 'iff':
      collectPredicates(f.left, out)
      collectPredicates(f.right, out)
      return
    case 'forall':
    case 'exists':
      collectPredicates(f.body, out)
  }
}

/** Ground a first-order formula under a substitution into a propositional formula over ground atom names. */
function groundFormula(f: FolFormula, s: Substitution, domain: readonly Term[]): Formula {
  switch (f.kind) {
    case 'atom': {
      const t = applySubstitution(f.atom, s)
      if (t.kind === 'compound' && t.functor === '=') {
        const [a, b] = t.args as [Term, Term]
        return { kind: 'const', value: termEquals(a, b) }
      }
      if (termVariables(t).length > 0) throw new MdmlError('value', `unbound variable in ${termToString(t)}`)
      return { kind: 'var', name: termToString(t) }
    }
    case 'const':
      return { kind: 'const', value: f.value }
    case 'not':
      return { kind: 'not', arg: groundFormula(f.arg, s, domain) }
    case 'and':
      return { kind: 'and', args: f.args.map((a) => groundFormula(a, s, domain)) }
    case 'or':
      return { kind: 'or', args: f.args.map((a) => groundFormula(a, s, domain)) }
    case 'implies':
      return { kind: 'implies', left: groundFormula(f.left, s, domain), right: groundFormula(f.right, s, domain) }
    case 'iff':
      return { kind: 'iff', left: groundFormula(f.left, s, domain), right: groundFormula(f.right, s, domain) }
    case 'forall':
    case 'exists': {
      const args = domain.map((c) => groundFormula(f.body, new Map([...s, [f.variable, c]]), domain))
      return f.kind === 'forall' ? { kind: 'and', args } : { kind: 'or', args }
    }
  }
}

/** Compile a propositional formula over atom indices into a function of a world. */
function compileFormula(f: Formula, index: ReadonlyMap<string, number>): (w: ArrayLike<number>) => boolean {
  switch (f.kind) {
    case 'var': {
      const i = index.get(f.name)
      if (i === undefined) throw new MdmlError('value', `unknown ground atom ${f.name}`)
      return (w) => (w[i] ?? 0) === 1
    }
    case 'const':
      return () => f.value
    case 'not': {
      const a = compileFormula(f.arg, index)
      return (w) => !a(w)
    }
    case 'and': {
      const as = f.args.map((a) => compileFormula(a, index))
      return (w) => as.every((a) => a(w))
    }
    case 'or': {
      const as = f.args.map((a) => compileFormula(a, index))
      return (w) => as.some((a) => a(w))
    }
    case 'implies': {
      const a = compileFormula(f.left, index)
      const b = compileFormula(f.right, index)
      return (w) => !a(w) || b(w)
    }
    case 'iff': {
      const a = compileFormula(f.left, index)
      const b = compileFormula(f.right, index)
      return (w) => a(w) === b(w)
    }
    case 'xor': {
      const a = compileFormula(f.left, index)
      const b = compileFormula(f.right, index)
      return (w) => a(w) !== b(w)
    }
  }
}

/**
 * Ground a Markov logic network over its domain: every predicate gets an atom per tuple of constants, every formula
 * one ground formula per assignment of its free variables (quantifiers expand over the domain, `=` is decided).
 *
 * @param network - The domain and weighted formulas.
 * @returns The ground network.
 * @throws MdmlError with code `value` for a domain constant that is not a plain lower-case identifier, a repeated
 *   constant, a predicate used with two arities or a weight that is NaN.
 * @example
 * ```ts
 * import { groundMln } from 'mdml-engine/logic'
 * const g = groundMln({
 *   domain: ['a', 'b'],
 *   formulas: [{ weight: 1.5, formula: 'smokes(X) -> cancer(X)' }],
 * })
 * [g.atoms.length, g.formulas.length] // => [4, 2]
 * ```
 */
export function groundMln(network: MarkovLogicNetwork): GroundMln {
  const { domain } = network
  if (domain.length > DOMAIN_SIZE_LIMIT) throw new MdmlError('value', 'groundMln: the domain is too large')
  if (new Set(domain).size !== domain.length) throw new MdmlError('value', 'groundMln: repeated constant in the domain')
  for (const c of domain)
    if (!isPlainAtom(c) || !/^[a-z]/.test(c))
      throw new MdmlError('value', `groundMln: constant '${c}' must be a lower-case identifier`)
  const parsed = network.formulas.map((wf) => {
    if (Number.isNaN(wf.weight)) throw new MdmlError('value', 'groundMln: a weight is NaN')
    return parseFol(wf.formula)
  })
  const predicates = new Map<string, number>()
  for (const p of parsed) collectPredicates(p, predicates)
  for (const p of network.predicates ?? []) {
    const known = predicates.get(p.name)
    if (known !== undefined && known !== p.arity)
      throw new MdmlError('value', `predicate ${p.name} is used with arities ${known} and ${p.arity}`)
    predicates.set(p.name, p.arity)
  }
  const atoms: string[] = []
  for (const [name, arity] of predicates) {
    const total = domain.length ** arity
    if (total > 1_000_000) throw new MdmlError('value', `groundMln: predicate ${name} has too many groundings`)
    for (let m = 0; m < total; m++) {
      const args: string[] = []
      let k = m
      for (let j = 0; j < arity; j++) {
        args.unshift(domain[k % domain.length] as string)
        k = Math.floor(k / domain.length)
      }
      atoms.push(atomKey(name, args))
    }
  }
  const atomIndex = new Map(atoms.map((a, i) => [a, i]))
  const consts = domain.map((d) => atom(d))
  const formulas: GroundFormula[] = []
  parsed.forEach((p, source) => {
    const weight = network.formulas[source]?.weight as number
    const free = new Set<string>()
    freeVariables(p, new Set(), free)
    const vars = [...free]
    const total = domain.length ** vars.length
    for (let m = 0; m < total; m++) {
      const s = new Map<string, Term>()
      let k = m
      for (const v of vars) {
        s.set(v, consts[k % domain.length] as Term)
        k = Math.floor(k / domain.length)
      }
      const formula = groundFormula(p, s, consts)
      const mentioned = formulaVariables(formula).map((n) => atomIndex.get(n) as number)
      if (mentioned.length === 0) continue
      formulas.push({ source, weight, formula, atoms: mentioned, holds: compileFormula(formula, atomIndex) })
    }
  })
  const occurrences: number[][] = atoms.map(() => [])
  formulas.forEach((g, gi) => {
    for (const a of g.atoms) occurrences[a]?.push(gi)
  })
  return {
    network,
    atoms,
    atomIndex,
    formulas,
    weights: network.formulas.map((f) => f.weight),
    occurrences,
  }
}

/**
 * The same ground network with other weights for its first-order formulas (no re-grounding).
 *
 * @param ground - The ground network.
 * @param weights - One weight per first-order formula.
 * @returns The reweighted network.
 * @throws MdmlError with code `value` when the weights do not match the formulas.
 * @example
 * ```ts
 * import { groundMln, withWeights } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: 1, formula: 'p(X)' }] })
 * withWeights(g, [2]).formulas[0]?.weight // => 2
 * ```
 */
export function withWeights(ground: GroundMln, weights: readonly number[]): GroundMln {
  if (weights.length !== ground.weights.length)
    throw new MdmlError('value', `withWeights: expected ${ground.weights.length} weights, got ${weights.length}`)
  return {
    ...ground,
    weights,
    formulas: ground.formulas.map((g) => ({ ...g, weight: weights[g.source] as number })),
  }
}

// ------------------------------------------------------------------------------------------------ worlds

/**
 * A world as an array of 0 and 1 by atom index.
 *
 * @param ground - The ground network.
 * @param world - Truth values by atom name (unlisted atoms are false).
 * @returns The array.
 * @throws MdmlError with code `value` for an atom the network does not have.
 * @example
 * ```ts
 * import { groundMln, worldVector } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 1, formula: 'p(X)' }] })
 * Array.from(worldVector(g, { 'p(b)': true })) // => [0, 1]
 * ```
 */
export function worldVector(ground: GroundMln, world: World): Uint8Array {
  const out = new Uint8Array(ground.atoms.length)
  for (const [name, value] of Object.entries(world)) {
    const i = ground.atomIndex.get(name)
    if (i === undefined) throw new MdmlError('value', `unknown ground atom ${name}`)
    out[i] = value ? 1 : 0
  }
  return out
}

/**
 * A world as a record of every ground atom's truth value.
 *
 * @param ground - The ground network.
 * @param vector - The array of 0 and 1 by atom index.
 * @returns The record.
 * @example
 * ```ts
 * import { groundMln, worldRecord } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: 1, formula: 'p(X)' }] })
 * worldRecord(g, Uint8Array.of(1)) // => { 'p(a)': true }
 * ```
 */
export function worldRecord(ground: GroundMln, vector: ArrayLike<number>): Record<string, boolean> {
  return Object.fromEntries(ground.atoms.map((a, i) => [a, (vector[i] ?? 0) === 1]))
}

/**
 * The number of true groundings of each first-order formula in a world.
 *
 * @param ground - The ground network.
 * @param world - The world as an array of 0 and 1.
 * @returns One count per first-order formula.
 * @example
 * ```ts
 * import { groundMln, mlnCounts, worldVector } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 1, formula: 'p(X)' }] })
 * mlnCounts(g, worldVector(g, { 'p(a)': true })) // => [1]
 * ```
 */
export function mlnCounts(ground: GroundMln, world: ArrayLike<number>): number[] {
  const counts = new Array<number>(ground.weights.length).fill(0)
  for (const g of ground.formulas) if (g.holds(world)) counts[g.source] = (counts[g.source] ?? 0) + 1
  return counts
}

/**
 * The unnormalised log probability of a world, `sum_i w_i n_i(x)`; negative infinity when a hard formula fails.
 *
 * @param ground - The ground network.
 * @param world - The world as an array of 0 and 1.
 * @returns The log weight.
 * @example
 * ```ts
 * import { groundMln, mlnLogWeight, worldVector } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 2, formula: 'p(X)' }] })
 * mlnLogWeight(g, worldVector(g, { 'p(a)': true, 'p(b)': true })) // => 4
 * ```
 */
export function mlnLogWeight(ground: GroundMln, world: ArrayLike<number>): number {
  let total = 0
  for (const g of ground.formulas) {
    const v = g.holds(world)
    if (g.weight === Number.POSITIVE_INFINITY) {
      if (!v) return Number.NEGATIVE_INFINITY
    } else if (g.weight === Number.NEGATIVE_INFINITY) {
      if (v) return Number.NEGATIVE_INFINITY
    } else if (v) total += g.weight
  }
  return total
}

/**
 * A ground (variable-free) query formula, compiled to a function of a world.
 *
 * @param ground - The ground network.
 * @param text - The query in `parseFol` syntax over ground atoms (`cancer(a) & smokes(b)`).
 * @returns The function.
 * @throws MdmlError with code `value` for a free variable or an unknown atom.
 * @example
 * ```ts
 * import { groundMln, groundQuery, worldVector } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 1, formula: 'p(X)' }] })
 * groundQuery(g, 'p(a) & -p(b)')(worldVector(g, { 'p(a)': true })) // => true
 * ```
 */
export function groundQuery(ground: GroundMln, text: string): (world: ArrayLike<number>) => boolean {
  const f = parseFol(text)
  const free = new Set<string>()
  freeVariables(f, new Set(), free)
  if (free.size > 0) throw new MdmlError('value', `groundQuery: free variable ${[...free][0]} in '${text}'`)
  return compileFormula(
    groundFormula(
      f,
      new Map(),
      ground.network.domain.map((d) => atom(d)),
    ),
    ground.atomIndex,
  )
}

// ------------------------------------------------------------------------------------------------ exact inference

/** The exact distribution of a ground network by enumeration. */
export interface MlnExact {
  /** The log partition function over the worlds consistent with the evidence. */
  readonly logPartition: number
  /** The marginal probability of each atom (evidence atoms are 0 or 1). */
  readonly marginals: Readonly<Record<string, number>>
  /** The probability of each query formula, given the evidence. */
  readonly queries: readonly number[]
  /** The expected count of true groundings of each first-order formula. */
  readonly expectedCounts: readonly number[]
  /** The worlds enumerated. */
  readonly worlds: number
}

const ENUMERATION_LIMIT = 22

/**
 * The exact distribution of a ground Markov logic network given evidence, by enumerating the worlds over the
 * unobserved atoms: the log partition function, every atom's marginal, query probabilities and the expected counts of
 * each formula's true groundings. Only for tiny domains; this is the truth the samplers are tested against.
 *
 * @param ground - The ground network.
 * @param options - Evidence atoms and ground query formulas.
 * @returns The exact quantities.
 * @throws MdmlError with code `value` for more than 22 unobserved atoms, an unknown atom, or evidence that is
 *   impossible under the hard formulas.
 * @example
 * ```ts
 * import { groundMln, mlnExact } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: Math.log(3), formula: 'p(a)' }] })
 * (mlnExact(g).marginals['p(a)'] ?? 0).toFixed(4) // => '0.7500'
 * ```
 */
export function mlnExact(
  ground: GroundMln,
  options: { readonly evidence?: World; readonly queries?: readonly string[] } = {},
): MlnExact {
  const n = ground.atoms.length
  const fixed = new Map<number, number>()
  for (const [name, v] of Object.entries(options.evidence ?? {})) {
    const i = ground.atomIndex.get(name)
    if (i === undefined) throw new MdmlError('value', `mlnExact: unknown ground atom ${name}`)
    fixed.set(i, v ? 1 : 0)
  }
  const free = [...Array(n).keys()].filter((i) => !fixed.has(i))
  if (free.length > ENUMERATION_LIMIT)
    throw new MdmlError('value', `mlnExact: ${free.length} unobserved atoms is too many to enumerate`)
  const queries = (options.queries ?? []).map((q) => groundQuery(ground, q))
  const world = new Uint8Array(n)
  for (const [i, v] of fixed) world[i] = v
  const total = 2 ** free.length
  const logWeights = new Float64Array(total)
  let max = Number.NEGATIVE_INFINITY
  for (let m = 0; m < total; m++) {
    for (let j = 0; j < free.length; j++) world[free[j] as number] = (m >> j) & 1
    const lw = mlnLogWeight(ground, world)
    logWeights[m] = lw
    if (lw > max) max = lw
  }
  if (max === Number.NEGATIVE_INFINITY) throw new MdmlError('value', 'mlnExact: the evidence has probability zero')
  let z = 0
  for (let m = 0; m < total; m++) z += Math.exp((logWeights[m] as number) - max)
  const marginals = new Float64Array(n)
  const qp = new Array<number>(queries.length).fill(0)
  const counts = new Array<number>(ground.weights.length).fill(0)
  for (let m = 0; m < total; m++) {
    const lw = logWeights[m] as number
    if (lw === Number.NEGATIVE_INFINITY) continue
    const p = Math.exp(lw - max) / z
    for (let j = 0; j < free.length; j++) world[free[j] as number] = (m >> j) & 1
    for (let i = 0; i < n; i++) if (world[i] === 1) marginals[i] = (marginals[i] ?? 0) + p
    queries.forEach((q, k) => {
      if (q(world)) qp[k] = (qp[k] ?? 0) + p
    })
    const c = mlnCounts(ground, world)
    c.forEach((v, k) => {
      counts[k] = (counts[k] ?? 0) + p * v
    })
  }
  return {
    logPartition: max + Math.log(z),
    marginals: Object.fromEntries(ground.atoms.map((a, i) => [a, marginals[i] as number])),
    queries: qp,
    expectedCounts: counts,
    worlds: total,
  }
}

/**
 * Exact samples of complete worlds from a ground network by enumeration (a generator whose truth is the network's
 * weights), for tiny domains: the data weight learning is tested against.
 *
 * @param ground - The ground network.
 * @param count - The number of worlds.
 * @param k - The random key.
 * @param options - Evidence atoms the worlds are conditioned on.
 * @returns The worlds as records of every atom's truth value.
 * @throws MdmlError with code `value` for more than 22 unobserved atoms or impossible evidence.
 * @example
 * ```ts
 * import { groundMln, mlnSampleWorlds } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: 30, formula: 'p(a)' }] })
 * mlnSampleWorlds(g, 2, key(0)) // => [{ 'p(a)': true }, { 'p(a)': true }]
 * ```
 */
export function mlnSampleWorlds(
  ground: GroundMln,
  count: number,
  k: Key,
  options: { readonly evidence?: World } = {},
): Record<string, boolean>[] {
  const n = ground.atoms.length
  const fixed = new Map<number, number>()
  for (const [name, v] of Object.entries(options.evidence ?? {})) {
    const i = ground.atomIndex.get(name)
    if (i === undefined) throw new MdmlError('value', `mlnSampleWorlds: unknown ground atom ${name}`)
    fixed.set(i, v ? 1 : 0)
  }
  const free = [...Array(n).keys()].filter((i) => !fixed.has(i))
  if (free.length > ENUMERATION_LIMIT)
    throw new MdmlError('value', `mlnSampleWorlds: ${free.length} unobserved atoms is too many to enumerate`)
  const world = new Uint8Array(n)
  for (const [i, v] of fixed) world[i] = v
  const logits = new Array<number>(2 ** free.length)
  for (let m = 0; m < logits.length; m++) {
    for (let j = 0; j < free.length; j++) world[free[j] as number] = (m >> j) & 1
    logits[m] = mlnLogWeight(ground, world)
  }
  if (!logits.some((l) => l > Number.NEGATIVE_INFINITY))
    throw new MdmlError('value', 'mlnSampleWorlds: the evidence has probability zero')
  const picks = categorical(k, logits, [count], { logits: true }).data
  return Array.from(picks, (m) => {
    for (let j = 0; j < free.length; j++) world[free[j] as number] = (m >> j) & 1
    return worldRecord(ground, world)
  })
}

// ------------------------------------------------------------------------------------------------ clause encodings

/**
 * Clauses over variables 1..atoms (the ground atoms) and further auxiliary variables, imposing or encoding ground
 * formulas, shared by MC-SAT and MAP.
 */
export class Encoder {
  /** The largest variable used so far (atoms first, then auxiliaries). */
  next: number
  constructor(readonly ground: GroundMln) {
    this.next = ground.atoms.length
  }

  private literal(name: string, positive: boolean): number {
    const i = this.ground.atomIndex.get(name) as number
    return positive ? i + 1 : -(i + 1)
  }

  /** The Tseitin clauses of a formula over the atoms and fresh auxiliaries; the last clause asserts the root. */
  private tseitinClauses(f: Formula): number[][] {
    const t = tseitin(f)
    const base = this.next
    const names = [...t.variables.keys()]
    const m = names.length
    const map = (v: number): number => {
      const a = Math.abs(v)
      const g = a <= m ? (this.ground.atomIndex.get(names[a - 1] as string) as number) + 1 : base + (a - m)
      return v < 0 ? -g : g
    }
    this.next = base + (t.cnf.numVars - m)
    return t.cnf.clauses.map((c) => c.map(map))
  }

  /** Clauses that hold exactly in the worlds (with the auxiliaries determined by them) where the formula holds. */
  constraint(f: Formula): number[][] {
    if (formulaVariables(f).length <= 6) {
      const cl = cnfClauses(f)
      if (cl.length <= 12) return cl.map((c) => c.map((l) => this.literal(l.name, l.positive)))
    }
    return this.tseitinClauses(f)
  }

  /**
   * A soft encoding: the definitions (hard clauses) and one clause whose satisfaction is the formula's truth, or the
   * constant `true` or `false`.
   */
  soft(f: Formula): { readonly definitions: number[][]; readonly clause: number[] | 'true' | 'false' } {
    if (formulaVariables(f).length <= 6) {
      const cl = cnfClauses(f)
      if (cl.length === 0) return { definitions: [], clause: 'true' }
      if (cl.length === 1) {
        const c = cl[0] as (typeof cl)[number]
        if (c.length === 0) return { definitions: [], clause: 'false' }
        return { definitions: [], clause: c.map((l) => this.literal(l.name, l.positive)) }
      }
    }
    const all = this.tseitinClauses(f)
    return { definitions: all.slice(0, -1), clause: all[all.length - 1] as number[] }
  }
}

// ------------------------------------------------------------------------------------------------ SampleSAT

/** Knobs of {@link sampleSat}. */
export interface SampleSatOptions {
  /** The probability of a simulated-annealing move rather than a WalkSAT move (default 0.5). */
  readonly annealing?: number
  /** The annealing temperature (default 0.5). */
  readonly temperature?: number
  /** The probability of a random rather than a greedy WalkSAT move (default 0.5). */
  readonly noise?: number
  /** Annealing moves taken after the first solution before the sample is read, per variable (default 10). */
  readonly mixing?: number
  /** Moves before giving up (default 1e6). */
  readonly maxFlips?: number
}

/** The paths {@link sampleSatAlgorithm} logs. */
export type SampleSatPath = 'sat/flips' | 'sat/unsat' | 'sat/move' | 'sat/sample'

/** What SampleSAT logs per flip: the flips so far, the clauses unsatisfied, the kind of move, and the model when it is read. */
export const sampleSatContract: Contract<unknown, SampleSatPath> = defineContract({
  'sat/flips': scalar({ description: 'flips made so far' }),
  'sat/unsat': scalar({ description: 'clauses unsatisfied after the flip', direction: 'lower' }),
  'sat/move': scalar({
    description: 'the kind of move: 1 a simulated-annealing proposal, 2 a WalkSAT move, 0 no flip',
  }),
  'sat/sample': tensor({
    role: 'output',
    shape: ['variables'],
    description: 'the model read (0 and 1 by variable), at the end',
  }),
})

/**
 * The state of {@link sampleSatAlgorithm}: the assignment, the satisfied-literal count of each clause and the unsatisfied
 * clauses (kept incrementally), the flips made, the flip at which the first model was reached, the coins drawn, and the
 * outcome once reached. Steps update the arrays in place; keep a copy of a state you want to look at later.
 */
export interface SampleSatState {
  readonly value: Uint8Array
  readonly trueCount: Int32Array
  readonly unsat: number[]
  readonly where: Int32Array
  flips: number
  foundAt: number
  /** Coins drawn from the key's counter-based stream. */
  coins: number
  /** `running`, `sampled` (the model is `value`) or `failed` (none found). */
  status: 'running' | 'sampled' | 'failed'
}

interface SampleSatMachine {
  readonly start: (uniform: () => number) => SampleSatState
  readonly flip: (s: SampleSatState, uniform: () => number) => number
  readonly settle: (s: SampleSatState) => void
}

/** The moves of SampleSAT over a clause set, shared by the algorithm and the function. */
function sampleSatMachine(
  clauses: readonly (readonly number[])[],
  numVars: number,
  options: SampleSatOptions,
): SampleSatMachine {
  const pAnneal = options.annealing ?? 0.5
  const temperature = options.temperature ?? 0.5
  const noise = options.noise ?? 0.5
  const mixing = options.mixing ?? 10
  const maxFlips = options.maxFlips ?? 1_000_000
  const m = clauses.length
  const occ: { c: number; positive: boolean }[][] = Array.from({ length: numVars }, () => [])
  clauses.forEach((c, ci) => {
    for (const l of c) occ[Math.abs(l) - 1]?.push({ c: ci, positive: l > 0 })
  })
  const used: number[] = []
  for (let v = 0; v < numVars; v++) if ((occ[v] as unknown[]).length > 0) used.push(v)
  const addUnsat = (s: SampleSatState, c: number): void => {
    s.where[c] = s.unsat.length
    s.unsat.push(c)
  }
  const removeUnsat = (s: SampleSatState, c: number): void => {
    const at = s.where[c] as number
    const last = s.unsat.pop() as number
    if (last !== c) {
      s.unsat[at] = last
      s.where[last] = at
    }
    s.where[c] = -1
  }
  const doFlip = (s: SampleSatState, v: number): void => {
    const was = s.value[v] === 1
    s.value[v] = was ? 0 : 1
    for (const { c, positive } of occ[v] as { c: number; positive: boolean }[]) {
      if (positive === was) {
        // The literal was true and is now false.
        s.trueCount[c] = (s.trueCount[c] as number) - 1
        if (s.trueCount[c] === 0) addUnsat(s, c)
      } else {
        s.trueCount[c] = (s.trueCount[c] as number) + 1
        if (s.trueCount[c] === 1) removeUnsat(s, c)
      }
    }
  }
  const delta = (s: SampleSatState, v: number): number => {
    const was = s.value[v] === 1
    let d = 0
    for (const { c, positive } of occ[v] as { c: number; positive: boolean }[]) {
      if (positive === was) {
        if (s.trueCount[c] === 1) d++
      } else if (s.trueCount[c] === 0) d--
    }
    return d
  }
  const breaks = (s: SampleSatState, v: number): number => {
    const was = s.value[v] === 1
    let b = 0
    for (const { c, positive } of occ[v] as { c: number; positive: boolean }[])
      if (positive === was && s.trueCount[c] === 1) b++
    return b
  }
  // The check at the top of each iteration: the first model starts the mixing, and enough mixing reads the sample.
  const settle = (s: SampleSatState): void => {
    if (s.status !== 'running') return
    if (s.unsat.length === 0) {
      if (s.foundAt < 0) s.foundAt = s.flips
      if (s.flips - s.foundAt >= mixing * used.length) s.status = 'sampled'
    }
    if (s.status === 'running' && s.flips >= maxFlips) s.status = s.unsat.length === 0 ? 'sampled' : 'failed'
  }
  return {
    start: (uniform) => {
      const s: SampleSatState = {
        value: new Uint8Array(numVars),
        trueCount: new Int32Array(m),
        unsat: [],
        where: new Int32Array(m).fill(-1),
        flips: 0,
        foundAt: -1,
        coins: 0,
        status: 'running',
      }
      for (let v = 0; v < numVars; v++) s.value[v] = uniform() < 0.5 ? 1 : 0
      if (clauses.some((c) => c.length === 0)) {
        s.status = 'failed'
        return s
      }
      clauses.forEach((c, ci) => {
        let t = 0
        for (const l of c) if ((s.value[Math.abs(l) - 1] === 1) === l > 0) t++
        s.trueCount[ci] = t
        if (t === 0) addUnsat(s, ci)
      })
      if (used.length === 0) s.status = s.unsat.length === 0 ? 'sampled' : 'failed'
      else settle(s)
      return s
    },
    // One flip: returns the kind of move (1 annealing, 2 WalkSAT).
    flip: (s, uniform) => {
      let kind: number
      if (s.unsat.length === 0 || uniform() < pAnneal) {
        kind = 1
        const v = used[Math.floor(uniform() * used.length)] as number
        const d = delta(s, v)
        if (d <= 0 || uniform() < Math.exp(-d / temperature)) doFlip(s, v)
      } else {
        kind = 2
        const c = s.unsat[Math.floor(uniform() * s.unsat.length)] as number
        const lits = clauses[c] as readonly number[]
        let pick: number
        if (uniform() < noise) pick = Math.abs(lits[Math.floor(uniform() * lits.length)] as number) - 1
        else {
          pick = Math.abs(lits[0] as number) - 1
          let best = Number.POSITIVE_INFINITY
          let ties = 0
          for (const l of lits) {
            const b = breaks(s, Math.abs(l) - 1)
            if (b < best) {
              best = b
              ties = 1
              pick = Math.abs(l) - 1
            } else if (b === best && uniform() * ++ties < 1) pick = Math.abs(l) - 1
          }
        }
        doFlip(s, pick)
      }
      s.flips += 1
      settle(s)
      return kind
    },
    settle,
  }
}

/**
 * SampleSAT (Wei, Erenrich and Selman 2004), a near-uniform sampler of the models of a CNF, as a stepped algorithm, one
 * flip per step: from a random assignment it mixes WalkSAT moves with simulated-annealing moves at a fixed temperature, and
 * once a model is found it keeps making annealing moves (which accept a move out of the solution cluster with Boltzmann
 * probability and walk back) for a number of flips before the model is read, so the sample is spread over the solution
 * space rather than the first solution reached. The coins are a counter-based stream of the key, so a stored state
 * resumes. It logs the flips, the clauses unsatisfied, the move kind and, at the end, the sample, and is done
 * (`converged`) when a model is read and `failed` when none was found in budget. The state serialises.
 *
 * @param clauses - The clauses over variables `1..numVars`.
 * @param numVars - The number of variables.
 * @param key - The key the coins come from.
 * @param options - The annealing probability, temperature, noise, mixing flips and flip budget.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { sampleSatAlgorithm } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(sampleSatAlgorithm([[1, 2], [-1, -2]], 2, key(0)), { maxSteps: 1000 })
 * [run.status, run.state.value[0] !== run.state.value[1]] // => ['converged', true]
 * ```
 */
export function sampleSatAlgorithm(
  clauses: readonly (readonly number[])[],
  numVars: number,
  key: Key,
  options: SampleSatOptions = {},
): Algorithm<SampleSatState, SampleSatPath> {
  const machine = sampleSatMachine(clauses, numVars, options)
  // A counter-based stream positioned at the coins drawn so far (kept across steps while the state is the same run).
  let cursor: { stream: () => number; at: number } | undefined
  const draw = (s: SampleSatState): (() => number) => {
    if (cursor === undefined || cursor.at !== s.coins)
      cursor = { stream: uniformStream(key, 256, s.coins), at: s.coins }
    const c = cursor
    return () => {
      c.at += 1
      s.coins += 1
      return c.stream()
    }
  }
  return {
    name: 'logic/sample-sat',
    config: { clauses: clauses.length, numVars, ...options } as Json,
    init: () => {
      let drawn = 0
      const stream = uniformStream(key, 256, 0)
      const s = machine.start(() => {
        drawn += 1
        return stream()
      })
      s.coins = drawn
      cursor = { stream, at: drawn }
      return s
    },
    step: (s, ctx) => {
      const move = machine.flip(s, draw(s))
      ctx.log('sat/flips', s.flips)
      ctx.log('sat/unsat', s.unsat.length)
      ctx.log('sat/move', move)
      if (s.status === 'sampled') ctx.log('sat/sample', s.value)
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : s.status === 'sampled' ? 'converged' : 'failed'),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SampleSatState,
  }
}

/**
 * SampleSAT (Wei, Erenrich and Selman 2004), a near-uniform sampler of the models of a CNF: from a random assignment
 * it mixes WalkSAT moves with simulated-annealing moves at a fixed temperature, and once a model is found it keeps
 * making annealing moves (which accept a move out of the solution cluster with Boltzmann probability and walk back)
 * for a number of flips before the model is read, so the sample is spread over the solution space rather than the
 * first solution reached. The same moves as {@link sampleSatAlgorithm}, drawing from the given source of uniforms.
 *
 * @param clauses - The clauses over variables `1..numVars`.
 * @param numVars - The number of variables.
 * @param uniform - A source of uniform draws in [0, 1).
 * @param options - The annealing probability, temperature, noise, mixing flips and flip budget.
 * @returns A model as an array of 0 and 1 by variable (index 0 is variable 1), or null when none was found in budget.
 * @example
 * ```ts
 * import { sampleSat } from 'mdml-engine/logic'
 * import { key, uniformStream } from 'mdml-engine/random'
 * const m = sampleSat([[1, 2], [-1, -2]], 2, uniformStream(key(0)))
 * m !== null && m[0] !== m[1] // => true
 * ```
 */
export function sampleSat(
  clauses: readonly (readonly number[])[],
  numVars: number,
  uniform: () => number,
  options: SampleSatOptions = {},
): Uint8Array | null {
  const machine = sampleSatMachine(clauses, numVars, options)
  const s = machine.start(uniform)
  while (s.status === 'running') machine.flip(s, uniform)
  return s.status === 'sampled' ? s.value : null
}

/** Uniform sampling of the models of a small CNF by enumeration (the exact slice sampler). */
function exactModel(
  clauses: readonly (readonly number[])[],
  numVars: number,
  uniform: () => number,
): Uint8Array | null {
  if (numVars > ENUMERATION_LIMIT) throw new MdmlError('value', 'MC-SAT exact slice: too many variables to enumerate')
  const pos = new Int32Array(clauses.length)
  const neg = new Int32Array(clauses.length)
  clauses.forEach((c, i) => {
    for (const l of c) {
      if (l > 0) pos[i] = (pos[i] as number) | (1 << (l - 1))
      else neg[i] = (neg[i] as number) | (1 << (-l - 1))
    }
  })
  const total = 2 ** numVars
  const ok = (mask: number): boolean => {
    for (let i = 0; i < clauses.length; i++)
      if (((mask & (pos[i] as number)) | (~mask & (neg[i] as number))) === 0) return false
    return true
  }
  let count = 0
  for (let mask = 0; mask < total; mask++) if (ok(mask)) count++
  if (count === 0) return null
  let pick = Math.floor(uniform() * count)
  for (let mask = 0; mask < total; mask++) {
    if (!ok(mask)) continue
    if (pick-- === 0) {
      const out = new Uint8Array(numVars)
      for (let v = 0; v < numVars; v++) out[v] = (mask >> v) & 1
      return out
    }
  }
  return null
}

// ------------------------------------------------------------------------------------------------ samplers

/** The paths of the samplers. */
export type MlnSamplerPath = 'mln/world' | 'mln/marginals' | 'mln/queries' | 'mln/energy' | 'mln/samples'

/** The log contract of {@link mlnGibbsAlgorithm} and {@link mlnMcSatAlgorithm}. */
export const mlnSamplerContract: Contract<MlnSamplerState, MlnSamplerPath> = defineContract({
  'mln/world': tensor({ role: 'state', shape: ['atoms'], description: 'the current world, 1 where the atom is true' }),
  'mln/marginals': tensor({
    role: 'output',
    shape: ['atoms'],
    description: 'the running estimate of each atom’s marginal probability (the current world before the first sample)',
  }),
  'mln/queries': tensor({
    role: 'output',
    shape: ['queries'],
    description: 'the running estimate of each query formula’s probability',
  }),
  'mln/energy': scalar({ description: 'the log weight of the current world', direction: 'higher' }),
  'mln/samples': scalar({ description: 'samples averaged so far, after burn-in' }),
})

/** The state of the samplers. */
export interface MlnSamplerState {
  /** The current world, 0 and 1 by atom. */
  readonly world: Uint8Array
  /** Sweeps (Gibbs) or iterations (MC-SAT) taken. */
  readonly iterations: number
  /** Samples averaged (those after burn-in). */
  readonly samples: number
  /** How often each atom was true among the samples. */
  readonly atomCounts: Float64Array
  /** How often each query formula was true among the samples. */
  readonly queryCounts: Float64Array
  /** The log weight of the world. */
  readonly energy: number
}

/** Options shared by the samplers. */
export interface MlnSamplerOptions {
  /** Atoms with known truth values. */
  readonly evidence?: World
  /** Ground formulas whose probability is estimated. */
  readonly queries?: readonly string[]
  /** Iterations discarded before averaging (default 100). */
  readonly burnIn?: number
}

/** Options of {@link mlnGibbsAlgorithm}. */
export interface MlnGibbsOptions extends MlnSamplerOptions {
  /** `systematic` visits the atoms in order, `random` picks them with replacement (default systematic). */
  readonly scan?: 'systematic' | 'random'
}

/** Options of {@link mlnMcSatAlgorithm}. */
export interface MlnMcSatOptions extends MlnSamplerOptions {
  /** How the slice is sampled: SampleSAT (near-uniform, any size) or by enumeration (exact, tiny slices; default SampleSAT). */
  readonly sampler?: 'samplesat' | 'exact'
  /** Knobs of SampleSAT. */
  readonly sampleSat?: SampleSatOptions
}

interface Prepared {
  readonly n: number
  readonly fixed: Int8Array
  readonly freeAtoms: readonly number[]
  readonly queries: readonly ((w: ArrayLike<number>) => boolean)[]
  readonly burnIn: number
}

function prepare(ground: GroundMln, options: MlnSamplerOptions, who: string): Prepared {
  const n = ground.atoms.length
  const fixed = new Int8Array(n).fill(-1)
  for (const [name, v] of Object.entries(options.evidence ?? {})) {
    const i = ground.atomIndex.get(name)
    if (i === undefined) throw new MdmlError('value', `${who}: unknown ground atom ${name}`)
    fixed[i] = v ? 1 : 0
  }
  const burnIn = options.burnIn ?? 100
  if (!(Number.isInteger(burnIn) && burnIn >= 0))
    throw new MdmlError('value', `${who}: burnIn must be a non-negative integer`)
  return {
    n,
    fixed,
    freeAtoms: [...Array(n).keys()].filter((i) => fixed[i] === -1),
    queries: (options.queries ?? []).map((q) => groundQuery(ground, q)),
    burnIn,
  }
}

const isHard = (w: number): boolean => w === Number.POSITIVE_INFINITY || w === Number.NEGATIVE_INFINITY

/** A first world: evidence fixed, the rest random, repaired to satisfy the hard formulas by SampleSAT. */
function initialWorld(ground: GroundMln, p: Prepared, key: Key, who: string): Uint8Array {
  const u = uniformStream(key)
  const world = new Uint8Array(p.n)
  for (let i = 0; i < p.n; i++) world[i] = p.fixed[i] === -1 ? (u() < 0.5 ? 1 : 0) : (p.fixed[i] as number)
  if (!ground.formulas.some((g) => isHard(g.weight))) return world
  const enc = new Encoder(ground)
  const clauses: number[][] = []
  for (const g of ground.formulas) {
    if (g.weight === Number.POSITIVE_INFINITY) clauses.push(...enc.constraint(g.formula))
    else if (g.weight === Number.NEGATIVE_INFINITY) clauses.push(...enc.constraint({ kind: 'not', arg: g.formula }))
  }
  for (let i = 0; i < p.n; i++) if (p.fixed[i] !== -1) clauses.push([p.fixed[i] === 1 ? i + 1 : -(i + 1)])
  const model = sampleSat(clauses, enc.next, u, { maxFlips: 2_000_000 })
  if (model === null) throw new MdmlError('value', `${who}: the hard formulas and the evidence are unsatisfiable`)
  for (let i = 0; i < p.n; i++) world[i] = model[i] as number
  return world
}

function freshState(ground: GroundMln, p: Prepared, world: Uint8Array): MlnSamplerState {
  return {
    world,
    iterations: 0,
    samples: 0,
    atomCounts: new Float64Array(p.n),
    queryCounts: new Float64Array(p.queries.length),
    energy: mlnLogWeight(ground, world),
  }
}

function accumulate(ground: GroundMln, p: Prepared, s: MlnSamplerState, world: Uint8Array): MlnSamplerState {
  const iterations = s.iterations + 1
  const energy = mlnLogWeight(ground, world)
  if (iterations <= p.burnIn) return { ...s, world, iterations, energy }
  const atomCounts = Float64Array.from(s.atomCounts)
  for (let i = 0; i < p.n; i++) atomCounts[i] = (atomCounts[i] as number) + (world[i] as number)
  const queryCounts = Float64Array.from(s.queryCounts)
  p.queries.forEach((q, k) => {
    if (q(world)) queryCounts[k] = (queryCounts[k] as number) + 1
  })
  return { world, iterations, samples: s.samples + 1, atomCounts, queryCounts, energy }
}

/** The estimate of each atom's marginal: the running frequency, or the current world before any sample. */
function marginalsOf(s: MlnSamplerState): Float64Array {
  return s.samples === 0 ? Float64Array.from(s.world) : s.atomCounts.map((c) => c / s.samples)
}

function observeSampler(s: MlnSamplerState): TraceObservation {
  return {
    metrics: { 'mln/energy': s.energy, 'mln/samples': s.samples },
    tensors: {
      'mln/world': { role: 'state', value: s.world },
      'mln/marginals': { role: 'output', value: marginalsOf(s) },
      'mln/queries': { role: 'output', value: s.queryCounts.map((c) => (s.samples === 0 ? 0 : c / s.samples)) },
    },
  }
}

/** The change in log weight, and in hard-formula violations, between an atom true and false. */
function conditional(ground: GroundMln, world: Uint8Array, a: number): number {
  let delta = 0
  let hard = 0
  const before = world[a] as number
  for (const gi of ground.occurrences[a] as readonly number[]) {
    const g = ground.formulas[gi] as GroundFormula
    world[a] = 1
    const v1 = g.holds(world)
    world[a] = 0
    const v0 = g.holds(world)
    if (v1 === v0) continue
    if (g.weight === Number.POSITIVE_INFINITY) hard += (v1 ? 0 : 1) - (v0 ? 0 : 1)
    else if (g.weight === Number.NEGATIVE_INFINITY) hard += (v1 ? 1 : 0) - (v0 ? 1 : 0)
    else delta += g.weight * ((v1 ? 1 : 0) - (v0 ? 1 : 0))
  }
  world[a] = before
  // hard > 0: more violations with the atom true, so it must be false.
  return hard > 0 ? Number.NEGATIVE_INFINITY : hard < 0 ? Number.POSITIVE_INFINITY : delta
}

/**
 * Gibbs sampling of a ground Markov logic network, one sweep per step: each unobserved atom is resampled from its
 * conditional given the atoms that share a ground formula with it (the Markov blanket), `P(a = 1 | rest) = sigmoidOf(sum
 * of the weights of the groundings it makes true minus those it makes false)`. After the burn-in each sweep adds to
 * the atoms' and the queries' frequencies. Hard formulas are respected by a conditional of 0 or 1, so a network with
 * hard formulas needs a start that satisfies them (found by SampleSAT) and may not be ergodic.
 *
 * @param ground - The ground network.
 * @param options - Evidence, query formulas, burn-in and the scan order.
 * @returns The algorithm, whose state is the world and the running frequencies.
 * @throws MdmlError with code `value` for an unknown evidence atom or hard formulas that cannot be satisfied.
 * @example
 * ```ts
 * import { groundMln, mlnGibbsAlgorithm } from 'mdml-engine/logic'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: Math.log(3), formula: 'p(a)' }] })
 * const s = runAlgorithm(mlnGibbsAlgorithm(g, { burnIn: 10 }), { seed: 1, maxSteps: 4000 }).state
 * Math.abs(s.atomCounts[0] / s.samples - 0.75) < 0.05 // => true
 * ```
 */
export function mlnGibbsAlgorithm(
  ground: GroundMln,
  options: MlnGibbsOptions = {},
): Algorithm<MlnSamplerState, MlnSamplerPath> {
  const p = prepare(ground, options, 'mlnGibbs')
  const scan = options.scan ?? 'systematic'
  return {
    name: 'mln-gibbs',
    config: {
      atoms: p.n,
      formulas: ground.formulas.length,
      scan,
      burnIn: p.burnIn,
      queries: options.queries ?? [],
    } as Json,
    schema: contractSchema(mlnSamplerContract),
    init: (key) => freshState(ground, p, initialWorld(ground, p, key, 'mlnGibbs')),
    step: (s, ctx) => {
      const u = uniformStream(ctx.key)
      const world = Uint8Array.from(s.world)
      const count = p.freeAtoms.length
      for (let k = 0; k < count; k++) {
        const a = scan === 'systematic' ? (p.freeAtoms[k] as number) : (p.freeAtoms[Math.floor(u() * count)] as number)
        const d = conditional(ground, world, a)
        const p1 = d === Number.POSITIVE_INFINITY ? 1 : d === Number.NEGATIVE_INFINITY ? 0 : sigmoidOf(d)
        world[a] = u() < p1 ? 1 : 0
      }
      return accumulate(ground, p, s, world)
    },
    observe: observeSampler,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MlnSamplerState,
  }
}

/**
 * MC-SAT (Poon and Domingos 2006): slice sampling for Markov logic. From the current world, each ground formula that
 * is true and has positive weight `w` (or false with negative weight, in which case its negation is true) is added to
 * a set `M` of constraints with probability `1 - exp(-|w|)`, hard formulas always; the next world is a uniform draw
 * from the worlds satisfying every constraint in `M` (SampleSAT, or enumeration with `sampler: 'exact'`). Unlike Gibbs
 * it moves between modes that deterministic constraints separate. One iteration per step.
 *
 * @param ground - The ground network.
 * @param options - Evidence, query formulas, burn-in, the slice sampler and its knobs.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for an unknown evidence atom or unsatisfiable hard formulas.
 * @example
 * ```ts
 * import { groundMln, mlnMcSatAlgorithm } from 'mdml-engine/logic'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: Math.log(3), formula: 'p(a)' }] })
 * const s = runAlgorithm(mlnMcSatAlgorithm(g, { burnIn: 10, sampler: 'exact' }), { seed: 1, maxSteps: 4000 }).state
 * Math.abs(s.atomCounts[0] / s.samples - 0.75) < 0.05 // => true
 * ```
 */
export function mlnMcSatAlgorithm(
  ground: GroundMln,
  options: MlnMcSatOptions = {},
): Algorithm<MlnSamplerState, MlnSamplerPath> {
  const p = prepare(ground, options, 'mlnMcSat')
  const sampler = options.sampler ?? 'samplesat'
  const enc = new Encoder(ground)
  // Constraints for each ground formula holding (positive) and failing (negated), with auxiliaries allocated once.
  const holdsClauses = ground.formulas.map((g) => enc.constraint(g.formula))
  const failsClauses = ground.formulas.map((g) => enc.constraint({ kind: 'not', arg: g.formula }))
  const evidenceUnits: number[][] = []
  for (let i = 0; i < p.n; i++) if (p.fixed[i] !== -1) evidenceUnits.push([p.fixed[i] === 1 ? i + 1 : -(i + 1)])
  return {
    name: 'mln-mc-sat',
    config: {
      atoms: p.n,
      formulas: ground.formulas.length,
      sampler,
      burnIn: p.burnIn,
      queries: options.queries ?? [],
    } as Json,
    schema: contractSchema(mlnSamplerContract),
    init: (key) => freshState(ground, p, initialWorld(ground, p, key, 'mlnMcSat')),
    step: (s, ctx) => {
      const u = uniformStream(ctx.key)
      const m: number[][] = [...evidenceUnits]
      ground.formulas.forEach((g, gi) => {
        const w = g.weight
        const holds = g.holds(s.world)
        if (w > 0 && holds) {
          if (isHard(w) || u() < 1 - Math.exp(-w)) m.push(...(holdsClauses[gi] as number[][]))
        } else if (w < 0 && !holds) {
          if (isHard(w) || u() < 1 - Math.exp(w)) m.push(...(failsClauses[gi] as number[][]))
        }
      })
      // Compact the variables the constraints mention; atoms outside them are uniform (or evidence).
      const usedVars = new Map<number, number>()
      for (const c of m) for (const l of c) if (!usedVars.has(Math.abs(l))) usedVars.set(Math.abs(l), usedVars.size + 1)
      const compact = m.map((c) => c.map((l) => (l > 0 ? usedVars.get(l) : -(usedVars.get(-l) as number)) as number))
      const model =
        sampler === 'exact'
          ? exactModel(compact, usedVars.size, u)
          : sampleSat(compact, usedVars.size, u, options.sampleSat)
      if (model === null) throw new MdmlError('value', 'mlnMcSat: the slice has no model within budget')
      const world = new Uint8Array(p.n)
      for (let i = 0; i < p.n; i++) {
        const v = usedVars.get(i + 1)
        world[i] =
          v !== undefined ? (model[v - 1] as number) : p.fixed[i] === -1 ? (u() < 0.5 ? 1 : 0) : (p.fixed[i] as number)
      }
      return accumulate(ground, p, s, world)
    },
    observe: observeSampler,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MlnSamplerState,
  }
}

/** The result of a sampler run. */
export interface MlnSampleResult {
  /** The estimated marginal probability of each atom. */
  readonly marginals: Readonly<Record<string, number>>
  /** The estimated probability of each query formula. */
  readonly queries: readonly number[]
  readonly samples: number
  /** The last world. */
  readonly world: World
}

function resultOf(ground: GroundMln, s: MlnSamplerState): MlnSampleResult {
  const m = marginalsOf(s)
  return {
    marginals: Object.fromEntries(ground.atoms.map((a, i) => [a, m[i] as number])),
    queries: Array.from(s.queryCounts, (c) => (s.samples === 0 ? 0 : c / s.samples)),
    samples: s.samples,
    world: worldRecord(ground, s.world),
  }
}

/**
 * Marginals and query probabilities by Gibbs sampling.
 *
 * @param ground - The ground network.
 * @param options - As {@link mlnGibbsAlgorithm}, plus the number of sweeps after burn-in (default 2000) and a seed.
 * @returns The estimates.
 * @example
 * ```ts
 * import { groundMln, mlnGibbs } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: Math.log(3), formula: 'p(a)' }] })
 * Math.abs((mlnGibbs(g, { samples: 4000, seed: 3 }).marginals['p(a)'] ?? 0) - 0.75) < 0.05 // => true
 * ```
 */
export function mlnGibbs(
  ground: GroundMln,
  options: MlnGibbsOptions & { readonly samples?: number; readonly seed?: number } = {},
): MlnSampleResult {
  const { samples = 2000, seed = 0, ...rest } = options
  const alg = mlnGibbsAlgorithm(ground, rest)
  const burn = rest.burnIn ?? 100
  return resultOf(ground, runAlgorithmState(alg, seed, burn + samples))
}

/**
 * Marginals and query probabilities by MC-SAT.
 *
 * @param ground - The ground network.
 * @param options - As {@link mlnMcSatAlgorithm}, plus the number of iterations after burn-in (default 2000) and a seed.
 * @returns The estimates.
 * @example
 * ```ts
 * import { groundMln, mlnMcSat } from 'mdml-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: Math.log(3), formula: 'p(a)' }] })
 * Math.abs((mlnMcSat(g, { samples: 4000, seed: 3 }).marginals['p(a)'] ?? 0) - 0.75) < 0.05 // => true
 * ```
 */
export function mlnMcSat(
  ground: GroundMln,
  options: MlnMcSatOptions & { readonly samples?: number; readonly seed?: number } = {},
): MlnSampleResult {
  const { samples = 2000, seed = 0, ...rest } = options
  const alg = mlnMcSatAlgorithm(ground, rest)
  const burn = rest.burnIn ?? 100
  return resultOf(ground, runAlgorithmState(alg, seed, burn + samples))
}

// ------------------------------------------------------------------------------------------------ MAP

/** The most probable world given evidence. */
export interface MlnMap {
  readonly status: 'optimal' | 'unsat' | 'unknown'
  /** The most probable world (undefined unless optimal). */
  readonly world: World | null
  /** The total weight of the falsified soft formulas. */
  readonly cost: number
  /** The log weight of the world, `sum_i w_i n_i(x)`. */
  readonly logWeight: number
}

/**
 * MAP inference: the most probable world given the evidence, as a weighted partial MaxSAT problem (a hard formula is a
 * hard clause set; a formula of positive weight is a soft clause, falsifying it costs its weight; a negative weight
 * makes its negation soft). Solved exactly by the branch and bound of `maxsat`.
 *
 * @param ground - The ground network.
 * @param options - Evidence atoms and a node budget.
 * @returns The world, its cost and its log weight.
 * @throws MdmlError with code `value` for an unknown evidence atom.
 * @example
 * ```ts
 * import { groundMln, mlnMap } from 'mdml-engine/logic'
 * const g = groundMln({
 *   domain: ['a'],
 *   formulas: [{ weight: 2, formula: 'smokes(X) -> cancer(X)' }, { weight: 1, formula: 'smokes(X)' }],
 * })
 * mlnMap(g).world // => { 'smokes(a)': true, 'cancer(a)': true }
 * ```
 */
export function mlnMap(
  ground: GroundMln,
  options: { readonly evidence?: World; readonly maxNodes?: number } = {},
): MlnMap {
  const enc = new Encoder(ground)
  const hard: number[][] = []
  const soft: { clause: number[]; weight: number }[] = []
  for (const g of ground.formulas) {
    if (g.weight === 0) continue
    const positive = g.weight > 0
    const f: Formula = positive ? g.formula : { kind: 'not', arg: g.formula }
    if (isHard(g.weight)) {
      hard.push(...enc.constraint(f))
      continue
    }
    const e = enc.soft(f)
    hard.push(...e.definitions)
    if (e.clause === 'true' || e.clause === 'false') continue
    soft.push({ clause: e.clause, weight: Math.abs(g.weight) })
  }
  for (const [name, v] of Object.entries(options.evidence ?? {})) {
    const i = ground.atomIndex.get(name)
    if (i === undefined) throw new MdmlError('value', `mlnMap: unknown ground atom ${name}`)
    hard.push([v ? i + 1 : -(i + 1)])
  }
  const r = maxsat(
    { numVars: enc.next, hard, soft },
    options.maxNodes === undefined ? {} : { maxNodes: options.maxNodes },
  )
  if (r.model === null)
    return {
      status: r.status === 'optimal' ? 'unknown' : r.status,
      world: null,
      cost: r.cost,
      logWeight: Number.NEGATIVE_INFINITY,
    }
  const vector = new Uint8Array(ground.atoms.length)
  for (const l of r.model) if (l > 0 && l <= vector.length) vector[l - 1] = 1
  return { status: r.status, world: worldRecord(ground, vector), cost: r.cost, logWeight: mlnLogWeight(ground, vector) }
}

function runAlgorithmState<S>(alg: Algorithm<S, never>, seed: number, steps: number): S {
  return runAlgorithm(alg as unknown as Algorithm<S>, { seed, maxSteps: steps }).state
}
