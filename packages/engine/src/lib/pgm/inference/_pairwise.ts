// A discrete model with unary and pairwise factors only, as energies (the negative log values): the form the graph-cut
// MAP methods work on. Factors over the same variable or the same pair are summed; a factor over three or more
// variables, or a log value that is not finite, is an error. Shared by the graph-cut and move-making methods (not
// re-exported).
import { OmaiError } from '../../base/index.ts'
import { type DiscreteModel, stateNames } from '../models/factor-graph.ts'
import { type Compiled, compile } from './_compiled.ts'
import type { MapResult } from './map.ts'

/** One pairwise term: variables `i < j` and the energy table, row-major over (state of i, state of j). */
export interface PairTerm {
  readonly i: number
  readonly j: number
  readonly table: Float64Array
}

/** A neighbour of a variable: the pair term, the other variable and whether the variable is the first of the pair. */
export interface Neighbour {
  readonly term: number
  readonly other: number
  readonly first: boolean
}

/** A model as energies: E(x) = constant + sum_i unary[i][x_i] + sum over pairs of table(x_i, x_j). */
export interface PairwiseEnergy {
  readonly compiled: Compiled
  readonly names: readonly string[]
  readonly card: readonly number[]
  readonly unary: readonly Float64Array[]
  readonly pairs: readonly PairTerm[]
  readonly neighbours: readonly (readonly Neighbour[])[]
  readonly constant: number
}

/** Reduce a model (under evidence) to unary and pairwise energies. */
export function pairwiseEnergy(
  model: DiscreteModel,
  evidence: Readonly<Record<string, string | number>> = {},
  caller = 'pairwise MAP',
): PairwiseEnergy {
  const c = compile(model, evidence)
  const unary = c.names.map((_, i) => new Float64Array(c.card[i] ?? 1))
  const tables = new Map<number, PairTerm>()
  const n = c.names.length
  for (const f of c.factors) {
    if (f.vars.length > 2)
      throw new OmaiError('value', `${caller}: a factor over ${f.vars.length} variables; pairwise models only`)
    for (const v of f.logValues)
      if (!Number.isFinite(v)) throw new OmaiError('value', `${caller}: a factor value is ${v}; energies must be finite`)
    if (f.vars.length === 1) {
      const u = unary[f.vars[0] ?? 0] as Float64Array
      for (let x = 0; x < u.length; x++) u[x] = (u[x] ?? 0) - (f.logValues[x] ?? 0)
      continue
    }
    const a = f.vars[0] ?? 0
    const b = f.vars[1] ?? 0
    if (a === b) throw new OmaiError('value', `${caller}: a factor repeats variable ${c.names[a]}`)
    const [i, j] = a < b ? [a, b] : [b, a]
    const ci = c.card[i] ?? 1
    const cj = c.card[j] ?? 1
    const key = i * n + j
    let term = tables.get(key)
    if (term === undefined) {
      term = { i, j, table: new Float64Array(ci * cj) }
      tables.set(key, term)
    }
    const cb = c.card[b] ?? 1
    for (let x = 0; x < ci * cj; x++) {
      const xi = Math.floor(x / cj)
      const xj = x % cj
      const idx = a < b ? xi * cb + xj : xj * (c.card[a] ?? 1) + xi
      term.table[x] = (term.table[x] ?? 0) - (f.logValues[idx] ?? 0)
    }
  }
  const pairs = [...tables.values()].sort((p, q) => p.i - q.i || p.j - q.j)
  const neighbours: Neighbour[][] = c.names.map(() => [])
  for (const [t, p] of pairs.entries()) {
    neighbours[p.i]?.push({ term: t, other: p.j, first: true })
    neighbours[p.j]?.push({ term: t, other: p.i, first: false })
  }
  return { compiled: c, names: c.names, card: c.card, unary, pairs, neighbours, constant: -c.constant }
}

/** The energy of a full assignment. */
export function energyOf(p: PairwiseEnergy, x: ArrayLike<number>): number {
  let e = p.constant
  for (const [i, u] of p.unary.entries()) e += u[x[i] ?? 0] ?? 0
  for (const t of p.pairs) e += t.table[(x[t.i] ?? 0) * (p.card[t.j] ?? 1) + (x[t.j] ?? 0)] ?? 0
  return e
}

/** The pair energy of the term between its two variables for states (a of the first, b of the second). */
export function pairEnergy(p: PairwiseEnergy, term: number, a: number, b: number): number {
  const t = p.pairs[term] as PairTerm
  return t.table[a * (p.card[t.j] ?? 1) + b] ?? 0
}

/** The energy a neighbour link adds to a variable in state `s` when the other variable is in state `o`. */
export function linkEnergy(p: PairwiseEnergy, link: Neighbour, s: number, o: number): number {
  return link.first ? pairEnergy(p, link.term, s, o) : pairEnergy(p, link.term, o, s)
}

/** The unary-energy minimiser of every variable: a starting assignment. */
export function unaryArgmin(p: PairwiseEnergy): Int32Array {
  return Int32Array.from(p.unary, (u) => {
    let best = 0
    for (let k = 1; k < u.length; k++) if ((u[k] ?? 0) < (u[best] ?? 0)) best = k
    return best
  })
}

/** A MAP result (log value = minus the energy) for an assignment of the unobserved variables. */
export function mapResultOf(model: DiscreteModel, p: PairwiseEnergy, x: ArrayLike<number>): MapResult {
  const indices = Object.fromEntries(p.names.map((v, i) => [v, x[i] ?? 0]))
  return { assignment: stateNames(model, indices), indices, logValue: -energyOf(p, x) }
}

/** The energy of variable `i` in state `s` given the current states of the others. */
function localEnergy(p: PairwiseEnergy, x: ArrayLike<number>, i: number, s: number): number {
  let e = p.unary[i]?.[s] ?? 0
  for (const link of p.neighbours[i] ?? []) e += linkEnergy(p, link, s, x[link.other] ?? 0)
  return e
}

/** Coordinate descent over the listed variables only: each moves to a strictly lower-energy state given the others. */
export function descendOver(p: PairwiseEnergy, x: Int32Array, free: readonly number[], maxSweeps = 100): void {
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let changed = false
    for (const i of free) {
      let argmin = x[i] ?? 0
      let min = localEnergy(p, x, i, argmin)
      for (let s = 0; s < (p.card[i] ?? 1); s++) {
        const e = localEnergy(p, x, i, s)
        if (e < min - 1e-12) {
          min = e
          argmin = s
        }
      }
      if (argmin !== x[i]) {
        x[i] = argmin
        changed = true
      }
    }
    if (!changed) return
  }
}
