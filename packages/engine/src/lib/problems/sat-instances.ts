/**
 * Instances of Boolean satisfiability with their truth: random k-SAT at a chosen clause-to-variable ratio (satisfiable
 * with probability falling from 1 to 0 around the phase-transition ratio, about 4.27 for 3-SAT), planted k-SAT (every
 * clause satisfied by a hidden assignment, so the formula is satisfiable and the assignment is the witness) and the
 * pigeonhole formulas, unsatisfiable for any number of holes. Formulas are `Cnf` values for the solvers of `logic`.
 */

import { OmaiError } from '../base/index.ts'
import { type Cnf, cnf } from '../logic/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'

/** The clause-to-variable ratio at which random k-SAT turns from almost always to almost never satisfiable. */
export const SAT_THRESHOLDS: Readonly<Record<2 | 3 | 4 | 5, number>> = { 2: 1, 3: 4.267, 4: 9.931, 5: 21.117 }

/** A planted instance: the formula and the assignment it was built around. */
export interface PlantedSat {
  readonly formula: Cnf
  /** The hidden assignment as signed literals, one per variable; it satisfies every clause. */
  readonly assignment: readonly number[]
}

function clause(key: Key, n: number, k: number, attempt: number): number[] {
  const u = uniform(foldIn(key, attempt), [n + k]).data
  // k distinct variables: the k smallest of n random keys (a random sample without replacement).
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (u[a] ?? 0) - (u[b] ?? 0))
  return order.slice(0, k).map((v, j) => ((u[n + j] ?? 0) < 0.5 ? -(v + 1) : v + 1))
}

/**
 * Uniform random k-SAT: `round(ratio n)` clauses, each on `k` distinct variables with independent random signs. Its
 * satisfiability is not known in advance; above the threshold ({@link SAT_THRESHOLDS}) it is unsatisfiable with
 * probability near 1 and below it satisfiable, the sharp transition that makes these instances hard near it.
 *
 * @param key - The key.
 * @param n - The number of variables.
 * @param ratio - The clauses per variable (default the 3-SAT threshold).
 * @param k - The clause width (default 3).
 * @returns The formula.
 * @throws OmaiError with code `value` for `k` above `n`.
 * @example
 * ```ts
 * import { randomKSat } from 'omai-engine/problems'
 * import { key } from 'omai-engine/random'
 * randomKSat(key(0), 20, 4).clauses.length // => 80
 * ```
 */
export function randomKSat(key: Key, n: number, ratio: number = SAT_THRESHOLDS[3], k = 3): Cnf {
  if (k > n) throw new OmaiError('value', `randomKSat: clause width ${k} exceeds ${n} variables`)
  const m = Math.round(ratio * n)
  return cnf(
    Array.from({ length: m }, (_, i) => clause(foldIn(key, 1), n, k, i)),
    n,
  )
}

/**
 * Planted k-SAT: a hidden assignment is drawn, and every clause is redrawn until the assignment satisfies it, so the
 * formula is satisfiable and the assignment is its witness. (Planted formulas are easier than uniform ones at the same
 * ratio: the planted solution biases the signs.)
 *
 * @param key - The key.
 * @param n - The number of variables.
 * @param ratio - The clauses per variable.
 * @param k - The clause width (default 3).
 * @returns The formula and its assignment.
 * @example
 * ```ts
 * import { plantedKSat } from 'omai-engine/problems'
 * import { satisfies } from 'omai-engine/logic'
 * import { key } from 'omai-engine/random'
 * const p = plantedKSat(key(0), 30, 4.3)
 * satisfies(p.formula, p.assignment) // => true
 * ```
 */
export function plantedKSat(key: Key, n: number, ratio: number, k = 3): PlantedSat {
  if (k > n) throw new OmaiError('value', `plantedKSat: clause width ${k} exceeds ${n} variables`)
  const bits = uniform(foldIn(key, 0), [n]).data
  const assignment = Array.from({ length: n }, (_, i) => ((bits[i] ?? 0) < 0.5 ? -(i + 1) : i + 1))
  const truth = (lit: number): boolean => assignment[Math.abs(lit) - 1] === lit
  const m = Math.round(ratio * n)
  const clauses: number[][] = []
  let attempt = 0
  while (clauses.length < m) {
    const c = clause(foldIn(key, 1), n, k, attempt++)
    if (c.some(truth)) clauses.push(c)
  }
  return { formula: cnf(clauses, n), assignment }
}

/**
 * The pigeonhole formula: `pigeons` pigeons into `holes` holes, each pigeon in a hole and no hole with two pigeons
 * (variable `p(i, j) = i * holes + j + 1` says pigeon `i` is in hole `j`). Unsatisfiable exactly when there are more
 * pigeons than holes, and a standing hard case for resolution: every refutation is exponentially long.
 *
 * @param holes - The number of holes.
 * @param pigeons - The number of pigeons (default `holes + 1`: unsatisfiable).
 * @returns The formula.
 * @example
 * ```ts
 * import { pigeonhole } from 'omai-engine/problems'
 * pigeonhole(3).clauses.length // => 4 + 3 * 6
 * ```
 */
export function pigeonhole(holes: number, pigeons: number = holes + 1): Cnf {
  const p = (i: number, j: number): number => i * holes + j + 1
  const clauses: number[][] = []
  for (let i = 0; i < pigeons; i++) clauses.push(Array.from({ length: holes }, (_, j) => p(i, j)))
  for (let j = 0; j < holes; j++)
    for (let a = 0; a < pigeons; a++) for (let b = a + 1; b < pigeons; b++) clauses.push([-p(a, j), -p(b, j)])
  return cnf(clauses, pigeons * holes)
}
