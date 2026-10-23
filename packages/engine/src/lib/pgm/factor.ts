import { WhiteboxError } from '../base/index.ts'

/**
 * A discrete factor in log space: a non-negative function of a few discrete variables, stored as log values in
 * row-major order (the last variable varies fastest, as pgmpy's DiscreteFactor).
 *
 * @concept factors
 */
export interface Factor {
  readonly variables: readonly string[]
  readonly cardinality: readonly number[]
  readonly logValues: Float64Array
}

const size = (card: readonly number[]): number => card.reduce((a, b) => a * b, 1)

/** Row-major strides of a cardinality vector. */
function strides(card: readonly number[]): number[] {
  const s = new Array<number>(card.length).fill(1)
  for (let i = card.length - 2; i >= 0; i--) s[i] = (s[i + 1] ?? 1) * (card[i + 1] ?? 1)
  return s
}

/**
 * A factor from (non-negative) values in row-major order.
 *
 * @param variables - Variable names.
 * @param cardinality - Number of states of each.
 * @param values - The values (stored as logs; zeros become -inf).
 * @returns The factor.
 * @concept factors
 * @example
 * ```ts
 * import { factor } from 'whitebox-engine/pgm'
 * factor(['a'], [2], [0.3, 0.7]).logValues.length // => 2
 * ```
 */
export function factor(
  variables: readonly string[],
  cardinality: readonly number[],
  values: ArrayLike<number>,
): Factor {
  if (variables.length !== cardinality.length) throw new WhiteboxError('shape', 'factor: one cardinality per variable')
  if (new Set(variables).size !== variables.length)
    throw new WhiteboxError('value', 'factor: variables must be distinct')
  if (values.length !== size(cardinality)) {
    throw new WhiteboxError('shape', `factor: ${values.length} values for ${size(cardinality)} assignments`)
  }
  return { variables, cardinality, logValues: Float64Array.from(values, (v) => Math.log(v)) }
}

/**
 * A factor's values (not logs), in row-major order.
 *
 * @param f - The factor.
 * @returns The values.
 * @example
 * ```ts
 * import { factor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(factor(['a'], [2], [0.25, 0.75])) // => [0.25, 0.75]
 * ```
 */
export function factorValues(f: Factor): number[] {
  return Array.from(f.logValues, Math.exp)
}

/** The index of an assignment (one state per variable, in the factor's order). */
function indexOf(card: readonly number[], st: readonly number[], states: readonly number[]): number {
  let i = 0
  for (let k = 0; k < card.length; k++) i += (states[k] ?? 0) * (st[k] ?? 0)
  return i
}

/** Iterate over all assignments of a cardinality vector. */
function* assignments(card: readonly number[]): Generator<number[]> {
  const n = size(card)
  const cur = new Array<number>(card.length).fill(0)
  for (let i = 0; i < n; i++) {
    yield cur.slice()
    for (let k = card.length - 1; k >= 0; k--) {
      cur[k] = (cur[k] ?? 0) + 1
      if ((cur[k] ?? 0) < (card[k] ?? 1)) break
      cur[k] = 0
    }
  }
}

/**
 * The product of two factors (a sum of log values over the union of their variables).
 *
 * @param a - First factor.
 * @param b - Second factor.
 * @returns The product, over a's variables followed by b's new ones.
 * @throws WhiteboxError with code `value` when a shared variable has different cardinalities.
 * @concept factors
 * @example
 * ```ts
 * import { factor, factorProduct, factorValues } from 'whitebox-engine/pgm'
 * factorValues(factorProduct(factor(['a'], [2], [1, 2]), factor(['b'], [2], [3, 4]))) // => [3, 4, 6, 8]
 * ```
 */
export function factorProduct(a: Factor, b: Factor): Factor {
  const variables = [...a.variables, ...b.variables.filter((v) => !a.variables.includes(v))]
  const cardinality = variables.map((v) => {
    const ia = a.variables.indexOf(v)
    const ib = b.variables.indexOf(v)
    if (ia >= 0 && ib >= 0 && a.cardinality[ia] !== b.cardinality[ib]) {
      throw new WhiteboxError(
        'value',
        `factorProduct: ${v} has cardinality ${a.cardinality[ia]} and ${b.cardinality[ib]}`,
      )
    }
    return (ia >= 0 ? a.cardinality[ia] : b.cardinality[ib]) ?? 1
  })
  const sa = strides(a.cardinality)
  const sb = strides(b.cardinality)
  const posA = a.variables.map((v) => variables.indexOf(v))
  const posB = b.variables.map((v) => variables.indexOf(v))
  const out = new Float64Array(size(cardinality))
  let i = 0
  for (const s of assignments(cardinality)) {
    const ia = indexOf(
      a.cardinality,
      sa,
      posA.map((p) => s[p] ?? 0),
    )
    const ib = indexOf(
      b.cardinality,
      sb,
      posB.map((p) => s[p] ?? 0),
    )
    out[i++] = (a.logValues[ia] ?? 0) + (b.logValues[ib] ?? 0)
  }
  return { variables, cardinality, logValues: out }
}

const logAdd = (x: number, y: number): number => {
  if (x === Number.NEGATIVE_INFINITY) return y
  if (y === Number.NEGATIVE_INFINITY) return x
  const m = Math.max(x, y)
  return m + Math.log(Math.exp(x - m) + Math.exp(y - m))
}

/** Eliminate variables by summing (log-sum-exp) or maximising over their states. */
function eliminate(f: Factor, variables: readonly string[], mode: 'sum' | 'max'): Factor {
  for (const v of variables)
    if (!f.variables.includes(v)) throw new WhiteboxError('value', `marginalise: ${v} is not in the factor`)
  const keep = f.variables.filter((v) => !variables.includes(v))
  const card = keep.map((v) => f.cardinality[f.variables.indexOf(v)] ?? 1)
  const sk = strides(card)
  const pos = keep.map((v) => f.variables.indexOf(v))
  const out = new Float64Array(size(card)).fill(Number.NEGATIVE_INFINITY)
  let i = 0
  for (const s of assignments(f.cardinality)) {
    const j = indexOf(
      card,
      sk,
      pos.map((p) => s[p] ?? 0),
    )
    const v = f.logValues[i++] ?? Number.NEGATIVE_INFINITY
    out[j] =
      mode === 'sum' ? logAdd(out[j] ?? Number.NEGATIVE_INFINITY, v) : Math.max(out[j] ?? Number.NEGATIVE_INFINITY, v)
  }
  return { variables: keep, cardinality: card, logValues: out }
}

/**
 * Sum variables out of a factor.
 *
 * @param f - The factor.
 * @param variables - Variables to sum out.
 * @returns The marginal factor.
 * @concept factors
 * @example
 * ```ts
 * import { factor, marginalise, factorValues } from 'whitebox-engine/pgm'
 * factorValues(marginalise(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), ['b'])) // => [3, 7]
 * ```
 */
export function marginalise(f: Factor, variables: readonly string[]): Factor {
  return eliminate(f, variables, 'sum')
}

/**
 * Maximise variables out of a factor (max-product elimination).
 *
 * @param f - The factor.
 * @param variables - Variables to maximise out.
 * @returns The max-marginal factor.
 * @example
 * ```ts
 * import { factor, maxMarginalise, factorValues } from 'whitebox-engine/pgm'
 * factorValues(maxMarginalise(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), ['b'])) // => [2, 4]
 * ```
 */
export function maxMarginalise(f: Factor, variables: readonly string[]): Factor {
  return eliminate(f, variables, 'max')
}

/**
 * Condition a factor on evidence: keep the entries consistent with the observed states and drop those variables.
 *
 * @param f - The factor.
 * @param evidence - Observed states (indices) by variable; variables not in the factor are ignored.
 * @returns The reduced factor.
 * @example
 * ```ts
 * import { factor, reduceFactor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(reduceFactor(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), { a: 1 })) // => [3, 4]
 * ```
 */
export function reduceFactor(f: Factor, evidence: Readonly<Record<string, number>>): Factor {
  const fixed = f.variables.filter((v) => v in evidence)
  if (fixed.length === 0) return f
  const keep = f.variables.filter((v) => !(v in evidence))
  const card = keep.map((v) => f.cardinality[f.variables.indexOf(v)] ?? 1)
  const values: number[] = []
  let i = 0
  for (const s of assignments(f.cardinality)) {
    const v = f.logValues[i++] ?? Number.NEGATIVE_INFINITY
    if (fixed.every((name) => s[f.variables.indexOf(name)] === evidence[name])) values.push(v)
  }
  return { variables: keep, cardinality: card, logValues: Float64Array.from(values) }
}

/**
 * Scale a factor to sum to one.
 *
 * @param f - The factor.
 * @returns The normalised factor.
 * @throws WhiteboxError with code `value` when every entry is zero.
 * @example
 * ```ts
 * import { factor, normaliseFactor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(normaliseFactor(factor(['a'], [2], [1, 3]))) // => [0.25, 0.75]
 * ```
 */
export function normaliseFactor(f: Factor): Factor {
  let total = Number.NEGATIVE_INFINITY
  for (const v of f.logValues) total = logAdd(total, v)
  if (total === Number.NEGATIVE_INFINITY)
    throw new WhiteboxError('value', 'normaliseFactor: the factor is zero everywhere (inconsistent evidence?)')
  return { ...f, logValues: f.logValues.map((v) => v - total) }
}

/**
 * The logarithm of the total mass of a factor (the log partition function, or log-evidence after reduction).
 *
 * @param f - The factor.
 * @returns log sum of its values.
 * @example
 * ```ts
 * import { factor, logTotal } from 'whitebox-engine/pgm'
 * logTotal(factor(['a'], [2], [1, 1])) === Math.LN2 // => true
 * ```
 */
export function logTotal(f: Factor): number {
  let total = Number.NEGATIVE_INFINITY
  for (const v of f.logValues) total = logAdd(total, v)
  return total
}

/**
 * The value of a factor at a full assignment.
 *
 * @param f - The factor.
 * @param assignment - A state index for each of the factor's variables.
 * @returns The value there.
 * @example
 * ```ts
 * import { factor, factorAt } from 'whitebox-engine/pgm'
 * factorAt(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), { a: 1, b: 0 }) // => 3
 * ```
 */
export function factorAt(f: Factor, assignment: Readonly<Record<string, number>>): number {
  return Math.exp(
    f.logValues[
      indexOf(
        f.cardinality,
        strides(f.cardinality),
        f.variables.map((v) => assignment[v] ?? 0),
      )
    ] ?? Number.NEGATIVE_INFINITY,
  )
}

/**
 * The assignment of largest value (the first, on ties).
 *
 * @param f - The factor.
 * @returns A state index for each variable.
 * @example
 * ```ts
 * import { factor, factorArgmax } from 'whitebox-engine/pgm'
 * factorArgmax(factor(['a', 'b'], [2, 2], [1, 5, 3, 4])) // => { a: 0, b: 1 }
 * ```
 */
export function factorArgmax(f: Factor): Record<string, number> {
  let best = Number.NEGATIVE_INFINITY
  let arg: number[] = []
  let i = 0
  for (const s of assignments(f.cardinality)) {
    const v = f.logValues[i++] ?? Number.NEGATIVE_INFINITY
    if (v > best) {
      best = v
      arg = s
    }
  }
  return Object.fromEntries(f.variables.map((v, k) => [v, arg[k] ?? 0]))
}

/**
 * Every joint assignment of some variables, in row-major order (the last variable fastest).
 *
 * @param variables - The variables.
 * @param cardinality - Their numbers of states.
 * @returns The assignments.
 * @example
 * ```ts
 * import { allAssignments } from 'whitebox-engine/pgm'
 * allAssignments(['a', 'b'], [2, 2]).length // => 4
 * ```
 */
export function allAssignments(variables: readonly string[], cardinality: readonly number[]): Record<string, number>[] {
  return Array.from(assignments(cardinality), (s) => Object.fromEntries(variables.map((v, k) => [v, s[k] ?? 0])))
}
