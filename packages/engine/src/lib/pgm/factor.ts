import { logSumExpOf, WhiteboxError } from '../base/index.ts'

/**
 * A discrete variable: a name and the names of its states.
 *
 * @concept factors
 */
export interface DiscreteVariable {
  readonly name: string
  readonly states: readonly string[]
}

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

const NEG = Number.NEGATIVE_INFINITY
const size = (card: readonly number[]): number => card.reduce((a, b) => a * b, 1)

/** Row-major strides of a cardinality vector. */
function strides(card: readonly number[]): number[] {
  const s = new Array<number>(card.length).fill(1)
  for (let i = card.length - 2; i >= 0; i--) s[i] = (s[i + 1] ?? 1) * (card[i + 1] ?? 1)
  return s
}

/** The strides of `f` laid out along `variables` (zero for the variables `f` does not mention). */
function aligned(f: Factor, variables: readonly string[]): number[] {
  const s = strides(f.cardinality)
  return variables.map((v) => {
    const i = f.variables.indexOf(v)
    return i < 0 ? 0 : (s[i] ?? 0)
  })
}

/**
 * Visit every assignment of `card` in row-major order, keeping one running flat index per stride vector (an odometer:
 * no per-assignment allocation).
 */
function odometer(
  card: readonly number[],
  strideSets: readonly (readonly number[])[],
  visit: (i: number, idx: Int32Array) => void,
): void {
  const n = size(card)
  const d = card.length
  const m = strideSets.length
  const ctr = new Int32Array(d)
  const idx = new Int32Array(m)
  for (let i = 0; i < n; i++) {
    visit(i, idx)
    for (let k = d - 1; k >= 0; k--) {
      ctr[k] = (ctr[k] ?? 0) + 1
      const c = card[k] ?? 1
      for (let j = 0; j < m; j++) idx[j] = (idx[j] ?? 0) + (strideSets[j]?.[k] ?? 0)
      if ((ctr[k] ?? 0) < c) break
      for (let j = 0; j < m; j++) idx[j] = (idx[j] ?? 0) - (strideSets[j]?.[k] ?? 0) * c
      ctr[k] = 0
    }
  }
}

function checkScope(variables: readonly string[], cardinality: readonly number[], n: number, name: string): void {
  if (variables.length !== cardinality.length) throw new WhiteboxError('shape', `${name}: one cardinality per variable`)
  if (new Set(variables).size !== variables.length)
    throw new WhiteboxError('value', `${name}: variables must be distinct`)
  if (n !== size(cardinality))
    throw new WhiteboxError('shape', `${name}: ${n} values for ${size(cardinality)} assignments`)
}

/**
 * A factor from (non-negative) values in row-major order.
 *
 * @param variables - Variable names.
 * @param cardinality - Number of states of each.
 * @param values - The values (stored as logs; zeros become -inf).
 * @returns The factor.
 * @throws WhiteboxError with code `shape` when the sizes disagree, `value` for repeated variables.
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
  checkScope(variables, cardinality, values.length, 'factor')
  return { variables: [...variables], cardinality: [...cardinality], logValues: Float64Array.from(values, Math.log) }
}

/**
 * A factor from log values in row-major order (for potentials written as energies, log-linear models and messages).
 *
 * @param variables - Variable names.
 * @param cardinality - Number of states of each.
 * @param logValues - The log values.
 * @returns The factor.
 * @throws WhiteboxError with code `shape` when the sizes disagree.
 * @concept factors
 * @example
 * ```ts
 * import { logFactor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(logFactor(['a'], [2], [0, Math.log(2)])) // => [1, 2]
 * ```
 */
export function logFactor(
  variables: readonly string[],
  cardinality: readonly number[],
  logValues: ArrayLike<number>,
): Factor {
  checkScope(variables, cardinality, logValues.length, 'logFactor')
  return { variables: [...variables], cardinality: [...cardinality], logValues: Float64Array.from(logValues) }
}

/**
 * The unit factor (all ones) over some variables.
 *
 * @param variables - Variable names.
 * @param cardinality - Number of states of each.
 * @returns The factor.
 * @example
 * ```ts
 * import { unitFactor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(unitFactor(['a'], [3])) // => [1, 1, 1]
 * ```
 */
export function unitFactor(variables: readonly string[], cardinality: readonly number[]): Factor {
  return logFactor(variables, cardinality, new Float64Array(size(cardinality)))
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

/** The cardinality of each variable of a list, from the factors that mention them. */
function unionScope(a: Factor, b: Factor, name: string): { variables: string[]; cardinality: number[] } {
  const variables = [...a.variables, ...b.variables.filter((v) => !a.variables.includes(v))]
  const cardinality = variables.map((v) => {
    const ia = a.variables.indexOf(v)
    const ib = b.variables.indexOf(v)
    if (ia >= 0 && ib >= 0 && a.cardinality[ia] !== b.cardinality[ib]) {
      throw new WhiteboxError('value', `${name}: ${v} has cardinality ${a.cardinality[ia]} and ${b.cardinality[ib]}`)
    }
    return (ia >= 0 ? a.cardinality[ia] : b.cardinality[ib]) ?? 1
  })
  return { variables, cardinality }
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
  const { variables, cardinality } = unionScope(a, b, 'factorProduct')
  const out = new Float64Array(size(cardinality))
  const av = a.logValues
  const bv = b.logValues
  odometer(cardinality, [aligned(a, variables), aligned(b, variables)], (i, idx) => {
    out[i] = (av[idx[0] ?? 0] ?? 0) + (bv[idx[1] ?? 0] ?? 0)
  })
  return { variables, cardinality, logValues: out }
}

/**
 * The product of a list of factors (the unit factor over no variables for an empty list).
 *
 * @param factors - The factors.
 * @returns Their product.
 * @example
 * ```ts
 * import { factor, factorProductAll, factorValues } from 'whitebox-engine/pgm'
 * factorValues(factorProductAll([factor(['a'], [2], [1, 2]), factor(['a'], [2], [3, 4])])) // => [3, 8]
 * ```
 */
export function factorProductAll(factors: readonly Factor[]): Factor {
  if (factors.length === 0) return unitFactor([], [])
  return factors.reduce((a, b) => factorProduct(a, b))
}

/**
 * The quotient of two factors, as used by HUGIN propagation and belief updates: `a / b`, where `b`'s variables are a
 * subset of `a`'s, with the convention 0 / 0 = 0.
 *
 * @param a - The numerator.
 * @param b - The denominator.
 * @returns The quotient over a's variables.
 * @throws WhiteboxError with code `value` when `b` mentions a variable `a` does not, or divides a non-zero by zero.
 * @concept factors
 * @example
 * ```ts
 * import { factor, factorDivide, factorValues } from 'whitebox-engine/pgm'
 * factorValues(factorDivide(factor(['a', 'b'], [2, 2], [2, 4, 0, 8]), factor(['a'], [2], [2, 0]))) // => [1, 2, 0, Infinity]
 * ```
 */
export function factorDivide(a: Factor, b: Factor): Factor {
  for (const v of b.variables) {
    if (!a.variables.includes(v)) throw new WhiteboxError('value', `factorDivide: ${v} is not in the numerator`)
  }
  unionScope(a, b, 'factorDivide')
  const out = new Float64Array(a.logValues.length)
  const av = a.logValues
  const bv = b.logValues
  odometer(a.cardinality, [aligned(b, a.variables)], (i, idx) => {
    const x = av[i] ?? NEG
    const y = bv[idx[0] ?? 0] ?? NEG
    out[i] = x === NEG ? NEG : y === NEG ? Number.POSITIVE_INFINITY : x - y
  })
  return { variables: [...a.variables], cardinality: [...a.cardinality], logValues: out }
}

/**
 * A factor raised to a power (its log values scaled), as used by fractional, power-EP and tree-reweighted updates.
 *
 * @param f - The factor.
 * @param power - The exponent.
 * @returns f^power (0^0 is taken as 1 only for power 0).
 * @example
 * ```ts
 * import { factor, factorPower, factorValues } from 'whitebox-engine/pgm'
 * factorValues(factorPower(factor(['a'], [2], [4, 9]), 0.5)) // => [2, 3]
 * ```
 */
export function factorPower(f: Factor, power: number): Factor {
  if (power === 0) return unitFactor(f.variables, f.cardinality)
  return { ...f, logValues: f.logValues.map((v) => (v === NEG ? NEG : v * power)) }
}

/** Eliminate variables by summing (log-sum-exp) or maximising over their states. */
function eliminate(f: Factor, variables: readonly string[], mode: 'sum' | 'max', name: string): Factor {
  for (const v of variables)
    if (!f.variables.includes(v)) throw new WhiteboxError('value', `${name}: ${v} is not in the factor`)
  const keep = f.variables.filter((v) => !variables.includes(v))
  const card = keep.map((v) => f.cardinality[f.variables.indexOf(v)] ?? 1)
  const s = strides(card)
  const out: Factor = { variables: keep, cardinality: card, logValues: new Float64Array(size(card)) }
  const into = f.variables.map((v) => {
    const k = keep.indexOf(v)
    return k < 0 ? 0 : (s[k] ?? 0)
  })
  const fv = f.logValues
  const best = new Float64Array(out.logValues.length).fill(NEG)
  odometer(f.cardinality, [into], (i, idx) => {
    const j = idx[0] ?? 0
    const v = fv[i] ?? NEG
    if (v > (best[j] ?? NEG) || Number.isNaN(v)) best[j] = v
  })
  if (mode === 'max') return { ...out, logValues: best }
  const acc = new Float64Array(out.logValues.length)
  odometer(f.cardinality, [into], (i, idx) => {
    const j = idx[0] ?? 0
    const m = best[j] ?? NEG
    if (Number.isFinite(m)) acc[j] = (acc[j] ?? 0) + Math.exp((fv[i] ?? NEG) - m)
  })
  return { ...out, logValues: best.map((m, j) => (Number.isFinite(m) ? m + Math.log(acc[j] ?? 0) : m)) }
}

/**
 * Sum variables out of a factor.
 *
 * @param f - The factor.
 * @param variables - Variables to sum out.
 * @returns The marginal factor.
 * @throws WhiteboxError with code `value` for a variable not in the factor.
 * @concept factors
 * @example
 * ```ts
 * import { factor, marginalise, factorValues } from 'whitebox-engine/pgm'
 * factorValues(marginalise(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), ['b'])) // => [3, 7]
 * ```
 */
export function marginalise(f: Factor, variables: readonly string[]): Factor {
  return eliminate(f, variables, 'sum', 'marginalise')
}

/**
 * Maximise variables out of a factor (max-product elimination).
 *
 * @param f - The factor.
 * @param variables - Variables to maximise out.
 * @returns The max-marginal factor.
 * @throws WhiteboxError with code `value` for a variable not in the factor.
 * @example
 * ```ts
 * import { factor, maxMarginalise, factorValues } from 'whitebox-engine/pgm'
 * factorValues(maxMarginalise(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), ['b'])) // => [2, 4]
 * ```
 */
export function maxMarginalise(f: Factor, variables: readonly string[]): Factor {
  return eliminate(f, variables, 'max', 'maxMarginalise')
}

/**
 * The same factor with its variables in a given order (a transpose of the table).
 *
 * @param f - The factor.
 * @param variables - A permutation of its variables.
 * @returns The permuted factor.
 * @throws WhiteboxError with code `value` when `variables` is not a permutation of the factor's variables.
 * @example
 * ```ts
 * import { factor, permuteFactor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(permuteFactor(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), ['b', 'a'])) // => [1, 3, 2, 4]
 * ```
 */
export function permuteFactor(f: Factor, variables: readonly string[]): Factor {
  if (variables.length !== f.variables.length || variables.some((v) => !f.variables.includes(v))) {
    throw new WhiteboxError('value', `permuteFactor: [${variables.join(', ')}] is not a permutation of the scope`)
  }
  if (f.variables.every((v, i) => v === variables[i])) return f
  const card = variables.map((v) => f.cardinality[f.variables.indexOf(v)] ?? 1)
  const out = new Float64Array(f.logValues.length)
  const fv = f.logValues
  odometer(card, [aligned(f, variables)], (i, idx) => {
    out[i] = fv[idx[0] ?? 0] ?? NEG
  })
  return { variables: [...variables], cardinality: card, logValues: out }
}

/**
 * The marginal of a factor over some of its variables, in the order given (sum or max over the rest).
 *
 * @param f - The factor.
 * @param variables - The variables to keep, in the order wanted.
 * @param mode - Sum (default) or max marginal.
 * @returns The marginal factor over exactly `variables`.
 * @throws WhiteboxError with code `value` for a variable not in the factor.
 * @example
 * ```ts
 * import { factor, factorMarginal, factorValues } from 'whitebox-engine/pgm'
 * factorValues(factorMarginal(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), ['b'])) // => [4, 6]
 * ```
 */
export function factorMarginal(f: Factor, variables: readonly string[], mode: 'sum' | 'max' = 'sum'): Factor {
  for (const v of variables) {
    if (!f.variables.includes(v)) throw new WhiteboxError('value', `factorMarginal: ${v} is not in the factor`)
  }
  const rest = f.variables.filter((v) => !variables.includes(v))
  return permuteFactor(eliminate(f, rest, mode, 'factorMarginal'), variables)
}

/**
 * Condition a factor on evidence: keep the entries consistent with the observed states and drop those variables.
 *
 * @param f - The factor.
 * @param evidence - Observed states (indices) by variable; variables not in the factor are ignored.
 * @returns The reduced factor.
 * @throws WhiteboxError with code `value` for a state index out of range.
 * @example
 * ```ts
 * import { factor, reduceFactor, factorValues } from 'whitebox-engine/pgm'
 * factorValues(reduceFactor(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), { a: 1 })) // => [3, 4]
 * ```
 */
export function reduceFactor(f: Factor, evidence: Readonly<Record<string, number>>): Factor {
  const fixed = f.variables.filter((v) => v in evidence)
  if (fixed.length === 0) return f
  const s = strides(f.cardinality)
  let offset = 0
  for (const v of fixed) {
    const k = f.variables.indexOf(v)
    const state = evidence[v] ?? 0
    if (!(state >= 0 && state < (f.cardinality[k] ?? 0)) || !Number.isInteger(state)) {
      throw new WhiteboxError('value', `reduceFactor: ${v} has no state ${state}`)
    }
    offset += state * (s[k] ?? 0)
  }
  const keep = f.variables.filter((v) => !(v in evidence))
  const card = keep.map((v) => f.cardinality[f.variables.indexOf(v)] ?? 1)
  const out = new Float64Array(size(card))
  const fv = f.logValues
  odometer(card, [aligned(f, keep)], (i, idx) => {
    out[i] = fv[offset + (idx[0] ?? 0)] ?? NEG
  })
  return { variables: keep, cardinality: card, logValues: out }
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
  const total = logSumExpOf(f.logValues)
  if (total === NEG) {
    throw new WhiteboxError('value', 'normaliseFactor: the factor is zero everywhere (inconsistent evidence?)')
  }
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
  return logSumExpOf(f.logValues)
}

/** The flat index of an assignment. */
function flatIndex(f: Factor, assignment: Readonly<Record<string, number>>): number {
  let i = 0
  for (let k = 0; k < f.variables.length; k++) {
    i = i * (f.cardinality[k] ?? 1) + (assignment[f.variables[k] ?? ''] ?? 0)
  }
  return i
}

/**
 * The value of a factor at a full assignment.
 *
 * @param f - The factor.
 * @param assignment - A state index for each of the factor's variables (others are ignored).
 * @returns The value there.
 * @example
 * ```ts
 * import { factor, factorAt } from 'whitebox-engine/pgm'
 * factorAt(factor(['a', 'b'], [2, 2], [1, 2, 3, 4]), { a: 1, b: 0 }) // => 3
 * ```
 */
export function factorAt(f: Factor, assignment: Readonly<Record<string, number>>): number {
  return Math.exp(logFactorAt(f, assignment))
}

/**
 * The log value of a factor at a full assignment.
 *
 * @param f - The factor.
 * @param assignment - A state index for each of the factor's variables (others are ignored).
 * @returns The log value there.
 * @example
 * ```ts
 * import { factor, logFactorAt } from 'whitebox-engine/pgm'
 * logFactorAt(factor(['a'], [2], [1, 0]), { a: 1 }) // => -Infinity
 * ```
 */
export function logFactorAt(f: Factor, assignment: Readonly<Record<string, number>>): number {
  return f.logValues[flatIndex(f, assignment)] ?? NEG
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
  let best = 0
  for (let i = 1; i < f.logValues.length; i++) if ((f.logValues[i] ?? NEG) > (f.logValues[best] ?? NEG)) best = i
  return unflatten(f.variables, f.cardinality, best)
}

/** The assignment at a flat row-major index. */
function unflatten(
  variables: readonly string[],
  cardinality: readonly number[],
  index: number,
): Record<string, number> {
  const out: Record<string, number> = {}
  let rest = index
  for (let k = variables.length - 1; k >= 0; k--) {
    const c = cardinality[k] ?? 1
    out[variables[k] ?? ''] = rest % c
    rest = Math.floor(rest / c)
  }
  return Object.fromEntries(variables.map((v) => [v, out[v] ?? 0]))
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
  return Array.from({ length: size(cardinality) }, (_, i) => unflatten(variables, cardinality, i))
}

/**
 * The entropy (in nats) of a normalised factor, as a joint distribution over its variables.
 *
 * @param f - A normalised factor.
 * @returns -sum p log p.
 * @example
 * ```ts
 * import { factor, factorEntropy } from 'whitebox-engine/pgm'
 * factorEntropy(factor(['a'], [2], [0.5, 0.5])) === Math.LN2 // => true
 * ```
 */
export function factorEntropy(f: Factor): number {
  let h = 0
  for (const l of f.logValues) if (l > NEG) h -= Math.exp(l) * l
  return h
}
