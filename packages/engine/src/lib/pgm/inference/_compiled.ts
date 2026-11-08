// A discrete model compiled for message passing and sampling: evidence applied, variables and factors indexed, and the
// variable-factor edges listed. Shared by belief propagation, the variational methods, EP, Gibbs sampling and MAP
// (not re-exported).
import { OmaiError } from '../../base/index.ts'
import { type Factor, logFactor, reduceFactor } from '../factor.ts'
import { type DiscreteModel, evidenceIndices, modelFactors, modelVariables } from '../models/factor-graph.ts'

/** One factor of a compiled model. */
export interface CompiledFactor {
  /** Variable indices of the scope. */
  readonly vars: readonly number[]
  readonly logValues: Float64Array
  /** Row-major state digits of every entry: `digits[entry * vars.length + k]`. */
  readonly digits: Int32Array
  /** Row-major strides of the scope. */
  readonly strides: readonly number[]
}

/** A compiled model. */
export interface Compiled {
  /** Unobserved variables. */
  readonly names: readonly string[]
  readonly card: readonly number[]
  readonly index: ReadonlyMap<string, number>
  readonly factors: readonly CompiledFactor[]
  /** Edges (factor, variable, position of the variable in the factor's scope). */
  readonly edges: readonly { readonly factor: number; readonly variable: number; readonly position: number }[]
  /** Edge ids of each factor, in scope order. */
  readonly factorEdges: readonly (readonly number[])[]
  /** Edge ids of each variable. */
  readonly variableEdges: readonly (readonly number[])[]
  /** The sum of the log values of factors left with no variables by the evidence. */
  readonly constant: number
  readonly evidence: Readonly<Record<string, number>>
  /** All model variables with their states (for reporting observed ones). */
  readonly all: readonly { readonly name: string; readonly states: readonly string[] }[]
}

/** Compile a model under evidence. */
export function compile(model: DiscreteModel, evidence: Readonly<Record<string, string | number>> = {}): Compiled {
  const ev = evidenceIndices(model, evidence)
  const all = modelVariables(model)
  const free = all.filter((v) => !(v.name in ev))
  const names = free.map((v) => v.name)
  const card = free.map((v) => v.states.length)
  const index = new Map(names.map((n, i) => [n, i]))
  const factors: CompiledFactor[] = []
  let constant = 0
  for (const f of modelFactors(model)) {
    const r = reduceFactor(f, ev)
    if (r.variables.length === 0) {
      constant += r.logValues[0] ?? 0
      continue
    }
    factors.push(compileFactor(r, index))
  }
  const edges: { factor: number; variable: number; position: number }[] = []
  const factorEdges: number[][] = factors.map(() => [])
  const variableEdges: number[][] = names.map(() => [])
  for (const [a, f] of factors.entries()) {
    for (const [k, i] of f.vars.entries()) {
      factorEdges[a]?.push(edges.length)
      variableEdges[i]?.push(edges.length)
      edges.push({ factor: a, variable: i, position: k })
    }
  }
  return { names, card, index, factors, edges, factorEdges, variableEdges, constant, evidence: ev, all }
}

/** Compile one factor over indexed variables. */
export function compileFactor(f: Factor, index: ReadonlyMap<string, number>): CompiledFactor {
  const vars = f.variables.map((v) => {
    const i = index.get(v)
    if (i === undefined) throw new OmaiError('value', `unknown variable ${v}`)
    return i
  })
  const d = vars.length
  const n = f.logValues.length
  const digits = new Int32Array(n * d)
  const ctr = new Int32Array(d)
  for (let e = 0; e < n; e++) {
    for (let k = 0; k < d; k++) digits[e * d + k] = ctr[k] ?? 0
    for (let k = d - 1; k >= 0; k--) {
      ctr[k] = (ctr[k] ?? 0) + 1
      if ((ctr[k] ?? 0) < (f.cardinality[k] ?? 1)) break
      ctr[k] = 0
    }
  }
  const strides = new Array<number>(d).fill(1)
  for (let k = d - 2; k >= 0; k--) strides[k] = (strides[k + 1] ?? 1) * (f.cardinality[k + 1] ?? 1)
  return { vars, logValues: f.logValues, digits, strides }
}

/** Shift log values so their log-sum-exp (sum) or maximum (max) is zero; returns the shift. Throws if all are -inf. */
export function normaliseLog(m: Float64Array, mode: 'sum' | 'max' = 'sum'): number {
  let max = Number.NEGATIVE_INFINITY
  for (const v of m) if (v > max) max = v
  if (max === Number.NEGATIVE_INFINITY) {
    throw new OmaiError('value', 'message passing: a message is zero everywhere (inconsistent evidence?)')
  }
  let shift = max
  if (mode === 'sum') {
    let s = 0
    for (const v of m) s += Math.exp(v - max)
    shift = max + Math.log(s)
  }
  for (let i = 0; i < m.length; i++) m[i] = (m[i] ?? 0) - shift
  return shift
}

/**
 * Project a factor table plus incoming log messages onto one of its variables: out[x_k] = reduce over the other
 * variables of (power * logf + sum_{j != k} incoming_j[x_j]), with log-sum-exp (sum) or max. `incoming[j]` is null
 * for position k (or for positions to ignore).
 */
export function projectFactor(
  f: CompiledFactor,
  card: number,
  k: number,
  incoming: readonly (Float64Array | null)[],
  mode: 'sum' | 'max',
  power = 1,
): Float64Array {
  const d = f.vars.length
  const n = f.logValues.length
  const vals = new Float64Array(n)
  const best = new Float64Array(card).fill(Number.NEGATIVE_INFINITY)
  for (let e = 0; e < n; e++) {
    const lf = f.logValues[e] ?? Number.NEGATIVE_INFINITY
    let s = lf === Number.NEGATIVE_INFINITY ? lf : power * lf
    for (let j = 0; j < d && s > Number.NEGATIVE_INFINITY; j++) {
      if (j === k) continue
      const m = incoming[j]
      if (m !== null && m !== undefined) s += m[f.digits[e * d + j] ?? 0] ?? 0
    }
    vals[e] = s
    const x = f.digits[e * d + k] ?? 0
    if (s > (best[x] ?? Number.NEGATIVE_INFINITY)) best[x] = s
  }
  if (mode === 'max') return best
  const acc = new Float64Array(card)
  for (let e = 0; e < n; e++) {
    const x = f.digits[e * d + k] ?? 0
    const b = best[x] ?? Number.NEGATIVE_INFINITY
    if (b > Number.NEGATIVE_INFINITY) acc[x] = (acc[x] ?? 0) + Math.exp((vals[e] ?? 0) - b)
  }
  return best.map((b, x) => (b > Number.NEGATIVE_INFINITY ? b + Math.log(acc[x] ?? 0) : b))
}

/** The largest absolute difference between two normalised log messages, in probability space. */
export function messageChange(a: Float64Array, b: Float64Array): number {
  let d = 0
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(Math.exp(a[i] ?? 0) - Math.exp(b[i] ?? 0)))
  return d
}

/** Damp a new log message towards the old one in probability space: (1 - damping) new + damping old, renormalised. */
export function damp(next: Float64Array, old: Float64Array, damping: number, mode: 'sum' | 'max'): Float64Array {
  if (damping <= 0) return next
  const out = next.map((v, i) => {
    const a = (1 - damping) * Math.exp(v)
    const b = damping * Math.exp(old[i] ?? 0)
    return Math.log(a + b)
  })
  normaliseLog(out, mode)
  return out
}

/** A single-variable factor from log values (normalised). */
export function marginalFactor(name: string, logs: Float64Array): Factor {
  return logFactor([name], [logs.length], logs)
}

/** One-hot factors for the observed variables. */
export function observedMarginals(c: Compiled): Record<string, Factor> {
  const out: Record<string, Factor> = {}
  for (const v of c.all) {
    const s = c.evidence[v.name]
    if (s === undefined) continue
    out[v.name] = logFactor(
      [v.name],
      [v.states.length],
      v.states.map((_, i) => (i === s ? 0 : Number.NEGATIVE_INFINITY)),
    )
  }
  return out
}

/** The log values of a compiled factor's scope belief: logf + sum of per-position log terms (normalised). */
export function factorBeliefLogs(f: CompiledFactor, incoming: readonly Float64Array[], power = 1): Float64Array {
  const d = f.vars.length
  const out = new Float64Array(f.logValues.length)
  for (let e = 0; e < out.length; e++) {
    const lf = f.logValues[e] ?? Number.NEGATIVE_INFINITY
    let s = lf === Number.NEGATIVE_INFINITY ? lf : power * lf
    for (let j = 0; j < d; j++) s += incoming[j]?.[f.digits[e * d + j] ?? 0] ?? 0
    out[e] = s
  }
  normaliseLog(out)
  return out
}
