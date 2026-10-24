import { WhiteboxError } from '../base/index.ts'
import type { Csp } from './csp.ts'

/** A compiled constraint: its scope as variable indices. */
export interface Compiled<D> {
  readonly scope: readonly number[]
  readonly test: (values: readonly D[]) => boolean
  readonly index: number
}

/** A CSP over variable indices and value indices, with each variable's constraints and neighbours. */
export interface Model<D> {
  readonly n: number
  readonly names: readonly string[]
  readonly values: readonly (readonly D[])[]
  readonly constraints: readonly Compiled<D>[]
  /** The constraints on each variable. */
  readonly on: readonly (readonly Compiled<D>[])[]
  /** The variables sharing a constraint with each variable, in variable order. */
  readonly neighbours: readonly (readonly number[])[]
  /** The number of constraint checks made through `check`. */
  checks: number
}

/** Compile a CSP, validating names and domains. */
export function compile<D>(csp: Csp<D>, name: string): Model<D> {
  const index = new Map<string, number>()
  csp.variables.forEach((v, i) => {
    if (index.has(v)) throw new WhiteboxError('value', `${name}: variable ${v} is repeated`)
    index.set(v, i)
  })
  const values = csp.variables.map((v) => {
    const d = csp.domains[v]
    if (d === undefined) throw new WhiteboxError('value', `${name}: variable ${v} has no domain`)
    return d
  })
  const constraints = csp.constraints.map((c, k) => ({
    scope: c.scope.map((v) => {
      const i = index.get(v)
      if (i === undefined) throw new WhiteboxError('value', `${name}: constraint on unknown variable ${v}`)
      return i
    }),
    test: c.test,
    index: k,
  }))
  const on: Compiled<D>[][] = csp.variables.map(() => [])
  const near: Set<number>[] = csp.variables.map(() => new Set())
  for (const c of constraints) {
    for (const i of new Set(c.scope)) {
      on[i]?.push(c)
      for (const j of c.scope) if (j !== i) near[i]?.add(j)
    }
  }
  return {
    n: csp.variables.length,
    names: csp.variables,
    values,
    constraints,
    on,
    neighbours: near.map((s) => [...s].sort((a, b) => a - b)),
    checks: 0,
  }
}

/** Test a constraint on a full tuple of value indices for its scope, counting the check. */
export function check<D>(m: Model<D>, c: Compiled<D>, value: (variable: number) => number): boolean {
  m.checks += 1
  return c.test(c.scope.map((v) => m.values[v]?.[value(v)] as D))
}

/** The number of constraints on `variable` violated when it takes `value` given the assigned variables. */
export function conflictsOf<D>(m: Model<D>, assigned: Int32Array, variable: number, value: number): number {
  let count = 0
  const at = (v: number): number => (v === variable ? value : (assigned[v] ?? -1))
  for (const c of m.on[variable] ?? []) {
    if (c.scope.every((v) => v === variable || (assigned[v] ?? -1) >= 0) && !check(m, c, at)) count += 1
  }
  return count
}
