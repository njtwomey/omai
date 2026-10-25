// Coding tabular data for the learners (not re-exported): discrete columns as state indices (-1 for missing) with their
// state names, and continuous columns as numbers.
import { MdmlError } from '../../base/index.ts'

/** One row of data: a value per variable; null or undefined (or a missing key) is missing. */
export type DataRow = Readonly<Record<string, string | number | null | undefined>>

/** Discrete data coded as state indices. */
export interface Coded {
  readonly names: readonly string[]
  readonly states: Readonly<Record<string, readonly string[]>>
  /** codes[v][row], -1 when missing. */
  readonly codes: Readonly<Record<string, Int32Array>>
  readonly n: number
  /** Row weights (default 1). */
  readonly weights: Float64Array
}

const isMissing = (v: unknown): boolean => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))

/** The observed states of a column, sorted numerically when they are all numbers and lexicographically otherwise. */
export function observedStates(rows: readonly DataRow[], v: string): string[] {
  const values = rows.map((r) => r[v]).filter((x) => !isMissing(x))
  const numeric = values.every((x) => typeof x === 'number')
  const unique = [...new Set(values.map(String))]
  return numeric ? unique.sort((a, b) => Number(a) - Number(b)) : unique.sort()
}

/** Code discrete data by state names (values are matched as strings). */
export function code(
  rows: readonly DataRow[],
  names: readonly string[],
  states: Readonly<Record<string, readonly string[]>> = {},
  weights?: readonly number[],
): Coded {
  const n = rows.length
  if (weights !== undefined && weights.length !== n) throw new MdmlError('shape', 'one weight per row')
  const resolved: Record<string, readonly string[]> = {}
  const codes: Record<string, Int32Array> = {}
  for (const v of names) {
    const s = states[v] ?? observedStates(rows, v)
    resolved[v] = s
    const index = new Map(s.map((x, i) => [x, i]))
    const col = new Int32Array(n)
    for (const [i, r] of rows.entries()) {
      const x = r[v]
      if (isMissing(x)) {
        col[i] = -1
        continue
      }
      const k = index.get(String(x))
      if (k === undefined) throw new MdmlError('value', `data: ${v} has state ${String(x)} outside [${s.join(', ')}]`)
      col[i] = k
    }
    codes[v] = col
  }
  return {
    names: [...names],
    states: resolved,
    codes,
    n,
    weights: Float64Array.from(weights ?? new Array<number>(n).fill(1)),
  }
}

/**
 * Weighted counts N[j][k] of a variable's states k under each joint parent configuration j (the first parent varies
 * slowest); rows with a missing value in the family are skipped.
 */
export function familyCounts(d: Coded, variable: string, parents: readonly string[]): number[][] {
  const r = d.states[variable]?.length ?? 0
  const card = parents.map((p) => d.states[p]?.length ?? 0)
  const q = card.reduce((a, b) => a * b, 1)
  const out = Array.from({ length: q }, () => new Array<number>(r).fill(0))
  const x = d.codes[variable] ?? new Int32Array(0)
  const cols = parents.map((p) => d.codes[p] ?? new Int32Array(0))
  for (let i = 0; i < d.n; i++) {
    const k = x[i] ?? -1
    if (k < 0) continue
    let j = 0
    let ok = true
    for (const [m, col] of cols.entries()) {
      const s = col[i] ?? -1
      if (s < 0) {
        ok = false
        break
      }
      j = j * (card[m] ?? 1) + s
    }
    if (!ok) continue
    const row = out[j] as number[]
    row[k] = (row[k] ?? 0) + (d.weights[i] ?? 1)
  }
  return out
}

/** A continuous data matrix: one column per variable (missing values are not allowed). */
export function numericColumns(rows: readonly DataRow[], names: readonly string[]): Float64Array[] {
  return names.map((v) =>
    Float64Array.from(rows, (r) => {
      const x = r[v]
      if (typeof x !== 'number' || Number.isNaN(x))
        throw new MdmlError('value', `data: ${v} needs a number in every row`)
      return x
    }),
  )
}

/** The variables of a data set (the keys of its first rows, in first-seen order). */
export function dataVariables(rows: readonly DataRow[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) {
        seen.add(k)
        out.push(k)
      }
    }
  }
  return out
}
