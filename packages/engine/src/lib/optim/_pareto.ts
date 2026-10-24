// Row-level Pareto machinery shared by the multiobjective functions and NSGA-II (not exported from the barrel).

/** Whether a dominates b (minimisation). */
export function dominatesRow(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  let better = false
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0
    const bi = b[i] ?? 0
    if (ai > bi) return false
    if (ai < bi) better = true
  }
  return better
}

/** Non-dominated sorting of rows (shared with NSGA-II), optionally with a custom dominance relation. */
export function sortRows(
  rows: readonly (readonly number[])[],
  better: (i: number, j: number) => boolean = (i, j) => dominatesRow(rows[i] as number[], rows[j] as number[]),
): number[][] {
  const n = rows.length
  const dominatedBy: number[][] = Array.from({ length: n }, () => [])
  const count = new Int32Array(n)
  const fronts: number[][] = [[]]
  for (let p = 0; p < n; p++) {
    for (let q = p + 1; q < n; q++) {
      if (better(p, q)) {
        dominatedBy[p]?.push(q)
        count[q] = (count[q] ?? 0) + 1
      } else if (better(q, p)) {
        dominatedBy[q]?.push(p)
        count[p] = (count[p] ?? 0) + 1
      }
    }
  }
  fronts[0] = Array.from({ length: n }, (_, i) => i).filter((i) => count[i] === 0)
  let k = 0
  while ((fronts[k] ?? []).length > 0) {
    const next: number[] = []
    for (const p of fronts[k] ?? []) {
      for (const q of dominatedBy[p] ?? []) {
        count[q] = (count[q] ?? 0) - 1
        if (count[q] === 0) next.push(q)
      }
    }
    k += 1
    fronts[k] = next.sort((a, b) => a - b)
  }
  fronts.pop()
  return fronts
}

/** Crowding distances of rows (shared with NSGA-II). */
export function crowdingRows(rows: readonly (readonly number[])[]): Float64Array {
  const n = rows.length
  const out = new Float64Array(n)
  if (n === 0) return out
  const m = rows[0]?.length ?? 0
  for (let k = 0; k < m; k++) {
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (rows[a]?.[k] ?? 0) - (rows[b]?.[k] ?? 0) || a - b,
    )
    const lo = rows[order[0] ?? 0]?.[k] ?? 0
    const hi = rows[order[n - 1] ?? 0]?.[k] ?? 0
    out[order[0] ?? 0] = Number.POSITIVE_INFINITY
    out[order[n - 1] ?? 0] = Number.POSITIVE_INFINITY
    if (hi === lo) continue
    for (let i = 1; i < n - 1; i++) {
      const at = order[i] ?? 0
      out[at] = (out[at] ?? 0) + ((rows[order[i + 1] ?? 0]?.[k] ?? 0) - (rows[order[i - 1] ?? 0]?.[k] ?? 0)) / (hi - lo)
    }
  }
  return out
}

/** The hypervolume of rows that all strictly dominate the reference point (WFG recursion; a sweep in two dimensions). */
export function hv(rows: number[][], ref: readonly number[]): number {
  const n = rows.length
  if (n === 0) return 0
  const m = ref.length
  if (m === 1) {
    let best = Number.POSITIVE_INFINITY
    for (const r of rows) best = Math.min(best, r[0] ?? 0)
    return (ref[0] ?? 0) - best
  }
  if (n === 1) {
    let v = 1
    for (let k = 0; k < m; k++) v *= (ref[k] ?? 0) - (rows[0]?.[k] ?? 0)
    return v
  }
  if (m === 2) {
    const sorted = [...rows].sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0) || (a[1] ?? 0) - (b[1] ?? 0))
    let last = ref[1] ?? 0
    let area = 0
    for (const p of sorted) {
      if ((p[1] ?? 0) < last) {
        area += ((ref[0] ?? 0) - (p[0] ?? 0)) * (last - (p[1] ?? 0))
        last = p[1] ?? 0
      }
    }
    return area
  }
  // WFG: the sum of exclusive hypervolumes, points sorted worst-first in the last objective.
  const sorted = [...rows].sort((a, b) => (b[m - 1] ?? 0) - (a[m - 1] ?? 0))
  let total = 0
  for (let k = 0; k < sorted.length; k++) {
    const p = sorted[k] as number[]
    let incl = 1
    for (let j = 0; j < m; j++) incl *= (ref[j] ?? 0) - (p[j] ?? 0)
    const limited = sorted.slice(k + 1).map((q) => q.map((v, j) => Math.max(v, p[j] ?? 0)))
    const kept = limited.filter(
      (q, i) => !limited.some((r, j) => j !== i && (dominatesRow(r, q) || (j < i && r.every((v, d) => v === q[d])))),
    )
    total += incl - hv(kept, ref)
  }
  return total
}
