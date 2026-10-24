// Regret matching and policy averages shared by the CFR family.

/** Regret matching on cumulative regrets: proportional to the positive part, uniform when none is positive. */
export function regretMatch(r: Float64Array): Float64Array {
  let total = 0
  for (const v of r) if (v > 0) total += v
  return total > 0 ? r.map((v) => (v > 0 ? v / total : 0)) : new Float64Array(r.length).fill(1 / r.length)
}

/** The average policy: the policy sums normalised, uniform where they are zero. */
export function averageOf(sums: readonly Float64Array[]): Float64Array[] {
  return sums.map((s) => {
    let total = 0
    for (const v of s) total += v
    return total > 0 ? s.map((v) => v / total) : new Float64Array(s.length).fill(1 / s.length)
  })
}
