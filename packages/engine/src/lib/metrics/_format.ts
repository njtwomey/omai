/** Python's fixed-point formatting `f'{x:.{digits}f}'`: like toFixed, but exact ties round half to even. */
export function fixed(x: number, digits: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf'
  const s = x.toFixed(digits)
  // A tie needs the exact expansion to end in 5 right after the kept digits; doubles that are ties are short.
  const exact = Math.abs(x).toFixed(Math.min(100, digits + 60))
  const point = exact.indexOf('.')
  const rest = exact.slice(point + 1 + digits)
  if (!/^50*$/.test(rest)) return s
  const kept = exact.slice(0, point + 1 + digits).replace(/\.$/, '')
  const last = Number(kept.replace('.', '').slice(-1))
  // toFixed rounded away from zero; keep the truncated value when its last digit is already even.
  if (last % 2 === 0) return (x < 0 ? '-' : '') + kept
  return s
}

/** Right-align `s` in a field of `width` characters (Python's `{:>width}`). */
export function right(s: string, width: number): string {
  return s.length >= width ? s : ' '.repeat(width - s.length) + s
}
