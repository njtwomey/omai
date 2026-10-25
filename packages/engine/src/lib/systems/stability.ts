/**
 * Poles, zeros and stability of linear systems: poles and (invariant) zeros, the test that every pole lies in the
 * stable region, the Routh-Hurwitz array for continuous-time polynomials and the Jury table for discrete-time ones, and
 * the root locus of a feedback loop, the paths of the closed-loop poles as the gain varies (python-control root_locus).
 */

import { WhiteboxError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { chebyshevNodes } from '../numerics/interpolate/index.ts'
import { det, eigvals, norm } from '../numerics/linalg/index.ts'
import { type ComplexArray, orthofit, orthoToPower, polyder, polymul } from '../numerics/polynomial/index.ts'
import { concat, eye, mul, primalOf, sub, type TensorLike, toFlat } from '../tensor/index.ts'
import { evalAt, float, fromList, numbers, polyAddHf, rootsOf, toList, trimLeading } from './_shared.ts'
import { isDiscrete, type Lti, toTf, toZpk } from './representations.ts'

const sorted = (z: ComplexArray): ComplexArray => {
  const pairs = z.re.map((re, i) => [re, z.im[i] ?? 0] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return { re: pairs.map((p) => p[0]), im: pairs.map((p) => p[1]) }
}

/**
 * The poles of a system (python-control poles): eigenvalues of A, or roots of the denominator, sorted by real then
 * imaginary part.
 *
 * @param sys - The system.
 * @returns The poles.
 * @concept poles-and-zeros
 * @example
 * ```ts
 * import { poles, tf } from 'whitebox-engine/systems'
 * poles(tf([1], [1, 3, 2])).re // => [-2, -1]
 * ```
 */
export function poles(sys: Lti): ComplexArray {
  switch (sys.kind) {
    case 'ss': {
      if ((sys.a.shape[0] ?? 0) === 0) return { re: [], im: [] }
      const { re, im } = eigvals(primalOf(sys.a))
      return sorted({ re: toFlat(re), im: toFlat(im) })
    }
    case 'tf':
      return rootsOf(numbers(sys.den))
    default:
      return sorted(toZpk(sys).poles)
  }
}

/**
 * The zeros of a system: the numerator roots of a single-channel system (as scipy's ss2zpk), and for a square
 * multivariable state-space system the invariant zeros, the roots of det [[sI - A, -B], [C, D]] (found by Chebyshev
 * interpolation of the determinant), sorted by real then imaginary part.
 *
 * @param sys - The system.
 * @returns The zeros.
 * @throws WhiteboxError with code `not-implemented` for a non-square multivariable system.
 * @concept poles-and-zeros
 * @example
 * ```ts
 * import { tf, zeros } from 'whitebox-engine/systems'
 * zeros(tf([1, 3], [1, 3, 2])).re // => [-3]
 * ```
 */
export function zeros(sys: Lti): ComplexArray {
  if (sys.kind !== 'ss') {
    if (sys.kind === 'tf' && sys.num.shape.length === 2) throw new WhiteboxError('shape', 'zeros: one output is needed')
    return sorted(toZpk(sys).zeros)
  }
  const m = sys.b.shape[1] ?? 0
  const p = sys.c.shape[0] ?? 0
  if (m === 1 && p === 1) return sorted(toZpk(sys).zeros)
  if (m !== p) {
    throw new WhiteboxError('not-implemented', 'zeros: invariant zeros of non-square systems are not implemented')
  }
  const n = sys.a.shape[0] ?? 0
  if (n === 0) return { re: [], im: [] }
  // det [[sI - A, -B], [C, D]] is a real polynomial of degree at most n: interpolate it at n + 1 Chebyshev points.
  const scale = Math.max(1, numbers(norm(sys.a)).at(0) ?? 1)
  const nodes = chebyshevNodes(n + 1)
  const values = nodes.map((x) => {
    const s = x * scale
    const top = concat([sub(mul(eye(n), s), sys.a), mul(sys.b, -1)], 1)
    return numbers(det(concat([top, concat([sys.c, sys.d], 1)], 0))).at(0) ?? 0
  })
  const cheb = numbers(orthofit('chebyshev', nodes, values, n))
  const lowest = orthoToPower('chebyshev', cheb)
  const big = lowest.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0)
  const highest = trimLeading([...lowest].reverse(), 1e-9 * big)
  const roots = toList(rootsOf(highest)).map((z) => C.mul(z, scale))
  return sorted(fromList(roots))
}

/**
 * Whether every pole is strictly stable: in the open left half-plane, or inside the unit circle in discrete time.
 *
 * @param sys - The system.
 * @returns `true` for an asymptotically stable system.
 * @concept stability
 * @example
 * ```ts
 * import { isStable, tf } from 'whitebox-engine/systems'
 * isStable(tf([1], [1, -0.5], { dt: 1 })) // => true
 * ```
 */
export function isStable(sys: Lti): boolean {
  const p = toList(poles(sys))
  return isDiscrete(sys) ? p.every((z) => Math.hypot(z.re, z.im) < 1) : p.every((z) => z.re < 0)
}

/** The Routh array and what it says. */
export interface RouthResult {
  /** The rows, s^n first. */
  readonly table: readonly (readonly number[])[]
  /** The first column. */
  readonly firstColumn: readonly number[]
  /** Sign changes in the first column: the number of right half-plane roots. */
  readonly signChanges: number
  /** Rows whose first entry was zero (replaced by a small epsilon). */
  readonly epsilonRows: readonly number[]
  /** Rows that were entirely zero (replaced by the derivative of the auxiliary polynomial): symmetric roots. */
  readonly zeroRows: readonly number[]
  /** All roots in the open left half-plane. */
  readonly stable: boolean
}

/**
 * The Routh-Hurwitz test of a polynomial: the Routh array, with the epsilon rule for a zero first entry and the
 * auxiliary-polynomial rule for a zero row.
 *
 * @param coefficients - The polynomial, highest degree first.
 * @returns The table, the first column, the number of right half-plane roots and whether all roots are stable.
 * @concept routh-hurwitz
 * @example
 * ```ts
 * import { routh } from 'whitebox-engine/systems'
 * routh([1, 2, 3, 10]).signChanges // => 2
 * ```
 */
export function routh(coefficients: TensorLike): RouthResult {
  const c = trimLeading(numbers(float(coefficients)))
  const n = c.length - 1
  const width = Math.floor(n / 2) + 1
  const row = (start: number): number[] => Array.from({ length: width }, (_, j) => c[start + 2 * j] ?? 0)
  const scale = c.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0)
  const eps = 1e-9 * scale
  const table: number[][] = [row(0)]
  if (n >= 1) table.push(row(1))
  const epsilonRows: number[] = []
  const zeroRows: number[] = []
  const fix = (i: number): void => {
    const r = table[i] as number[]
    if (r.every((v) => Math.abs(v) <= 1e-12 * scale)) {
      // Auxiliary polynomial from the row above (degree n - i + 1, even steps): replace by its derivative.
      zeroRows.push(i)
      const above = table[i - 1] as number[]
      const degree = n - i + 1
      table[i] = above.map((v, j) => v * (degree - 2 * j)).map((v) => (Number.isFinite(v) ? v : 0))
    }
    if (Math.abs((table[i] as number[])[0] ?? 0) <= 1e-12 * scale) {
      epsilonRows.push(i)
      ;(table[i] as number[])[0] = eps
    }
  }
  if (n >= 1) fix(1)
  for (let i = 2; i <= n; i++) {
    const a = table[i - 2] as number[]
    const b = table[i - 1] as number[]
    const b0 = b[0] ?? 0
    table.push(Array.from({ length: width }, (_, j) => (b0 * (a[j + 1] ?? 0) - (a[0] ?? 0) * (b[j + 1] ?? 0)) / b0))
    fix(i)
  }
  const firstColumn = table.map((r) => r[0] ?? 0)
  let signChanges = 0
  for (let i = 1; i < firstColumn.length; i++) {
    if (Math.sign(firstColumn[i] ?? 0) !== Math.sign(firstColumn[i - 1] ?? 0)) signChanges++
  }
  const stable = signChanges === 0 && epsilonRows.length === 0 && zeroRows.length === 0
  return { table, firstColumn, signChanges, epsilonRows, zeroRows, stable }
}

/** One condition of the Jury test. */
export interface JuryCondition {
  readonly description: string
  readonly satisfied: boolean
}

/** The Jury table and its conditions. */
export interface JuryResult {
  /** The rows: each pair is a row of coefficients and its reversal. */
  readonly table: readonly (readonly number[])[]
  readonly conditions: readonly JuryCondition[]
  /** All roots strictly inside the unit circle. */
  readonly stable: boolean
}

/**
 * The Jury stability test of a polynomial in z: P(1) > 0, (-1)^n P(-1) > 0, |a_0| < a_n, then |b_0| > |b_{n-1}| and so
 * on down the table of 2 x 2 determinants (Ogata's form).
 *
 * @param coefficients - The polynomial, highest degree first.
 * @returns The table, the conditions and whether every root is inside the unit circle.
 * @concept jury-test
 * @example
 * ```ts
 * import { jury } from 'whitebox-engine/systems'
 * jury([1, -1.2, 0.5]).stable // => true
 * ```
 */
export function jury(coefficients: TensorLike): JuryResult {
  let a = trimLeading(numbers(float(coefficients)))
  if ((a[0] ?? 0) < 0) a = a.map((v) => -v)
  const n = a.length - 1
  const at = (z: number): number => evalAt(a, C.of(z)).re
  const conditions: JuryCondition[] = [
    { description: 'P(1) > 0', satisfied: at(1) > 0 },
    { description: '(-1)^n P(-1) > 0', satisfied: (n % 2 === 0 ? 1 : -1) * at(-1) > 0 },
  ]
  // Rows in ascending powers: a_0, a_1, ..., a_n.
  let current = [...a].reverse()
  const table: number[][] = [current, [...current].reverse()]
  conditions.push({ description: '|a_0| < a_n', satisfied: Math.abs(current[0] ?? 0) < Math.abs(current[n] ?? 0) })
  while (current.length > 3) {
    const k = current.length - 1
    const next = Array.from(
      { length: k },
      (_, j) => (current[0] ?? 0) * (current[j] ?? 0) - (current[k] ?? 0) * (current[k - j] ?? 0),
    )
    table.push(next, [...next].reverse())
    conditions.push({
      description: `|row ${table.length / 2} first| > |last|`,
      satisfied: Math.abs(next[0] ?? 0) > Math.abs(next[k - 1] ?? 0),
    })
    current = next
  }
  return { table, conditions, stable: n >= 1 && conditions.every((c) => c.satisfied) }
}

/** Root locus data. */
export interface RootLocus {
  readonly gains: readonly number[]
  /** The closed-loop poles per gain, `[gains][poles]`, ordered for continuity along the gains. */
  readonly roots: readonly ComplexArray[]
  /** Centroid and angles (radians) of the asymptotes of the branches that go to infinity. */
  readonly asymptotes: { readonly centroid: number; readonly angles: readonly number[] }
  /** Break-away and break-in points on the real axis, with their (non-negative) gains. */
  readonly breakaway: { readonly points: readonly number[]; readonly gains: readonly number[] }
}

function rootsAt(num: readonly number[], den: readonly number[], k: number): ComplexNumber[] {
  const poly = polyAddHf(
    den,
    num.map((v) => v * k),
  )
  const r = toList(rootsOf(trimLeading(poly)))
  while (r.length < den.length - 1) r.push(C.of(Number.POSITIVE_INFINITY))
  return r
}

/**
 * The root locus of a single-input single-output loop L = N / D (python-control root_locus_map): the roots of
 * D + k N over the gains, sorted along the gains so branches stay continuous, with the asymptotes and the real-axis
 * break points. Default gains: 0 to a maximum past the break points and the asymptotes' reach, then three decades more.
 *
 * @param sys - The open-loop system.
 * @param gains - The gains (default as described).
 * @returns Gains, root rows, asymptotes and break points.
 * @concept root-locus
 * @example
 * ```ts
 * import { rootLocus, tf } from 'whitebox-engine/systems'
 * rootLocus(tf([1], [1, 3, 2]), [0]).roots[0].re // => [-2, -1]
 * ```
 */
export function rootLocus(sys: Lti, gains?: readonly number[]): RootLocus {
  const g = toTf(sys)
  if (g.num.shape.length !== 1) throw new WhiteboxError('shape', 'rootLocus: one input and one output are needed')
  const num = trimLeading(numbers(g.num))
  const den = trimLeading(numbers(g.den))
  const zs = toList(rootsOf(num))
  const ps = toList(rootsOf(den))
  const excess = ps.length - zs.length
  const centroid =
    excess > 0 ? (ps.reduce((s, p) => s + p.re, 0) - zs.reduce((s, z) => s + z.re, 0)) / excess : Number.NaN
  const angles = Array.from({ length: Math.max(excess, 0) }, (_, k) => ((2 * k + 1) * Math.PI) / excess)
  // Break points: real roots of D N' - N D' with k = -D / N >= 0.
  const lowest = (c: readonly number[]): number[] => [...c].reverse()
  const dn = lowest(polyder(lowest(num)))
  const dd = lowest(polyder(lowest(den)))
  const condition = polyAddHf(
    polymul(den, dn.length > 0 ? dn : [0]),
    polymul(num, dd.length > 0 ? dd : [0]).map((v) => -v),
  )
  const points: number[] = []
  const breakGains: number[] = []
  for (const r of toList(rootsOf(trimLeading(condition)))) {
    if (r.im !== 0) continue
    const nv = evalAt(num, r).re
    if (nv === 0) continue
    const k = -evalAt(den, r).re / nv
    if (k >= 0) {
      points.push(r.re)
      breakGains.push(k)
    }
  }
  let ks: number[]
  if (gains !== undefined) ks = [...gains]
  else {
    const lead = (den[0] ?? 1) / (num[0] ?? 1)
    let kmax = Math.abs(lead) * 3
    if (excess > 0) {
      const important = [...zs, ...ps, ...points.map((x) => C.of(x))]
      const reach = 4 * important.reduce((acc, z) => Math.max(acc, Math.hypot(z.re - centroid, z.im)), 0)
      for (const theta of angles) {
        const s = C.of(centroid + reach * Math.cos(theta), reach * Math.sin(theta))
        const ratio = C.div(evalAt(den, s), evalAt(num, s))
        kmax = Math.max(kmax, Math.hypot(ratio.re, ratio.im))
      }
    }
    for (const k of breakGains) kmax = Math.max(kmax, k)
    ks = [
      ...Array.from({ length: 50 }, (_, i) => (kmax * i) / 49),
      ...breakGains,
      kmax * 10,
      kmax * 100,
      kmax * 1000,
    ].sort((x, y) => x - y)
  }
  // Order each row by matching it to the previous one (python-control's _RLSortRoots).
  const rowsOf: ComplexNumber[][] = []
  for (const k of ks) {
    const row = rootsAt(num, den, k).sort((x, y) => x.re - y.re || x.im - y.im)
    const prev = rowsOf[rowsOf.length - 1]
    if (prev === undefined) {
      rowsOf.push(row)
      continue
    }
    const out: ComplexNumber[] = new Array<ComplexNumber>(row.length)
    const available = prev.map((_, i) => i)
    for (const z of row) {
      let best = 0
      let bestDistance = Number.POSITIVE_INFINITY
      available.forEach((idx, j) => {
        const q = prev[idx] as ComplexNumber
        const dist = Math.hypot(z.re - q.re, z.im - q.im)
        if (dist < bestDistance) {
          bestDistance = dist
          best = j
        }
      })
      const [slot] = available.splice(best, 1)
      out[slot ?? 0] = z
    }
    rowsOf.push(out)
  }
  return {
    gains: ks,
    roots: rowsOf.map((r) => fromList(r)),
    asymptotes: { centroid, angles },
    breakaway: { points, gains: breakGains },
  }
}
