/**
 * Partial fractions of rational functions, scipy.signal style: `residue` expands b(s)/a(s) into poles, residues and a
 * direct term (with repeated poles), `residuez` does the same for discrete-time transfer functions in z^-1, and
 * `invres` rebuilds the rational function from its expansion.
 */

import { type Complex, cabs, cadd, cdiv, cmul, csub, MdmlError } from '../../base/index.ts'
import { polydiv, polyroots } from './power.ts'

// Rational functions follow scipy.signal: coefficients highest degree first (in s for residue, in z^-1 for residuez).

/** Complex values as parallel real and imaginary parts. */
export interface ComplexArray {
  readonly re: number[]
  readonly im: number[]
}

const split = (xs: readonly Complex[]): ComplexArray => ({ re: xs.map((x) => x[0]), im: xs.map((x) => x[1]) })

/** Multiply a highest-first complex polynomial by (s - p). */
function timesLinear(poly: readonly Complex[], p: Complex): Complex[] {
  const out: Complex[] = [...poly, [0, 0]]
  for (let i = poly.length; i >= 1; i--) out[i] = csub(out[i] as Complex, cmul(poly[i - 1] as Complex, p))
  return out
}

/** Taylor coefficients of a highest-first polynomial about p (lowest order first). */
function taylor(poly: readonly Complex[], p: Complex): Complex[] {
  // Repeated synthetic division by (s - p): each remainder is the next Taylor coefficient.
  let current = [...poly]
  const out: Complex[] = []
  while (current.length > 0) {
    const quotient: Complex[] = []
    let acc: Complex = [0, 0]
    for (const c of current) {
      acc = cadd(cmul(acc, p), c)
      quotient.push(acc)
    }
    out.push(quotient.pop() as Complex)
    current = quotient
  }
  return out
}

/** The first m coefficients of the power series of num / den, both given as Taylor coefficients (den[0] != 0). */
function seriesDivide(num: readonly Complex[], den: readonly Complex[], m: number): Complex[] {
  const out: Complex[] = []
  for (let j = 0; j < m; j++) {
    let acc: Complex = num[j] ?? [0, 0]
    for (let i = 1; i <= j; i++) acc = csub(acc, cmul(den[i] ?? [0, 0], out[j - i] as Complex))
    out.push(cdiv(acc, den[0] as Complex))
  }
  return out
}

/** Group poles closer than tol (scipy's unique_roots with averaging). */
function groupPoles(poles: readonly Complex[], tol: number): { unique: Complex[]; multiplicity: number[] } {
  const used = poles.map(() => false)
  const unique: Complex[] = []
  const multiplicity: number[] = []
  poles.forEach((p, i) => {
    if (used[i]) return
    const group = poles.map((q, j) => ({ q, j })).filter(({ q, j }) => !used[j] && cabs(csub(q, p)) <= tol)
    for (const { j } of group) used[j] = true
    unique.push([
      group.reduce((s, g) => s + g.q[0], 0) / group.length,
      group.reduce((s, g) => s + g.q[1], 0) / group.length,
    ])
    multiplicity.push(group.length)
  })
  // scipy orders unique poles by magnitude.
  const order = unique.map((_, i) => i).sort((a, b) => cabs(unique[a] as Complex) - cabs(unique[b] as Complex))
  return { unique: order.map((i) => unique[i] as Complex), multiplicity: order.map((i) => multiplicity[i] as number) }
}

/** Residues of num / prod (s - u)^{m_u} (monic denominator), per pole for powers 1..m. */
function residuesOf(
  numerator: readonly Complex[],
  unique: readonly Complex[],
  multiplicity: readonly number[],
): Complex[] {
  const out: Complex[] = []
  unique.forEach((p, k) => {
    const m = multiplicity[k] ?? 1
    let others: Complex[] = [[1, 0]]
    unique.forEach((u, j) => {
      if (j === k) return
      for (let r = 0; r < (multiplicity[j] ?? 1); r++) others = timesLinear(others, u)
    })
    const h = seriesDivide(taylor(numerator, p), taylor(others, p), m)
    // The coefficient of (s - p)^{-j} is h_{m - j}.
    for (let j = 1; j <= m; j++) out.push(h[m - j] as Complex)
  })
  return out
}

function trimLeading(c: readonly number[]): number[] {
  let i = 0
  while (i < c.length - 1 && c[i] === 0) i++
  return c.slice(i)
}

/** Partial fractions r_i / (s - p_i)^j plus a direct polynomial k. */
export interface PartialFractions {
  /** Residues, one per pole entry (repeated poles appear once per power, lowest power first). */
  readonly residues: ComplexArray
  /** Poles, repeated by multiplicity. */
  readonly poles: ComplexArray
  /** The direct polynomial term (highest first). */
  readonly direct: number[]
}

/**
 * The partial-fraction expansion of b(s) / a(s) (scipy.signal.residue): sum_i r_i / (s - p_i)^j + k(s), with poles
 * closer than tol treated as one repeated pole. Residues of repeated poles come from the Taylor expansion of
 * b / (the rest of a) about the pole.
 *
 * @param b - Numerator, highest degree first.
 * @param a - Denominator, highest degree first.
 * @param tol - Tolerance for repeated poles (default 1e-3).
 * @returns Residues, poles and the direct term.
 * @concept partial-fractions
 * @example
 * ```ts
 * import { residue } from 'mdml-engine/numerics/polynomial'
 * // 1 / (s^2 + 3s + 2) = 1 / (s + 1) - 1 / (s + 2).
 * residue([1], [1, 3, 2]).residues.re.map((v) => Math.round(v * 1e12) / 1e12) // => [1, -1]
 * ```
 */
export function residue(b: readonly number[], a: readonly number[], tol = 1e-3): PartialFractions {
  const bt = trimLeading(b)
  const at = trimLeading(a)
  if (at.length === 0 || at[0] === 0) throw new MdmlError('value', 'residue: the denominator is zero')
  let numerator = bt
  let direct: number[] = []
  if (bt.length >= at.length) {
    const { quotient, remainder } = polydiv([...bt].reverse(), [...at].reverse())
    direct = quotient.reverse()
    numerator = remainder.reverse()
  }
  const roots = polyroots([...at].reverse())
  const poles: Complex[] = roots.re.map((r, i) => [r, roots.im[i] ?? 0])
  const { unique, multiplicity } = groupPoles(poles, tol)
  const lead = at[0] as number
  const rs = residuesOf(
    numerator.map((v): Complex => [v / lead, 0]),
    unique,
    multiplicity,
  )
  const repeated = unique.flatMap((p, k) => new Array<Complex>(multiplicity[k] ?? 1).fill(p))
  return { residues: split(rs), poles: split(repeated), direct }
}

/**
 * The partial-fraction expansion of a discrete-time transfer function in z^-1 (scipy.signal.residuez):
 * b(z) / a(z) = sum_i r_i / (1 - p_i z^-1)^j + k(z^-1).
 *
 * @param b - Numerator coefficients of z^0, z^-1, ...
 * @param a - Denominator coefficients of z^0, z^-1, ... (a[0] != 0).
 * @param tol - Tolerance for repeated poles (default 1e-3).
 * @returns Residues, poles and the direct term (in z^-1, lowest power first).
 * @example
 * ```ts
 * import { residuez } from 'mdml-engine/numerics/polynomial'
 * // 1 / (1 - 0.5 z^-1): one pole at 0.5 with residue 1.
 * residuez([1], [1, -0.5]).poles.re // => [0.5]
 * ```
 */
export function residuez(b: readonly number[], a: readonly number[], tol = 1e-3): PartialFractions {
  const trimEnd = (c: readonly number[]): number[] => [...trimLeading([...c].reverse())].reverse()
  const bt = trimEnd(b)
  const at = trimEnd(a)
  if (at.length === 0 || at[0] === 0) throw new MdmlError('value', 'residuez: a[0] must be non-zero')
  // In w = z^-1: b(w) = sum b_i w^i, so the arrays are already lowest first in w.
  let numerator = bt
  let direct: number[] = []
  if (bt.length >= at.length) {
    const { quotient, remainder } = polydiv(bt, at)
    direct = quotient
    numerator = remainder
  }
  // Poles in z are the roots of a_0 z^N + a_1 z^(N-1) + ... + a_N.
  const roots = polyroots([...at].reverse())
  const poles: Complex[] = roots.re.map((r, i) => [r, roots.im[i] ?? 0])
  const { unique, multiplicity } = groupPoles(poles, tol)
  // In w the poles are 1 / p; a(w) = a_N prod (w - 1/p).
  const lead = at[at.length - 1] as number
  const inverse = unique.map((p) => cdiv([1, 0], p))
  const raw = residuesOf(
    [...numerator].reverse().map((v): Complex => [v / lead, 0]),
    inverse,
    multiplicity,
  )
  // r / (w - 1/p)^j = r (-p)^j / (1 - p w)^j.
  let index = 0
  const rs: Complex[] = []
  unique.forEach((p, k) => {
    let power: Complex = [1, 0]
    for (let j = 1; j <= (multiplicity[k] ?? 1); j++) {
      power = cmul(power, [-p[0], -p[1]])
      rs.push(cmul(raw[index] as Complex, power))
      index++
    }
  })
  const repeated = unique.flatMap((p, k) => new Array<Complex>(multiplicity[k] ?? 1).fill(p))
  return { residues: split(rs), poles: split(repeated), direct }
}

/**
 * Rebuild b(s) / a(s) from a partial-fraction expansion (scipy.signal.invres): consecutive poles closer than tol form
 * one repeated pole, whose residues are for powers 1, 2, ...
 *
 * @param r - Residues.
 * @param p - Poles (repeated poles adjacent).
 * @param k - Direct term (highest first).
 * @param tol - Tolerance for grouping (default 1e-3).
 * @returns Numerator and denominator, highest first, complex in general.
 * @example
 * ```ts
 * import { invres } from 'mdml-engine/numerics/polynomial'
 * invres({ re: [1, -1], im: [0, 0] }, { re: [-1, -2], im: [0, 0] }, []).denominator.re // => [1, 3, 2]
 * ```
 */
export function invres(
  r: ComplexArray,
  p: ComplexArray,
  k: readonly number[],
  tol = 1e-3,
): { readonly numerator: ComplexArray; readonly denominator: ComplexArray } {
  const poles: Complex[] = p.re.map((v, i) => [v, p.im[i] ?? 0])
  const residues: Complex[] = r.re.map((v, i) => [v, r.im[i] ?? 0])
  // Group consecutive poles, as scipy's _group_poles.
  const unique: Complex[] = []
  const multiplicity: number[] = []
  let block: Complex[] = []
  poles.forEach((q, i) => {
    if (i > 0 && cabs(csub(q, block[0] as Complex)) <= tol) block.push(block[0] as Complex)
    else {
      if (block.length > 0) {
        unique.push([
          block.reduce((s, b) => s + b[0], 0) / block.length,
          block.reduce((s, b) => s + b[1], 0) / block.length,
        ])
        multiplicity.push(block.length)
      }
      block = [q]
    }
  })
  if (block.length > 0) {
    unique.push([
      block.reduce((s, b) => s + b[0], 0) / block.length,
      block.reduce((s, b) => s + b[1], 0) / block.length,
    ])
    multiplicity.push(block.length)
  }
  let denominator: Complex[] = [[1, 0]]
  unique.forEach((u, j) => {
    for (let m = 0; m < (multiplicity[j] ?? 1); m++) denominator = timesLinear(denominator, u)
  })
  const addInto = (acc: Complex[], poly: readonly Complex[], scale: Complex): Complex[] => {
    const n = Math.max(acc.length, poly.length)
    const out: Complex[] = Array.from({ length: n }, () => [0, 0] as Complex)
    acc.forEach((c, i) => {
      out[n - acc.length + i] = cadd(out[n - acc.length + i] as Complex, c)
    })
    poly.forEach((c, i) => {
      out[n - poly.length + i] = cadd(out[n - poly.length + i] as Complex, cmul(c, scale))
    })
    return out
  }
  let numerator: Complex[] = [[0, 0]]
  if (k.length > 0) {
    const kc: Complex[] = k.map((v) => [v, 0])
    // k * denominator.
    const prod: Complex[] = Array.from({ length: kc.length + denominator.length - 1 }, () => [0, 0] as Complex)
    kc.forEach((x, i) => {
      denominator.forEach((y, j) => {
        prod[i + j] = cadd(prod[i + j] as Complex, cmul(x, y))
      })
    })
    numerator = prod
  }
  let index = 0
  unique.forEach((_u, j) => {
    const m = multiplicity[j] ?? 1
    for (let power = 1; power <= m; power++) {
      // denominator / (s - u)^power.
      let factor: Complex[] = [[1, 0]]
      unique.forEach((v, i) => {
        const times = i === j ? m - power : (multiplicity[i] ?? 1)
        for (let t = 0; t < times; t++) factor = timesLinear(factor, v)
      })
      numerator = addInto(numerator, factor, residues[index] ?? [0, 0])
      index++
    }
  })
  return { numerator: split(numerator), denominator: split(denominator) }
}
