/**
 * Stability margins of a single-loop feedback system, the classical measures of how far a loop is from instability:
 * gain, phase, stability and delay margins with their crossover frequencies (python-control stability_margins and
 * margin), and the frequencies where the Nyquist curve crosses the real axis.
 */

import { WhiteboxError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { polyder, polymul } from '../numerics/polynomial/index.ts'
import { evalAt, numbers, polyAddHf, rootsOf, toList, trimLeading } from './_shared.ts'
import { dimensions, isDiscrete, type Lti, toTf } from './representations.ts'

// Stability margins by python-control's polynomial method: crossover frequencies are the real roots of polynomials in
// w (continuous time) or the unit-circle roots of polynomials in z (discrete time).

/** Every margin found, with its frequency (python-control stability_margins with returnall). */
export interface AllMargins {
  /** Gain margins 1 / |L| where the phase crosses -180 degrees, at the frequencies `wpc`. */
  readonly gm: readonly number[]
  readonly wpc: readonly number[]
  /** Phase margins (degrees) where |L| crosses 1, at the frequencies `wgc`. */
  readonly pm: readonly number[]
  readonly wgc: readonly number[]
  /** Distances from -1 to the Nyquist curve at its local minima, at the frequencies `wms`. */
  readonly sm: readonly number[]
  readonly wms: readonly number[]
  /** Delay margins (time units) at each gain crossover: the phase margin (in [0, 360) degrees) over its frequency. */
  readonly dm: readonly number[]
}

/** The deciding margins (python-control stability_margins), plus the smallest delay margin. */
export interface StabilityMargins {
  /** The gain margin closest to 1 in log terms (infinite when the phase never crosses -180 degrees). */
  readonly gm: number
  /** The phase margin (degrees) smallest in magnitude (infinite when the gain never crosses 1). */
  readonly pm: number
  /** The stability margin: the smallest distance from -1 to the Nyquist curve. */
  readonly sm: number
  /** Phase crossover frequency (for gm), NaN when none. */
  readonly wpc: number
  /** Gain crossover frequency (for pm), NaN when none. */
  readonly wgc: number
  /** Frequency of the stability margin, NaN when none. */
  readonly wms: number
  /** The smallest delay margin and its frequency (infinite and NaN without gain crossovers). */
  readonly dm: number
  readonly wdm: number
}

interface Loop {
  readonly num: number[]
  readonly den: number[]
  readonly dt: number
  readonly at: (w: number) => ComplexNumber
}

function loopOf(sys: Lti): Loop {
  const dims = dimensions(sys)
  if (dims.inputs !== 1 || dims.outputs !== 1) {
    throw new WhiteboxError('shape', 'margins: one input and one output are needed')
  }
  const g = toTf(sys)
  if (g.num.shape.length !== 1) throw new WhiteboxError('shape', 'margins: one input and one output are needed')
  const num = trimLeading(numbers(g.num))
  const den = trimLeading(numbers(g.den))
  const dt = g.dt
  const at = (w: number): ComplexNumber => {
    const z = dt > 0 ? C.exp(C.of(0, w * dt)) : C.of(0, w)
    const n = evalAt(num, z)
    const d = evalAt(den, z)
    // At a pole the response is infinite (numpy's complex division gives inf, not NaN, in the real part).
    if (d.re === 0 && d.im === 0) return C.of(n.re / 0, n.im / 0)
    return C.div(n, d)
  }
  return { num, den, dt, at }
}

const lowestFirst = (c: readonly number[]): number[] => [...c].reverse()
const derivative = (c: readonly number[]): number[] => {
  const d = lowestFirst(polyder(lowestFirst(c)))
  return d.length > 0 ? d : [0]
}
const realRoots = (c: readonly number[]): number[] =>
  toList(rootsOf(trimLeading(c)))
    .filter((r) => r.im === 0)
    .map((r) => r.re)
const negate = (c: readonly number[]): number[] => c.map((v) => -v)

/** num(jw) split into real and imaginary coefficient polynomials in w (highest first). */
function partsInW(c: readonly number[]): { re: number[]; im: number[] } {
  const n = c.length - 1
  const re: number[] = []
  const im: number[] = []
  c.forEach((v, k) => {
    // (j)^(n-k): 1, j, -1, -j.
    const power = (n - k) % 4
    re.push(power === 0 ? v : power === 2 ? -v : 0)
    im.push(power === 1 ? v : power === 3 ? -v : 0)
  })
  return { re, im }
}

/** |p(jw)|^2 as a real polynomial in w. */
function squared(p: { re: number[]; im: number[] }): number[] {
  return polyAddHf(polymul(p.re, p.re), polymul(p.im, p.im))
}

function continuousCrossings(loop: Loop): { w180: number[]; wc: number[]; wstab: number[] } {
  const n = partsInW(loop.num)
  const d = partsInW(loop.den)
  // Im H(jw) = 0: Im(num) Re(den) - Re(num) Im(den) = 0.
  const imag = polyAddHf(polymul(n.im, d.re), negate(polymul(n.re, d.im)))
  const w180 = realRoots(imag).filter((w) => w >= 0)
  const wc = realRoots(polyAddHf(squared(n), negate(squared(d)))).filter((w) => w > 0)
  // Extrema of |1 + H|^2 = |num + den|^2 / |den|^2 with a positive second derivative.
  const sumParts = { re: polyAddHf(n.re, d.re), im: polyAddHf(n.im, d.im) }
  const top = squared(sumParts)
  const bottom = squared(d)
  const test = polyAddHf(polymul(derivative(top), bottom), negate(polymul(derivative(bottom), top)))
  const slope = derivative(test)
  const wstab = realRoots(test).filter((w) => w > 0 && evalAt(slope, C.of(w)).re > 0)
  return { w180, wc, wstab }
}

function onUnitCircle(roots: readonly ComplexNumber[], dt: number, eps: number): number[] {
  return roots
    .filter((z) => Math.abs(Math.hypot(z.re, z.im) - 1) < eps)
    .map((z) => Math.atan2(z.im, z.re))
    .filter((arg) => arg >= 0 && arg < Math.PI)
    .map((arg) => arg / dt)
}

function discreteCrossings(loop: Loop): { w180: number[]; wc: number[]; wstab: number[] } {
  const { num, den, dt } = loop
  const pq = num.length - den.length
  if (pq > 0) throw new WhiteboxError('value', 'margins: the transfer function must be proper')
  const numInv = [...num].reverse()
  const denInv = [...den].reverse()
  const shift = [1, ...new Array<number>(-pq).fill(0)]
  // H(z) = H(1/z) on the unit circle: num(z) den(1/z) z^q = num(1/z) z^p den(z) (times z^-pq).
  const p1 = polymul(num, denInv)
  let p2 = polymul(numInv, den)
  if (pq < 0) p2 = polymul(p2, shift)
  const eps = Number.EPSILON ** (1 / p2.length)
  const w180 = onUnitCircle(toList(rootsOf(trimLeading(polyAddHf(p1, negate(p2))))), dt, eps)
  let m1 = polymul(num, numInv)
  const m2 = polymul(den, denInv)
  if (pq < 0) m1 = polymul(m1, shift)
  const eps2 = Number.EPSILON ** (1 / m2.length)
  const wc = onUnitCircle(toList(rootsOf(trimLeading(polyAddHf(m1, negate(m2))))), dt, eps2).filter((w) => w > 0)
  // The closest approach to -1, by a grid then golden-section refinement (python-control minimises from the grid).
  const dist = (wdt: number): number => {
    const h = loop.at(wdt / dt)
    return Math.hypot(h.re + 1, h.im)
  }
  const grid = Array.from({ length: 100 }, (_, i) => 1e-4 * ((2 * Math.PI) / 1e-4) ** (i / 99))
  let best = 0
  grid.forEach((x, i) => {
    if (dist(x) < dist(grid[best] ?? x)) best = i
  })
  let lo = grid[Math.max(best - 1, 0)] ?? 0
  let hi = grid[Math.min(best + 1, grid.length - 1)] ?? 2 * Math.PI
  const golden = (Math.sqrt(5) - 1) / 2
  for (let it = 0; it < 200 && hi - lo > 1e-14 * Math.max(1, hi); it++) {
    const x1 = hi - golden * (hi - lo)
    const x2 = lo + golden * (hi - lo)
    if (dist(x1) < dist(x2)) hi = x2
    else lo = x1
  }
  return { w180, wc, wstab: [(lo + hi) / 2 / dt] }
}

/**
 * Every gain, phase, stability and delay margin of a single-input single-output loop (python-control
 * stability_margins with returnall, polynomial method), sorted by frequency.
 *
 * @param sys - The loop transfer function.
 * @returns All margins and their frequencies.
 * @concept stability-margins
 * @example
 * ```ts
 * import { allMargins, tf } from 'whitebox-engine/systems'
 * allMargins(tf([6], [1, 6, 11, 6])).wpc.map((w) => Math.round(w * 1e9) / 1e9) // => [3.31662479]
 * ```
 */
export function allMargins(sys: Lti): AllMargins {
  const loop = loopOf(sys)
  const found = isDiscrete(sys) ? discreteCrossings(loop) : continuousCrossings(loop)
  const byW = (xs: readonly number[]): number[] => [...xs].sort((a, b) => a - b)
  const wpc = byW(found.w180).filter((w) => {
    const h = loop.at(w)
    // numpy orders complex numbers lexicographically: keep responses on the negative real half.
    return h.re < 0 || (h.re === 0 && h.im <= 0)
  })
  const wgc = byW(found.wc)
  const wms = byW(found.wstab)
  const gm = wpc.map((w) => {
    const h = loop.at(w)
    return 1 / Math.hypot(h.re, h.im)
  })
  const pm = wgc.map((w) => {
    const h = loop.at(w)
    const deg = (Math.atan2(h.im, h.re) * 180) / Math.PI
    return (((deg % 360) + 360) % 360) - 180
  })
  const dm = wgc.map((w, i) => (((((pm[i] ?? 0) % 360) + 360) % 360) * (Math.PI / 180)) / w)
  const sm = wms.map((w) => {
    const h = loop.at(w)
    return Math.hypot(h.re + 1, h.im)
  })
  return { gm, wpc, pm, wgc, sm, wms, dm }
}

const argBest = (xs: readonly number[], score: (x: number) => number): number => {
  let best = -1
  xs.forEach((x, i) => {
    if (best < 0 || score(x) < score(xs[best] ?? x)) best = i
  })
  return best
}

/**
 * The stability margins of a single-input single-output loop (python-control stability_margins): the gain margin
 * closest to 1 (in log terms), the phase margin smallest in magnitude, the stability margin, their frequencies, and the
 * smallest delay margin.
 *
 * @param sys - The loop transfer function.
 * @returns The margins (infinite, with NaN frequencies, when there is no crossing).
 * @concept stability-margins
 * @example
 * ```ts
 * import { stabilityMargins, tf } from 'whitebox-engine/systems'
 * Math.round(stabilityMargins(tf([6], [1, 6, 11, 6])).gm * 1e9) / 1e9 // => 10
 * ```
 */
export function stabilityMargins(sys: Lti): StabilityMargins {
  const all = allMargins(sys)
  const finiteGm = all.gm.filter((g) => Number.isFinite(g))
  const gi = finiteGm.length === 0 ? -1 : argBest(all.gm, (g) => Math.abs(Math.log(g)))
  const pi = argBest(all.pm, Math.abs)
  const si = argBest(all.sm, (s) => s)
  const di = argBest(all.dm, (d) => d)
  return {
    gm: gi < 0 ? Number.POSITIVE_INFINITY : (all.gm[gi] ?? Number.POSITIVE_INFINITY),
    wpc: gi < 0 ? Number.NaN : (all.wpc[gi] ?? Number.NaN),
    pm: pi < 0 ? Number.POSITIVE_INFINITY : (all.pm[pi] ?? Number.POSITIVE_INFINITY),
    wgc: pi < 0 ? Number.NaN : (all.wgc[pi] ?? Number.NaN),
    sm: si < 0 ? Number.POSITIVE_INFINITY : (all.sm[si] ?? Number.POSITIVE_INFINITY),
    wms: si < 0 ? Number.NaN : (all.wms[si] ?? Number.NaN),
    dm: di < 0 ? Number.POSITIVE_INFINITY : (all.dm[di] ?? Number.POSITIVE_INFINITY),
    wdm: di < 0 ? Number.NaN : (all.wgc[di] ?? Number.NaN),
  }
}

/**
 * Gain and phase margins with their crossover frequencies (python-control margin).
 *
 * @param sys - The loop transfer function.
 * @returns gm, pm (degrees), wcg (the phase crossover, for gm) and wcp (the gain crossover, for pm).
 * @concept stability-margins
 * @example
 * ```ts
 * import { margin, tf } from 'whitebox-engine/systems'
 * Math.round(margin(tf([6], [1, 6, 11, 6])).wcg * 1e6) / 1e6 // => 3.316625
 * ```
 */
export function margin(sys: Lti): {
  readonly gm: number
  readonly pm: number
  readonly wcg: number
  readonly wcp: number
} {
  const m = stabilityMargins(sys)
  return { gm: m.gm, pm: m.pm, wcg: m.wpc, wcp: m.wgc }
}

/**
 * Where the Nyquist curve crosses the real axis, and the gains there (python-control phase_crossover_frequencies).
 *
 * @param sys - A single-input single-output system.
 * @returns The non-negative frequencies and the real responses there.
 * @concept stability-margins
 * @example
 * ```ts
 * import { phaseCrossoverFrequencies, tf } from 'whitebox-engine/systems'
 * phaseCrossoverFrequencies(tf([1], [1, 1])).omega // => [0]
 * ```
 */
export function phaseCrossoverFrequencies(sys: Lti): { readonly omega: number[]; readonly gains: number[] } {
  const loop = loopOf(sys)
  const omega = isDiscrete(sys) ? discreteCrossings(loop).w180 : continuousCrossings(loop).w180
  return { omega, gains: omega.map((w) => loop.at(w).re) }
}
