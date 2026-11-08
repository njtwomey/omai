/**
 * System norms, which measure a system's size as an operator: the H2 norm (output energy under white-noise input, from
 * a Gramian) and the H-infinity norm (the peak gain over all frequencies, by the Bruinsma-Steinbuch algorithm), as
 * python-control's `system_norm`.
 */

import { OmaiError } from '../base/index.ts'
import { eigvals, solve, svdvals } from '../numerics/linalg/index.ts'
import {
  add,
  concat,
  eye,
  matmul,
  max,
  mul,
  primalOf,
  reshape,
  sqrt,
  sub,
  tensor,
  trace,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { numbers } from './_shared.ts'
import { gram } from './analysis.ts'
import { freqresp } from './frequency.ts'
import { isDiscrete, type Lti, type StateSpace, toSs } from './representations.ts'
import { isStable, poles } from './stability.ts'

/**
 * The H2 norm (python-control norm with p = 2): sqrt(trace(C Wc C^T)) in continuous time (infinite unless D = 0),
 * sqrt(trace(C Wc C^T + D D^T)) in discrete time, infinite for an unstable system. Differentiable in the matrices.
 *
 * @param sys - The system.
 * @returns The norm (a scalar value).
 * @concept system-norms
 * @example
 * ```ts
 * import { h2Norm, tf } from 'omai-engine/systems'
 * import { item } from 'omai-engine/tensor'
 * item(h2Norm(tf([1], [1, 1]))) // => 0.7071067811865476
 * ```
 */
export function h2Norm(sys: Lti): Value {
  const s = toSs(sys)
  if (!isStable(s)) return tensor(Number.POSITIVE_INFINITY)
  if (!isDiscrete(s) && numbers(s.d).some((v) => v !== 0)) return tensor(Number.POSITIVE_INFINITY)
  const wc = gram(s, 'c')
  const ccT = trace(matmul(s.c, matmul(wc, transpose(s.c))))
  return sqrt(isDiscrete(s) ? add(ccT, trace(matmul(s.d, transpose(s.d)))) : ccT)
}

/** The H-infinity norm and where it peaks. */
export interface HinfNorm {
  /** The norm: the peak gain over frequency (a scalar value, differentiable in the system). */
  readonly norm: Value
  /** The frequency of the peak (radians per time unit; 0 for a peak at dc). */
  readonly frequency: number
}

/** sigma_max of the response at one frequency, as a value (the real embedding [[X, -Y], [Y, X]] for many channels). */
function peakGain(sys: StateSpace, w: number): Value {
  const h = freqresp(sys, [w]).h
  if (h.re.shape.length === 1) return reshape(sqrt(add(mul(h.re, h.re), mul(h.im, h.im))), [])
  const x = reshape(h.re, h.re.shape.slice(1) as number[])
  const y = reshape(h.im, h.im.shape.slice(1) as number[])
  return max(svdvals(concat([concat([x, mul(y, -1)], 1), concat([y, x], 1)], 0)))
}

const gainAt = (sys: StateSpace, w: number): number => numbers(peakGain(sys, w)).at(0) ?? 0

/** Imaginary-axis eigenvalues (as frequencies) of the Hamiltonian of a continuous system at level gamma. */
function hamiltonianCrossings(s: StateSpace, gamma: number): number[] {
  const m = s.b.shape[1] ?? 0
  const p = s.c.shape[0] ?? 0
  const dT = transpose(s.d)
  const r = sub(mul(eye(m), gamma * gamma), matmul(dT, s.d))
  const sm = sub(mul(eye(p), gamma * gamma), matmul(s.d, dT))
  const rInvDtC = solve(r, matmul(dT, s.c))
  const ae = add(s.a, matmul(s.b, rInvDtC))
  const top = concat([ae, mul(matmul(s.b, solve(r, transpose(s.b))), gamma)], 1)
  const bottom = concat([mul(matmul(transpose(s.c), solve(sm, s.c)), -gamma), mul(transpose(ae), -1)], 1)
  const { re, im } = eigvals(primalOf(concat([top, bottom], 0)))
  const rv = numbers(re)
  const iv = numbers(im)
  const scale = Math.max(
    1,
    rv.reduce((acc, v, i) => Math.max(acc, Math.hypot(v, iv[i] ?? 0)), 0),
  )
  return iv.filter((w, i) => w >= 0 && Math.abs(rv[i] ?? 0) <= 1e-8 * scale).sort((a, b) => a - b)
}

/** A discrete system as the continuous system with the same H-infinity norm, through z = (1 + s) / (1 - s). */
function bilinearToContinuous(s: StateSpace): StateSpace {
  const n = s.a.shape[0] ?? 0
  const identity = eye(n)
  // A_c = (A + I)^{-1}(A - I), B_c = sqrt(2) (A + I)^{-1} B, C_c = sqrt(2) C (A + I)^{-1}, D_c = D - C (A + I)^{-1} B.
  const ap = add(s.a, identity)
  const a = solve(ap, sub(s.a, identity))
  const b = mul(solve(ap, s.b), Math.SQRT2)
  const cInv = transpose(solve(transpose(ap), transpose(s.c)))
  return { kind: 'ss', a, b, c: mul(cInv, Math.SQRT2), d: sub(s.d, matmul(cInv, s.b)), dt: 0 }
}

/**
 * The H-infinity norm of a stable system (python-control norm with p = 'inf'): the peak over frequency of the largest
 * singular value of the response, by the Bruinsma-Steinbuch two-step algorithm (Hamiltonian imaginary-axis eigenvalues
 * bracket the peaks; discrete systems through the bilinear map). The norm is evaluated at the peak as a value, so it is
 * differentiable in the system (the peak frequency is stationary). Infinite for an unstable system.
 *
 * @param sys - The system.
 * @param options - Relative tolerance (default 1e-10).
 * @returns The norm and the peak frequency.
 * @concept system-norms
 * @example
 * ```ts
 * import { hinfNorm, tf } from 'omai-engine/systems'
 * import { item } from 'omai-engine/tensor'
 * Math.round(item(hinfNorm(tf([1], [1, 0.2, 1])).norm) * 1e8) / 1e8 // => 5.02518908
 * ```
 */
export function hinfNorm(sys: Lti, options: { readonly tol?: number } = {}): HinfNorm {
  const s = toSs(sys)
  if (!isStable(s)) return { norm: tensor(Number.POSITIVE_INFINITY), frequency: Number.NaN }
  const n = s.a.shape[0] ?? 0
  const tol = options.tol ?? 1e-10
  const discrete = isDiscrete(s)
  const c = discrete ? bilinearToContinuous(s) : s
  // Map a continuous frequency of the transformed system back to the original's.
  const back = (w: number): number => (discrete ? (2 * Math.atan(w)) / s.dt : w)
  if (n === 0) return { norm: max(svdvals(s.d)), frequency: 0 }
  // Initial lower bound: dc, infinity and the pole frequencies.
  const ps = poles(c)
  const candidates = [0, ...ps.re.map((re, i) => Math.hypot(re, ps.im[i] ?? 0)), ...ps.im.filter((v) => v > 0)]
  let best = 0
  let gammaLow = numbers(max(svdvals(c.d))).at(0) ?? 0
  for (const w of candidates) {
    const g = gainAt(c, w)
    if (g > gammaLow) {
      gammaLow = g
      best = w
    }
  }
  for (let it = 0; it < 100; it++) {
    const gamma = (1 + 2 * tol) * gammaLow
    const crossings = hamiltonianCrossings(c, gamma)
    if (crossings.length === 0) break
    // The gain exceeds gamma between consecutive crossings (from dc when their number is odd).
    const ws = crossings.length % 2 === 1 ? [0, ...crossings] : crossings
    let improved = false
    for (let k = 0; k + 1 < ws.length; k += 2) {
      const mid = ((ws[k] ?? 0) + (ws[k + 1] ?? 0)) / 2
      const g = gainAt(c, mid)
      if (g > gammaLow) {
        gammaLow = g
        best = mid
        improved = true
      }
    }
    if (!improved) break
  }
  return { norm: peakGain(s, back(best)), frequency: back(best) }
}

/**
 * A system norm (python-control system_norm): 2 for the H2 norm, 'inf' for the H-infinity norm.
 *
 * @param sys - The system.
 * @param p - 2 or 'inf'.
 * @returns The norm, a scalar value.
 * @throws OmaiError with code `value` for another p.
 * @concept system-norms
 * @example
 * ```ts
 * import { systemNorm, tf } from 'omai-engine/systems'
 * import { item } from 'omai-engine/tensor'
 * item(systemNorm(tf([2], [1, 1]), 'inf')) // => 2
 * ```
 */
export function systemNorm(sys: Lti, p: 2 | 'inf'): Value {
  if (p === 2) return h2Norm(sys)
  if (p === 'inf') return hinfNorm(sys).norm
  throw new OmaiError('value', `systemNorm: p must be 2 or 'inf', got ${String(p)}`)
}
