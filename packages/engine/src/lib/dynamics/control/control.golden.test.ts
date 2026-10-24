import { describe, it } from 'vitest'
import { type GoldenFn, runFixture, sortedParts } from '../../../testing/systems.ts'
import { ss } from '../../systems/index.ts'
import { eye, reshape, type Tensor } from '../../tensor/index.ts'
import { dlqe, dlqr, lqe, lqg, lqr } from './optimal.ts'
import { acker, type Pole, place } from './placement.ts'

const lqgOf = ([a, b, c, q, r, rn]: Tensor[]) =>
  lqg(ss(a as Tensor, b as Tensor, c as Tensor), q as Tensor, r as Tensor, eye(2), rn as Tensor).controller

const OPTIMAL: Readonly<Record<string, GoldenFn>> = {
  lqr_K: ([a, b, q, r, n]) => lqr(a as Tensor, b as Tensor, q as Tensor, r as Tensor, n).K,
  lqr_S: ([a, b, q, r]) => lqr(a as Tensor, b as Tensor, q as Tensor, r as Tensor).S,
  dlqr_K: ([a, b, q, r, n]) => dlqr(a as Tensor, b as Tensor, q as Tensor, r as Tensor, n).K,
  dlqr_S: ([a, b, q, r]) => dlqr(a as Tensor, b as Tensor, q as Tensor, r as Tensor).S,
  lqe_L: ([a, g, c, qn, rn]) => lqe(a as Tensor, g as Tensor, c as Tensor, qn as Tensor, rn as Tensor).L,
  lqe_P: ([a, g, c, qn, rn]) => lqe(a as Tensor, g as Tensor, c as Tensor, qn as Tensor, rn as Tensor).P,
  dlqe_L: ([a, g, c, qn, rn]) => dlqe(a as Tensor, g as Tensor, c as Tensor, qn as Tensor, rn as Tensor).L,
  dlqe_P: ([a, g, c, qn, rn]) => dlqe(a as Tensor, g as Tensor, c as Tensor, qn as Tensor, rn as Tensor).P,
  lqg_A: (xs) => lqgOf(xs).a,
  lqg_B: (xs) => lqgOf(xs).b,
  lqg_C: (xs) => lqgOf(xs).c,
}

const polesOf = (p: Readonly<Record<string, unknown>>): Pole[] => {
  const t = p['poles'] as { data: number[]; shape: number[] }
  const n = t.shape[1] ?? 0
  return Array.from({ length: n }, (_, i) => {
    const im = t.data[n + i] ?? 0
    return im === 0 ? (t.data[i] ?? 0) : { re: t.data[i] ?? 0, im }
  })
}

const PLACEMENT: Readonly<Record<string, GoldenFn>> = {
  place_gain: ([a, b], p) => place(a as Tensor, b as Tensor, polesOf(p)).gain,
  // python-control returns the single-input gain as a vector.
  acker: ([a, b], p) => reshape(acker(a as Tensor, b as Tensor, polesOf(p)), [-1]),
  place_poles: ([a, b], p) => {
    const r = place(a as Tensor, b as Tensor, polesOf(p), { method: p['method'] as 'YT' | 'KNV0' })
    return sortedParts({ re: r.computedPoles.map((z) => z.re), im: r.computedPoles.map((z) => z.im) })
  },
}

describe('dynamics/control against python-control and scipy.signal', () => {
  describe('optimal', () => runFixture('dynamics/control/optimal', OPTIMAL, it))
  describe('placement', () => runFixture('dynamics/control/placement', PLACEMENT, it))
})
