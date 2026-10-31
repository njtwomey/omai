import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../../trace/index.ts'
import * as a from './index.ts'

const sum = (x: ArrayLike<number>): number => Array.from(x).reduce((s, v) => s + v, 0)

describe('lattice Boltzmann D2Q9', () => {
  it('the equilibrium has the moments it is built from', () => {
    const f = Float64Array.from({ length: 9 }, (_, k) => a.d2q9Equilibrium(k, 1.3, 0.04, -0.02))
    const m = a.d2q9Moments(f, 1)
    expect(m.rho[0]).toBeCloseTo(1.3, 14)
    expect((m.ux[0] ?? 0) * 1).toBeCloseTo(0.04, 14)
    expect(m.uy[0]).toBeCloseTo(-0.02, 14)
  })
  it('a fluid at rest stays at rest', () => {
    const s = runAlgorithm(a.latticeBoltzmann({ width: 8, height: 8 }), { seed: 0, maxSteps: 50 }).state
    const m = a.d2q9Moments(s.f, 64)
    for (let i = 0; i < 64; i++) {
      expect(m.rho[i]).toBeCloseTo(1, 12)
      expect(m.ux[i]).toBeCloseTo(0, 12)
    }
  })
  it('Poiseuille flow between walls matches the analytic parabola', () => {
    const [H, g] = [17, 1e-5]
    for (const [tau, tol] of [
      [1, 0.02],
      [0.5 + Math.sqrt(3) / 4, 0.005],
    ] as const) {
      const alg = a.latticeBoltzmann({ width: 4, height: H, walls: true, tau, force: [g, 0] })
      const s = runAlgorithm(alg, { seed: 0, maxSteps: 4000 }).state
      const m = a.d2q9Moments(s.f, 4 * H, [g, 0])
      const exact = a.poiseuilleProfile(H, g, tau)
      for (let j = 0; j < H; j++) {
        expect(Math.abs((m.ux[j * 4 + 1] ?? 0) / (exact[j] ?? 1) - 1), `tau=${tau} j=${j}`).toBeLessThan(tol)
        expect(Math.abs(m.uy[j * 4 + 1] ?? 1)).toBeLessThan(1e-9)
      }
    }
  })
  it('a shear wave decays at the rate nu k^2 set by tau', () => {
    const [H, U, tau] = [32, 0.01, 0.8]
    const alg = a.latticeBoltzmann({
      width: 4,
      height: H,
      tau,
      initial: (_x, y) => ({ rho: 1, ux: U * Math.sin((2 * Math.PI * y) / H), uy: 0 }),
    })
    const t = 200
    const s = runAlgorithm(alg, { seed: 0, maxSteps: t }).state
    const m = a.d2q9Moments(s.f, 4 * H)
    // The amplitude is the projection on the initial mode.
    let amp = 0
    for (let y = 0; y < H; y++) amp += ((m.ux[y * 4] ?? 0) * Math.sin((2 * Math.PI * y) / H) * 2) / H
    const nu = (tau - 0.5) / 3
    const expected = U * Math.exp(-nu * ((2 * Math.PI) / H) ** 2 * t)
    expect(Math.abs(amp / expected - 1)).toBeLessThan(0.01)
  })
  it('conserves mass with a body force and a cylinder in a periodic box', () => {
    const alg = a.latticeBoltzmann({ width: 40, height: 20, force: [1e-5, 0], cylinder: { x: 10, y: 10, radius: 3 } })
    const t0 = runAlgorithm(alg, { seed: 0, maxSteps: 0 }).state
    const t1 = runAlgorithm(alg, { seed: 0, maxSteps: 300 }).state
    expect(sum(t1.f)).toBeCloseTo(sum(t0.f), 8)
  })
  it('a body force along +x pushes the fluid toward +x', () => {
    const s = runAlgorithm(a.latticeBoltzmann({ width: 8, height: 8, force: [1e-4, 0] }), {
      seed: 0,
      maxSteps: 100,
    }).state
    const m = a.d2q9Moments(s.f, 64, [1e-4, 0])
    // The momentum after t steps is F t; the velocity adds the half-step of the force.
    expect(m.ux[0]).toBeCloseTo(1e-4 * 100.5, 12)
    expect(Math.abs(m.uy[0] ?? 1)).toBeLessThan(1e-12)
  })
  it('a density bump in a stream moving up the picture (to smaller rows) is carried that way', () => {
    const [W, H, u, t] = [4, 80, 0.05, 40]
    const alg = a.latticeBoltzmann({
      width: W,
      height: H,
      initial: (x, y) => ({ rho: x === 1 && y === 40 ? 1.2 : 1, ux: 0, uy: u }),
    })
    const s = runAlgorithm(alg, { seed: 0, maxSteps: t }).state
    const m = a.d2q9Moments(s.f, W * H)
    let mass = 0
    let moment = 0
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const d = (m.rho[y * W + x] ?? 1) - 1
        mass += d
        moment += d * y
      }
    // y points up while rows run down: speed u upward is -u in rows.
    expect(moment / mass - 40).toBeCloseTo(-u * t, 0)
  })
  it('flow past a cylinder from an inlet stays stable, is deflected by the obstacle, and mirrors top to bottom', () => {
    const [W, H] = [80, 31]
    const alg = a.latticeBoltzmann({
      width: W,
      height: H,
      tau: 0.6,
      inflow: 0.05,
      walls: false,
      cylinder: { x: 20, y: 15, radius: 4 },
    })
    const s = runAlgorithm(alg, { seed: 0, maxSteps: 400 }).state
    const m = a.d2q9Moments(s.f, W * H)
    for (const v of m.ux) expect(Number.isFinite(v)).toBe(true)
    expect(Math.max(...m.ux)).toBeLessThan(0.2)
    // Far upstream the stream is the inflow; right behind the cylinder it is slower.
    expect(m.ux[15 * W + 2]).toBeCloseTo(0.05, 2)
    expect(m.ux[15 * W + 26] ?? 1).toBeLessThan(0.02)
    // The box and the cylinder are mirror symmetric about row 15 and the dynamics is too, to rounding.
    for (let r = 0; r < 15; r++) expect(m.ux[r * W + 40]).toBeCloseTo(m.ux[(30 - r) * W + 40] ?? 0, 9)
  })
  it('rejects tau at or below one half', () => {
    expect(() => a.latticeBoltzmann({ tau: 0.5 })).toThrow(/tau/)
  })
})
