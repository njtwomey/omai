import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cos, mul, neg, sliceAxis, stack, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { convergenceOrder } from './convergence.ts'
import { solveIvp } from './solve-ivp.ts'
import {
  energyDrift,
  FOREST_RUTH,
  STORMER_VERLET,
  SYMPLECTIC_EULER,
  separableHamiltonian,
  symplecticIntegrate,
  VELOCITY_VERLET,
  YOSHIDA4,
} from './symplectic.ts'

const pendulum = separableHamiltonian((q) => neg(sum(cos(q))))
const sho = (k: number) => separableHamiltonian((q) => mul(sum(mul(q, q)), k / 2))

describe('symplectic integrators', () => {
  for (const m of [SYMPLECTIC_EULER, STORMER_VERLET, VELOCITY_VERLET, FOREST_RUTH, YOSHIDA4]) {
    it(`${m.name} converges at order ${m.order}`, () => {
      const exact = Math.cos(2)
      const study = convergenceOrder(
        (dt) => {
          const tr = symplecticIntegrate(sho(1), [1], [0], [0, 2], { dt, method: m })
          return Math.abs((toFlat(tr.q).at(-1) ?? 0) - exact)
        },
        [0.1, 0.05, 0.025],
      )
      expect(Math.abs(study.order - m.order)).toBeLessThan(0.3)
    })
  }

  it('keeps the energy error bounded over long times (property)', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.2, max: 2.5, noNaN: true }),
        fc.double({ min: -1, max: 1, noNaN: true }),
        (q0, p0) => {
          const tr = symplecticIntegrate(pendulum, [q0], [p0], [0, 100], { dt: 0.1, method: STORMER_VERLET, every: 10 })
          const d = energyDrift(tr.t, tr.energy)
          const h0 = tr.energy[0] ?? 0
          return tr.energy.every((e) => Math.abs(e - h0) < 0.02) && Math.abs(d.slope) < 1e-4
        },
      ),
      { numRuns: 8 },
    )
  })

  it('a non-symplectic method drifts where leapfrog does not', () => {
    const f = (_t: number, y: Value): Value => stack([sum(sliceAxis(y, 0, 1, 2)), neg(sum(sliceAxis(y, 0, 0, 1)))])
    const sol = solveIvp(f, [0, 200], [1, 0], { method: 'Euler', dt: 0.01 })
    const energies = sol.ys.map((y) => {
      const [q, p] = toFlat(y)
      return 0.5 * ((q ?? 0) ** 2 + (p ?? 0) ** 2)
    })
    const euler = energyDrift(sol.t, energies)
    const leap = symplecticIntegrate(sho(1), [1], [0], [0, 200], { dt: 0.01 })
    const verlet = energyDrift(leap.t, leap.energy)
    expect(euler.slope).toBeGreaterThan(100 * Math.abs(verlet.slope))
  })

  it('works on vector states', () => {
    const tr = symplecticIntegrate(sho(4), tensor([1, -1]), tensor([0, 0]), [0, 1], { dt: 0.01, method: YOSHIDA4 })
    const q = toFlat(tr.q).slice(-2)
    expect(Math.abs((q[0] ?? 0) - Math.cos(2))).toBeLessThan(1e-7)
  })
})
