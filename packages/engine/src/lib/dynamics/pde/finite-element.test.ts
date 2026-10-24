import { describe, expect, it } from 'vitest'
import { convergenceOrder } from '../ode/convergence.ts'
import { assemble1d, assemble2d, femL2Error, heatFem, rectangleMesh, solveFem } from './finite-element.ts'

const exact1 = (x: number): number => Math.sin(Math.PI * x)
const meshOf = (n: number): number[] => Array.from({ length: n + 1 }, (_, i) => i / n)

describe('finite elements', () => {
  it('P1 and P2 converge at orders 2 and 3 in L2 (1-D Poisson)', () => {
    for (const order of [1, 2] as const) {
      const study = convergenceOrder(
        (h) => {
          const sys = assemble1d(meshOf(Math.round(1 / h)), { f: (x) => Math.PI ** 2 * exact1(x) }, order)
          return femL2Error(sys, solveFem(sys), exact1)
        },
        [1 / 8, 1 / 16, 1 / 32],
      )
      expect(Math.abs(study.order - (order + 1))).toBeLessThan(0.2)
    }
  })

  it("variable coefficients and reaction: -(k u')' + c u = f", () => {
    // u = x (1 - x), k = 1 + x, c = 2: f = -(k u')' + 2 u = -(1 - 2x) + 2(1 + x) + 2x(1 - x).
    const f = (x: number): number => -(1 - 2 * x) + 2 * (1 + x) + 2 * x * (1 - x)
    const sys = assemble1d(meshOf(32), { k: (x) => 1 + x, c: 2, f }, 2)
    expect(femL2Error(sys, solveFem(sys), (x) => x * (1 - x))).toBeLessThan(1e-6)
  })

  it('P1 triangles converge at order 2 (2-D Poisson with boundary data)', () => {
    const u = (x: number, y: number): number =>
      (Math.sin(Math.PI * x) * Math.sinh(Math.PI * y)) / Math.sinh(Math.PI) + x * y
    const study = convergenceOrder(
      (h) => {
        const n = Math.round(1 / h) + 1
        const sys = assemble2d(rectangleMesh([0, 1], n, [0, 1], n))
        return femL2Error(
          sys,
          solveFem(sys, (x, y) => u(x ?? 0, y ?? 0)),
          (x, y) => u(x ?? 0, y ?? 0),
        )
      },
      [1 / 8, 1 / 16],
    )
    expect(study.order).toBeGreaterThan(1.8)
  })

  it('the heat problem decays like the first mode', () => {
    const sys = assemble2d(rectangleMesh([0, 1], 17, [0, 1], 17))
    const s = heatFem(sys, (x, y) => Math.sin(Math.PI * (x ?? 0)) * Math.sin(Math.PI * (y ?? 0)), {
      dt: 0.005,
      tEnd: 0.05,
      theta: 1,
    })
    const centre = 8 * 17 + 8
    const u = Array.from(s.u.data).slice(-sys.size)
    expect(Math.abs((u[centre] ?? 0) - Math.exp(-2 * Math.PI ** 2 * 0.05))).toBeLessThan(0.02)
  })

  it('validates meshes', () => {
    expect(() => assemble1d([0, 0])).toThrow()
    expect(() => rectangleMesh([0, 1], 1, [0, 1], 3)).toThrow()
  })
})
