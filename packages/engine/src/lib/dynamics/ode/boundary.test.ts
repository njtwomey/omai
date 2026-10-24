import { describe, expect, it } from 'vitest'
import { concat, exp, mul, neg, sliceAxis, sub, tensor, type Value } from '../../tensor/index.ts'
import { multipleShooting, shooting, solveBvp } from './boundary.ts'

const row = (y: Value, i: number): Value => sliceAxis(y, 0, i, i + 1)
// y'' = -y on [0, pi/2] with y(0) = 0, y(pi/2) = 1: y = sin x.
const field = (_t: number, y: Value): Value => concat([row(y, 1), neg(row(y, 0))], 0)
const bc = (ya: Value, yb: Value): Value => concat([row(ya, 0), sub(row(yb, 0), 1)], 0)

describe('boundary value problems', () => {
  it('collocation, single and multiple shooting agree with the exact solution', () => {
    const x = [0, 0.4, 0.8, 1.2, Math.PI / 2]
    const r = solveBvp((_x, y) => field(0, y), bc, x, tensor([x.map(() => 0.5), x.map(() => 1)]), { tol: 1e-8 })
    expect(r.success).toBe(true)
    expect(Math.abs((r.sol(1).data[0] ?? 0) - Math.sin(1))).toBeLessThan(1e-7)
    expect(Math.abs((r.sol(1, 1).data[0] ?? 0) - Math.cos(1))).toBeLessThan(1e-6)
    const s = shooting(field, bc, [0, Math.PI / 2], [0, 0])
    expect(s.converged).toBe(true)
    expect(Math.abs((s.y0.data[1] ?? 0) - 1)).toBeLessThan(1e-8)
    const ms = multipleShooting(
      field,
      bc,
      [0, 0.5, 1, Math.PI / 2],
      [
        [0, 0],
        [0, 0],
        [0, 0],
      ],
    )
    expect(ms.converged).toBe(true)
    expect(Math.abs((ms.nodes.data[2] ?? 0) - Math.sin(0.5))).toBeLessThan(1e-8)
  })

  it('a stiff boundary layer refines the mesh, and the node budget gives a status', () => {
    // eps y'' = y, y(0) = 1, y(1) = 0: a layer of width sqrt(eps) at x = 0.
    const eps = 1e-3
    const f = (_x: Value, y: Value): Value => concat([row(y, 1), mul(row(y, 0), 1 / eps)], 0)
    const b = (ya: Value, yb: Value): Value => concat([sub(row(ya, 0), 1), row(yb, 0)], 0)
    const x = [0, 0.25, 0.5, 0.75, 1]
    const y = tensor([x.map(() => 0), x.map(() => 0)])
    const good = solveBvp(f, b, x, y, { tol: 1e-5 })
    expect(good.success).toBe(true)
    expect(good.x.length).toBeGreaterThan(20)
    const exact = Math.exp(-0.05 / Math.sqrt(eps))
    expect(Math.abs((good.sol(0.05).data[0] ?? 0) - exact)).toBeLessThan(1e-4)
    const capped = solveBvp(f, b, x, y, { tol: 1e-8, maxNodes: 10 })
    expect(capped.status).toBe(1)
    expect(capped.success).toBe(false)
  })

  it('validates inputs', () => {
    expect(() =>
      solveBvp(
        (_x, y) => y,
        bc,
        [0, 0],
        tensor([
          [0, 0],
          [0, 0],
        ]),
      ),
    ).toThrow()
    expect(() => multipleShooting(field, bc, [0], [[0, 0]])).toThrow()
  })

  it('Bratu has two solutions', () => {
    const f = (_x: Value, y: Value): Value => concat([row(y, 1), neg(exp(row(y, 0)))], 0)
    const b = (ya: Value, yb: Value): Value => concat([row(ya, 0), row(yb, 0)], 0)
    const x = [0, 0.25, 0.5, 0.75, 1]
    const low = solveBvp(f, b, x, tensor([x.map(() => 0), x.map(() => 0)]), { tol: 1e-6 })
    const high = solveBvp(f, b, x, tensor([x.map(() => 3), x.map(() => 0)]), { tol: 1e-6 })
    expect((high.sol(0.5).data[0] ?? 0) - (low.sol(0.5).data[0] ?? 0)).toBeGreaterThan(3)
  })
})
