import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { mul, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import {
  allenCahn,
  chebyshevBvp,
  chebyshevMatrix,
  dealiasMask,
  fourierDerivative,
  kuramotoSivashinsky,
} from './spectral.ts'

describe('spectral methods', () => {
  it('Fourier derivatives are spectrally accurate and differentiable', () => {
    const n = 32
    const x = Array.from({ length: n }, (_, j) => (2 * Math.PI * j) / n)
    const u = x.map((v) => Math.exp(Math.sin(v)))
    const d = toFlat(fourierDerivative(u, 2 * Math.PI))
    x.forEach((v, j) => {
      expect(Math.abs((d[j] ?? 0) - Math.cos(v) * Math.exp(Math.sin(v)))).toBeLessThan(1e-11)
    })
    // d/du of sum(w * D u) = D^T w.
    const g = grad((v: Value) => sum(mul(fourierDerivative(v, 2 * Math.PI), tensor(x.map(Math.sin)))))(tensor(u))
    // D is antisymmetric: D^T sin = -cos.
    toFlat(g).forEach((v, j) => {
      expect(Math.abs(v + Math.cos(x[j] ?? 0))).toBeLessThan(1e-10)
    })
    expect(dealiasMask(9).reduce((a, b) => a + b, 0)).toBe(7)
  })

  it('Chebyshev differentiation is exact for polynomials and the collocation BVP converges spectrally', () => {
    const D = chebyshevMatrix(6)
    const t = Array.from({ length: 7 }, (_, j) => Math.cos((Math.PI * j) / 6))
    const p = toFlat(
      matvec(
        D.data,
        t.map((v) => v ** 5),
        7,
      ),
    )
    t.forEach((v, j) => {
      expect(Math.abs((p[j] ?? 0) - 5 * v ** 4)).toBeLessThan(1e-11)
    })
    const exact = (x: number): number => (Math.exp(4 * x) - x * Math.sinh(4) - Math.cosh(4)) / 16
    const r = chebyshevBvp((x) => Math.exp(4 * x), { n: 24 })
    r.x.forEach((v, j) => {
      expect(Math.abs((r.u.data[j] ?? 0) - exact(v))).toBeLessThan(1e-12)
    })
    // u'' + u = 0 on [0, pi/2], u(0) = 0, u(pi/2) = 1.
    const s = chebyshevBvp(() => 0, { n: 16, interval: [0, Math.PI / 2], a0: 1, left: 0, right: 1 })
    s.x.forEach((v, j) => {
      expect(Math.abs((s.u.data[j] ?? 0) - Math.sin(v))).toBeLessThan(1e-12)
    })
  })

  it('Kuramoto-Sivashinsky stays bounded and Allen-Cahn separates into phases', () => {
    const ks = kuramotoSivashinsky((x) => Math.cos(x / 16) * (1 + Math.sin(x / 16)), {
      n: 64,
      length: 32 * Math.PI,
      dt: 0.25,
      tEnd: 20,
      every: 20,
    })
    expect(ks.status).toBe('success')
    let biggest = 0
    for (const v of ks.u.data) biggest = Math.max(biggest, Math.abs(v))
    expect(biggest).toBeLessThan(10)
    const ac = allenCahn((x) => 0.1 * Math.sin(x) + 0.05 * Math.sin(3 * x), {
      n: 64,
      length: 2 * Math.PI,
      dt: 0.1,
      tEnd: 20,
      epsilon: 0.02,
    })
    const u = Array.from(ac.u.data).slice(-64)
    const near = u.filter((v) => Math.abs(Math.abs(v) - 1) < 0.05).length
    expect(near).toBeGreaterThan(40)
  })
})

function matvec(m: ArrayLike<number>, v: readonly number[], n: number): Value {
  return tensor(Array.from({ length: n }, (_, i) => v.reduce((s, x, j) => s + (m[i * n + j] ?? 0) * x, 0)))
}
