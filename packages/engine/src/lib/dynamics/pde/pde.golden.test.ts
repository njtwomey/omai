import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { matmul, type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import { heat, type PoissonMethod, poisson } from './finite-difference.ts'
import { fokkerPlanck, stationaryDensity } from './fokker-planck.ts'
import { dirichlet, grid1d, grid2d, laplacian1d } from './grids.ts'
import { chebyshevMatrix, fourierDerivative } from './spectral.ts'

const FD: Readonly<Record<string, GoldenFn>> = {
  poisson2d: (_x, p) => {
    const n = p['n'] as number
    const bc = dirichlet((_t, x, y) => (x ?? 0) * (x ?? 0) - (y ?? 0))
    const r = poisson(
      (x, y) => Math.sin(Math.PI * (x ?? 0)) * Math.cos(2 * (y ?? 0)) + (x ?? 0) * (y ?? 0),
      grid2d([0, 1], n, [0, 1], n),
      { left: bc, right: bc, bottom: bc, top: bc },
      { method: p['method'] as PoissonMethod, tol: 1e-12 },
    )
    return r.u
  },
  heat_cn: () => {
    const g = grid1d(0, 1, 21)
    const op = laplacian1d(g, dirichlet(0), dirichlet(1))
    const s = heat(
      op,
      Array.from(g.x, (x) => Math.sin(Math.PI * x) + x),
      { dt: 0.01, tEnd: 0.1, scheme: 'crank-nicolson' },
    )
    return tensor(toFlat(s.u).slice(-21))
  },
}

const SPECTRAL: Readonly<Record<string, GoldenFn>> = {
  fourier: ([u], p) => fourierDerivative(u as Tensor, 2 * Math.PI, p['order'] as number),
  chebyshev: ([f], p) => matmul(chebyshevMatrix(p['n'] as number), f as Tensor),
}

const FP: Readonly<Record<string, GoldenFn>> = {
  stationary: ([x], p) => {
    const theta = p['theta'] as number
    const sigma = p['sigma'] as number
    const xs = toFlat(x as Tensor)
    const s = stationaryDensity(
      { drift: (v) => -theta * v, sigma: () => sigma },
      [xs[0] ?? 0, xs.at(-1) ?? 0],
      xs.length,
    )
    return tensor(Array.from(s.p))
  },
  transient: (_x, p) => {
    const theta = p['theta'] as number
    const sigma = p['sigma'] as number
    const m0 = p['m0'] as number
    const v0 = p['v0'] as number
    const cells = p['cells'] as number
    const ev = fokkerPlanck(
      { drift: (v) => -theta * v, sigma: () => sigma },
      (x) => Math.exp(-((x - m0) ** 2) / (2 * v0)),
      [p['lo'] as number, p['hi'] as number],
      cells,
      { dt: 0.0025, tEnd: p['t'] as number, theta: 0.5 },
    )
    return tensor(toFlat(ev.p).slice(-cells))
  },
}

describe('dynamics/pde against scipy and numpy', () => {
  describe('finite differences', () => runFixture('dynamics/pde/finite_difference', FD, it))
  describe('spectral', () => runFixture('dynamics/pde/spectral', SPECTRAL, it))
  describe('fokker-planck', () => runFixture('dynamics/pde/fokker_planck', FP, it))
})
