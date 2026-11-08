/**
 * Operator-learning datasets of input and output function pairs on a grid, with exact solutions: the antiderivative of
 * Gaussian-process inputs (the DeepONet benchmark), and the periodic heat and viscous Burgers equations from random
 * Fourier-series initial conditions (the Fourier neural operator benchmark).
 */

import { type DatasetGenerator, type RegressionData, xy } from '../data/index.ts'
import { irfft, rfft } from '../numerics/fourier/index.ts'
import { choleskyJitter } from '../numerics/linalg/index.ts'
import { cumulativeTrapezoid } from '../numerics/quadrature/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { matmul, type Tensor, tensor, toFlat, transpose } from '../tensor/index.ts'
import { matrix, stamp } from './_util.ts'

/**
 * Operator-learning pairs for the antiderivative (the DeepONet benchmark of Lu et al., 2021): input functions `u`
 * drawn from a Gaussian process with a squared-exponential kernel of length scale `lengthScale` on a grid of `points`
 * in [0, 1], and outputs `s(x) = ∫_0^x u` by the trapezoid rule (exact for the piecewise-linear interpolant). Features
 * are `u` on the grid; targets are `s` on the same grid (`truth.extra.grid`).
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { antiderivativeOperator } from 'omai-engine/datasets'
 * generate(antiderivativeOperator, { knobs: { n: 3, points: 20 } }).data.y.shape // => [3, 20]
 * ```
 */
export const antiderivativeOperator: DatasetGenerator<RegressionData> = {
  name: 'antiderivativeOperator',
  kind: 'regression',
  description: 'Gaussian-process input functions and their antiderivatives on a grid.',
  knobs: dict({
    n: int(1, 100_000, { label: 'functions', description: 'The number of input functions.', default: 100 }),
    points: int(2, 1000, {
      label: 'grid points',
      description: 'The number of grid points each function is sampled at.',
      default: 50,
    }),
    lengthScale: real(0.01, 10, {
      label: 'length scale',
      description: 'The length scale of the Gaussian process the inputs are drawn from.',
      default: 0.2,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['points'] as number
    const ell = knobs['lengthScale'] as number
    const grid = Array.from({ length: m }, (_, j) => j / (m - 1))
    const kernel = tensor(grid.map((a) => grid.map((b) => Math.exp(-((a - b) ** 2) / (2 * ell * ell)))))
    const { l } = choleskyJitter(kernel, 1e-6)
    const u = matmul(normal(foldIn(k, 0), [n, m]), transpose(l)) as Tensor
    const operator = (values: Tensor): Tensor => cumulativeTrapezoid(values, { dx: 1 / (m - 1), initial: 0 }) as Tensor
    return {
      data: xy(u, operator(u), stamp('antiderivativeOperator', antiderivativeOperator.description, knobs)),
      truth: {
        regressionFunction: (v) => operator(v as Tensor),
        noiseScale: 0,
        extra: { grid, lengthScale: ell, operator },
      },
    }
  },
}

/** Random zero-mean Fourier series: coefficients `a_k, b_k ~ N(0, (amplitude / k)^2)` for k = 1..modes. */
function fourierCoefficients(k: Key, n: number, modes: number, amplitude: number): { a: number[][]; b: number[][] } {
  const z = normal(k, [n, 2, modes]).data
  const a = Array.from({ length: n }, (_, i) =>
    Array.from({ length: modes }, (_, j) => (amplitude * (z[i * 2 * modes + j] ?? 0)) / (j + 1)),
  )
  const b = Array.from({ length: n }, (_, i) =>
    Array.from({ length: modes }, (_, j) => (amplitude * (z[i * 2 * modes + modes + j] ?? 0)) / (j + 1)),
  )
  return { a, b }
}

function evaluateSeries(
  a: readonly number[],
  b: readonly number[],
  grid: readonly number[],
  decay: (k: number) => number = () => 1,
): number[] {
  return grid.map((x) =>
    a.reduce(
      (s, ak, j) =>
        s +
        decay(j + 1) * (ak * Math.cos(2 * Math.PI * (j + 1) * x) + (b[j] ?? 0) * Math.sin(2 * Math.PI * (j + 1) * x)),
      0,
    ),
  )
}

const periodicKnobs = {
  n: int(1, 100_000, { default: 100, label: 'functions', description: 'The number of initial conditions.' }),
  points: int(4, 2048, {
    default: 64,
    label: 'grid points',
    description: 'The number of grid points on [0, 1) each function is sampled at.',
  }),
  modes: int(1, 64, {
    default: 5,
    label: 'Fourier modes of the initial condition',
    description: 'The number of random Fourier modes in each initial condition (more is rougher).',
  }),
  amplitude: real(0, 100, {
    default: 1,
    label: 'amplitude',
    description: 'The amplitude of the random initial conditions.',
  }),
  time: real(0, 100, { default: 0.05, label: 'final time T', description: 'The time the solution is taken at.' }),
  viscosity: real(0.0001, 10, {
    default: 0.1,
    label: 'diffusivity or viscosity',
    description: 'The diffusivity ν of the heat equation or the viscosity of Burgers’ equation.',
  }),
}

/**
 * Operator-learning pairs for the periodic heat equation `u_t = ν u_xx` on [0, 1): initial conditions are random
 * zero-mean Fourier series and the solution at time `T` is exact (each mode decays by `exp(-ν (2πk)² T)`). Features
 * are `u(·, 0)` and targets `u(·, T)` on the grid; the truth solves any initial Fourier series.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { heatOperator } from 'omai-engine/datasets'
 * generate(heatOperator, { knobs: { n: 2, points: 16 } }).data.x.shape // => [2, 16]
 * ```
 */
export const heatOperator: DatasetGenerator<RegressionData> = {
  name: 'heatOperator',
  kind: 'regression',
  description: 'Initial conditions and exact solutions of the periodic heat equation.',
  knobs: dict(periodicKnobs),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['points'] as number
    const nu = knobs['viscosity'] as number
    const t = knobs['time'] as number
    const grid = Array.from({ length: m }, (_, j) => j / m)
    const { a, b } = fourierCoefficients(k, n, knobs['modes'] as number, knobs['amplitude'] as number)
    const decay = (q: number) => Math.exp(-nu * (2 * Math.PI * q) ** 2 * t)
    const u0 = a.flatMap((ai, i) => evaluateSeries(ai, b[i] ?? [], grid))
    const uT = a.flatMap((ai, i) => evaluateSeries(ai, b[i] ?? [], grid, decay))
    return {
      data: xy(matrix(u0, n, m), matrix(uT, n, m), stamp('heatOperator', heatOperator.description, knobs)),
      truth: {
        noiseScale: 0,
        extra: {
          grid,
          coefficients: { a, b },
          solve: (ca: readonly number[], cb: readonly number[], x: readonly number[], time = t) =>
            evaluateSeries(ca, cb, x, (q) => Math.exp(-nu * (2 * Math.PI * q) ** 2 * time)),
        },
      },
    }
  },
}

/**
 * Solve the periodic viscous Burgers equation `u_t + u u_x = ν u_xx` exactly by the Cole-Hopf transform: with
 * `ψ = ∫_0^x u0`, `φ0 = exp(-ψ / 2ν)` solves the heat equation, done spectrally on a fine grid, and
 * `u = -2ν φ_x / φ`.
 */
function burgersSolve(a: readonly number[], b: readonly number[], nu: number, time: number, fine: number): number[] {
  const grid = Array.from({ length: fine }, (_, j) => j / fine)
  const psi = grid.map((x) =>
    a.reduce((s, ak, j) => {
      const w = 2 * Math.PI * (j + 1)
      return s + (ak * Math.sin(w * x)) / w - ((b[j] ?? 0) * (Math.cos(w * x) - 1)) / w
    }, 0),
  )
  let shift = Number.POSITIVE_INFINITY
  for (const p of psi) shift = Math.min(shift, p)
  // Shifting psi rescales phi by a constant, which cancels in phi_x / phi and keeps the exponent bounded.
  const phi0 = tensor(psi.map((p) => Math.exp(-(p - shift) / (2 * nu))))
  const spec = rfft(phi0)
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const decayed = re.map((_, q) => Math.exp(-nu * (2 * Math.PI * q) ** 2 * time))
  const dRe = re.map((v, q) => v * (decayed[q] ?? 0))
  const dIm = im.map((v, q) => v * (decayed[q] ?? 0))
  const phi = toFlat(irfft({ re: tensor(dRe), im: tensor(dIm) }, fine))
  // d/dx multiplies mode q by 2 pi i q (the Nyquist mode carries no derivative).
  const xRe = dIm.map((v, q) => (q === fine / 2 ? 0 : -2 * Math.PI * q * v))
  const xIm = dRe.map((v, q) => (q === fine / 2 ? 0 : 2 * Math.PI * q * v))
  const phiX = toFlat(irfft({ re: tensor(xRe), im: tensor(xIm) }, fine))
  return phi.map((p, j) => (-2 * nu * (phiX[j] ?? 0)) / p)
}

/**
 * Operator-learning pairs for the periodic viscous Burgers equation (the Fourier neural operator benchmark): random
 * zero-mean Fourier-series initial conditions and the solution at time `T`, computed exactly by the Cole-Hopf
 * transform with spectral heat solves on a grid 8 times finer than the output (then subsampled).
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { burgersOperator } from 'omai-engine/datasets'
 * generate(burgersOperator, { knobs: { n: 2, points: 32 } }).data.y.shape // => [2, 32]
 * ```
 */
export const burgersOperator: DatasetGenerator<RegressionData> = {
  name: 'burgersOperator',
  kind: 'regression',
  description: 'Initial conditions and exact (Cole-Hopf) solutions of the periodic viscous Burgers equation.',
  knobs: dict(periodicKnobs),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['points'] as number
    const nu = knobs['viscosity'] as number
    const t = knobs['time'] as number
    const fine = 8 * m
    const grid = Array.from({ length: m }, (_, j) => j / m)
    const { a, b } = fourierCoefficients(k, n, knobs['modes'] as number, knobs['amplitude'] as number)
    const u0 = a.flatMap((ai, i) => evaluateSeries(ai, b[i] ?? [], grid))
    const uT = a.flatMap((ai, i) => burgersSolve(ai, b[i] ?? [], nu, t, fine).filter((_, j) => j % 8 === 0))
    return {
      data: xy(matrix(u0, n, m), matrix(uT, n, m), stamp('burgersOperator', burgersOperator.description, knobs)),
      truth: {
        noiseScale: 0,
        extra: {
          grid,
          coefficients: { a, b },
          solve: (ca: readonly number[], cb: readonly number[], time = t, resolution = fine) =>
            burgersSolve(ca, cb, nu, time, resolution),
        },
      },
    }
  },
}
