/**
 * The method of lines: discretise a PDE in space only, turning it into a system of ODEs for the grid values that any
 * ODE solver can integrate. First-derivative matrices, and semi-discretisations of 1-D advection-diffusion-reaction and
 * 2-D reaction-diffusion equations.
 */

import { MdmlError } from '../../base/index.ts'
import { type CsrMatrix, fromCoo, spmv } from '../../numerics/sparse/index.ts'
import { add, fromStorage, mul, type Tensor, type Value } from '../../tensor/index.ts'
import type { VectorField } from '../ode/problem.ts'
import {
  type BoundaryCondition,
  boundaryValue,
  type DiscreteOperator,
  type Grid1d,
  type Grid2d,
  laplacian1d,
  laplacian2d,
  type RectangleConditions,
} from './grids.ts'

/**
 * The first-derivative matrix on a 1-D grid: second-order central differences in the interior (or first-order upwind
 * for a velocity sign), one-sided second-order differences at the ends, or wrap-around when periodic.
 *
 * @param grid - The grid.
 * @param options - `central` (default) or `upwind` with the velocity's sign, and periodicity.
 * @returns The matrix `[n, n]`.
 * @concept finite-difference
 * @example
 * ```ts
 * import { firstDerivative1d, grid1d } from 'mdml-engine/dynamics/pde'
 * import { spmv } from 'mdml-engine/numerics/sparse'
 * import { toFlat } from 'mdml-engine/tensor'
 * const g = grid1d(0, 1, 5)
 * toFlat(spmv(firstDerivative1d(g), Array.from(g.x, (x) => x * x))).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0.5, 1, 1.5, 2]
 * ```
 */
export function firstDerivative1d(
  grid: Grid1d,
  options: { readonly scheme?: 'central' | 'upwind'; readonly velocity?: number; readonly periodic?: boolean } = {},
): CsrMatrix {
  const n = grid.n
  const h = grid.dx
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  const put = (r: number, c: number, v: number): void => {
    rows.push(r)
    cols.push(c)
    vals.push(v)
  }
  const upwind = options.scheme === 'upwind'
  const forward = (options.velocity ?? 1) < 0
  if (options.periodic === true) {
    const m = n - 1
    for (let i = 0; i < n; i++) {
      const j = i % m
      if (upwind) {
        if (forward) {
          put(i, (j + 1) % m, 1 / h)
          put(i, j, -1 / h)
        } else {
          put(i, j, 1 / h)
          put(i, (j - 1 + m) % m, -1 / h)
        }
      } else {
        put(i, (j + 1) % m, 0.5 / h)
        put(i, (j - 1 + m) % m, -0.5 / h)
      }
    }
    return fromCoo(rows, cols, vals, [n, n])
  }
  put(0, 0, -1.5 / h)
  put(0, 1, 2 / h)
  put(0, 2, -0.5 / h)
  for (let i = 1; i < n - 1; i++) {
    if (upwind) {
      if (forward) {
        put(i, i + 1, 1 / h)
        put(i, i, -1 / h)
      } else {
        put(i, i, 1 / h)
        put(i, i - 1, -1 / h)
      }
    } else {
      put(i, i + 1, 0.5 / h)
      put(i, i - 1, -0.5 / h)
    }
  }
  put(n - 1, n - 1, 1.5 / h)
  put(n - 1, n - 2, -2 / h)
  put(n - 1, n - 3, 0.5 / h)
  return fromCoo(rows, cols, vals, [n, n])
}

/** A semi-discretised PDE: the ODE system for {@link solveIvp}, with its grid and operator. */
export interface MolSystem {
  readonly field: VectorField
  readonly size: number
  readonly operator: DiscreteOperator
  /** Node coordinates (`[n]`, or `[nx * ny, 2]` flat in 2-D). */
  readonly x: Float64Array
}

/** Terms of a 1-D advection-diffusion-reaction equation u_t = D u_xx - a u_x + R(u, x, t) + s(x, t). */
export interface MolOptions1d {
  readonly diffusion?: number
  readonly velocity?: number
  readonly advectionScheme?: 'central' | 'upwind'
  /** A reaction term on the whole state (built from operations, so the system stays differentiable). */
  readonly reaction?: (u: Value, x: Tensor, t: number) => Value
  readonly source?: (x: number, t: number) => number
  readonly left: BoundaryCondition
  readonly right: BoundaryCondition
}

function boundaryRate(bc: BoundaryCondition, t: number, at: readonly number[]): number {
  // Dirichlet nodes are states that follow g(t): du/dt = g'(t), by a central difference of g.
  if (bc.kind !== 'dirichlet' || typeof bc.value === 'number') return 0
  const h = 1e-6 * Math.max(1, Math.abs(t))
  return (boundaryValue(bc, t + h, ...at) - boundaryValue(bc, t - h, ...at)) / (2 * h)
}

/**
 * The method of lines in 1-D: the advection-diffusion-reaction equation u_t = D u_xx - a u_x + R(u) + s becomes the
 * ODE system du/dt = F(t, u) on the grid nodes (Dirichlet nodes follow their boundary value; Neumann, Robin and
 * periodic conditions enter the stencil), ready for {@link solveIvp} with any method (stiff ones for fine grids). The
 * field is built from operations, so solutions are differentiable in the initial condition and in anything the
 * reaction closes over.
 *
 * @param grid - The grid.
 * @param options - Coefficients, terms and boundary conditions.
 * @returns The ODE system.
 * @throws MdmlError with code `value` when the velocity needs an upwind side the conditions do not allow.
 * @concept method-of-lines
 * @example
 * ```ts
 * import { dirichlet, grid1d, semiDiscretise } from 'mdml-engine/dynamics/pde'
 * import { solveIvp } from 'mdml-engine/dynamics/ode'
 * import { toFlat } from 'mdml-engine/tensor'
 * const g = grid1d(0, 1, 21)
 * const mol = semiDiscretise(g, { diffusion: 1, left: dirichlet(), right: dirichlet() })
 * const u = toFlat(solveIvp(mol.field, [0, 0.1], Array.from(g.x, (x) => Math.sin(Math.PI * x)), { method: 'BDF', rtol: 1e-8, atol: 1e-10, tEval: [0.1] }).y)
 * Math.abs((u[10] ?? 0) - Math.exp(-(Math.PI ** 2) * 0.1)) < 2e-3 // => true
 * ```
 */
export function semiDiscretise(grid: Grid1d, options: MolOptions1d): MolSystem {
  const D = options.diffusion ?? 0
  const a = options.velocity ?? 0
  const periodicBc = options.left.kind === 'periodic'
  const op = laplacian1d(grid, options.left, options.right, 1)
  const deriv =
    a === 0
      ? undefined
      : firstDerivative1d(grid, { scheme: options.advectionScheme ?? 'central', velocity: a, periodic: periodicBc })
  const x = fromStorage(Float64Array.from(grid.x), [grid.n], 'float64')
  const fixed = op.fixed
  const n = grid.n
  const freeMask = fromStorage(
    Float64Array.from({ length: n }, (_, i) => (fixed.some((f) => f.index === i) ? 0 : 1)),
    [n],
    'float64',
  )
  const field: VectorField = (t, u) => {
    let du: Value = mul(add(spmv(op.matrix, u), fromStorage(op.forcing(t), [n], 'float64')), D)
    if (deriv !== undefined) du = add(du, mul(spmv(deriv, u), -a))
    if (options.reaction !== undefined) du = add(du, options.reaction(u, x, t))
    if (options.source !== undefined) {
      du = add(
        du,
        fromStorage(
          Float64Array.from(grid.x, (xi) => (options.source as NonNullable<MolOptions1d['source']>)(xi, t)),
          [n],
          'float64',
        ),
      )
    }
    du = mul(du, freeMask)
    if (fixed.length > 0) {
      const rates = new Float64Array(n)
      for (const f of fixed) rates[f.index] = boundaryRate(f.bc, t, f.at)
      du = add(du, fromStorage(rates, [n], 'float64'))
    }
    return du
  }
  return { field, size: n, operator: op, x: Float64Array.from(grid.x) }
}

/**
 * The method of lines in 2-D for reaction-diffusion u_t = D Delta u + R(u, t) on a tensor grid (5-point Laplacian with
 * the four side conditions; Dirichlet nodes follow their values).
 *
 * @param grid - The grid.
 * @param options - Diffusion, reaction and the side conditions.
 * @returns The ODE system over the flattened nodes.
 * @throws MdmlError with code `value` for a negative diffusion coefficient.
 * @concept method-of-lines
 * @example
 * ```ts
 * import { dirichlet, grid2d, semiDiscretise2d } from 'mdml-engine/dynamics/pde'
 * const d = dirichlet()
 * semiDiscretise2d(grid2d([0, 1], 5, [0, 1], 5), { diffusion: 1, bcs: { left: d, right: d, bottom: d, top: d } }).size // => 25
 * ```
 */
export function semiDiscretise2d(
  grid: Grid2d,
  options: {
    readonly diffusion?: number
    readonly reaction?: (u: Value, t: number) => Value
    readonly bcs: RectangleConditions
  },
): MolSystem {
  const D = options.diffusion ?? 1
  if (D < 0) throw new MdmlError('value', 'semiDiscretise2d: the diffusion coefficient must be non-negative')
  const op = laplacian2d(grid, options.bcs)
  const n = grid.size
  const fixed = new Set(op.fixed.map((f) => f.index))
  const freeMask = fromStorage(
    Float64Array.from({ length: n }, (_, i) => (fixed.has(i) ? 0 : 1)),
    [n],
    'float64',
  )
  const field: VectorField = (t, u) => {
    let du: Value = mul(add(spmv(op.matrix, u), fromStorage(op.forcing(t), [n], 'float64')), D)
    if (options.reaction !== undefined) du = add(du, options.reaction(u, t))
    du = mul(du, freeMask)
    const rates = new Float64Array(n)
    for (const f of op.fixed) rates[f.index] = boundaryRate(f.bc, t, f.at)
    return add(du, fromStorage(rates, [n], 'float64'))
  }
  const x = new Float64Array(2 * n)
  for (let i = 0; i < grid.nx; i++) {
    for (let j = 0; j < grid.ny; j++) {
      x[2 * (i * grid.ny + j)] = grid.x.x[i] ?? 0
      x[2 * (i * grid.ny + j) + 1] = grid.y.x[j] ?? 0
    }
  }
  return { field, size: n, operator: op, x }
}
