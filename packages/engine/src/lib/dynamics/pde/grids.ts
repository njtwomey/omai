/**
 * Grids, boundary conditions and discrete operators for the PDE solvers: uniform 1-D and 2-D grids, Dirichlet, Neumann,
 * Robin and periodic conditions, and the second-order Laplacian matrices (3-point and 5-point stencils with ghost
 * points) that carry their boundary forcing.
 */

import { MdmlError } from '../../base/index.ts'
import { type CsrMatrix, fromCoo } from '../../numerics/sparse/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'

/** A uniform 1-D grid of n nodes on [a, b], boundaries included. */
export interface Grid1d {
  readonly a: number
  readonly b: number
  readonly n: number
  readonly dx: number
  readonly x: Float64Array
}

/** A uniform 2-D tensor grid; node (i, j) is x_i, y_j and has flat index i * ny + j. */
export interface Grid2d {
  readonly x: Grid1d
  readonly y: Grid1d
  readonly nx: number
  readonly ny: number
  readonly size: number
}

/**
 * A uniform grid of n >= 2 nodes on [a, b].
 *
 * @param a - Left end.
 * @param b - Right end (b > a).
 * @param n - Number of nodes.
 * @returns The grid.
 * @throws MdmlError with code `value` for fewer than two nodes or an empty interval.
 * @concept finite-difference
 * @example
 * ```ts
 * import { grid1d } from 'mdml-engine/dynamics/pde'
 * grid1d(0, 1, 5).dx // => 0.25
 * ```
 */
export function grid1d(a: number, b: number, n: number): Grid1d {
  if (!Number.isInteger(n) || n < 2) throw new MdmlError('value', 'grid1d: at least two nodes')
  if (!(b > a)) throw new MdmlError('value', 'grid1d: need b > a')
  const dx = (b - a) / (n - 1)
  return { a, b, n, dx, x: Float64Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a + i * dx)) }
}

/**
 * A uniform tensor grid on [ax, bx] x [ay, by].
 *
 * @param xRange - [ax, bx].
 * @param nx - Nodes in x.
 * @param yRange - [ay, by].
 * @param ny - Nodes in y.
 * @returns The grid.
 * @example
 * ```ts
 * import { grid2d } from 'mdml-engine/dynamics/pde'
 * grid2d([0, 1], 3, [0, 2], 5).size // => 15
 * ```
 */
export function grid2d(
  xRange: readonly [number, number],
  nx: number,
  yRange: readonly [number, number],
  ny: number,
): Grid2d {
  const x = grid1d(xRange[0], xRange[1], nx)
  const y = grid1d(yRange[0], yRange[1], ny)
  return { x, y, nx, ny, size: nx * ny }
}

/**
 * A boundary value: a constant, or a function of time and the boundary point's coordinates (x in 1-D, (x, y) in 2-D).
 */
export type BoundaryValue = number | ((t: number, ...x: number[]) => number)

/**
 * A boundary condition: Dirichlet u = g, Neumann du/dn = g (outward normal), Robin alpha u + beta du/dn = g, or
 * periodic.
 */
export type BoundaryCondition =
  | { readonly kind: 'dirichlet'; readonly value: BoundaryValue }
  | { readonly kind: 'neumann'; readonly value: BoundaryValue }
  | { readonly kind: 'robin'; readonly alpha: number; readonly beta: number; readonly value: BoundaryValue }
  | { readonly kind: 'periodic' }

/**
 * A Dirichlet condition u = value.
 *
 * @param value - The boundary value (or a function of time and position).
 * @returns The condition.
 * @example
 * ```ts
 * import { dirichlet } from 'mdml-engine/dynamics/pde'
 * dirichlet(1).kind // => 'dirichlet'
 * ```
 */
export function dirichlet(value: BoundaryValue = 0): BoundaryCondition {
  return { kind: 'dirichlet', value }
}

/**
 * A Neumann condition du/dn = value (the derivative along the outward normal).
 *
 * @param value - The flux (or a function of time and position).
 * @returns The condition.
 * @example
 * ```ts
 * import { neumann } from 'mdml-engine/dynamics/pde'
 * neumann().kind // => 'neumann'
 * ```
 */
export function neumann(value: BoundaryValue = 0): BoundaryCondition {
  return { kind: 'neumann', value }
}

/**
 * A Robin condition alpha u + beta du/dn = value.
 *
 * @param alpha - Coefficient of u.
 * @param beta - Coefficient of the outward normal derivative (non-zero).
 * @param value - The right-hand side (or a function of time and position).
 * @returns The condition.
 * @throws MdmlError with code `value` when beta is zero (that is a Dirichlet condition).
 * @example
 * ```ts
 * import { robin } from 'mdml-engine/dynamics/pde'
 * robin(1, 2, 0).kind // => 'robin'
 * ```
 */
export function robin(alpha: number, beta: number, value: BoundaryValue = 0): BoundaryCondition {
  if (beta === 0) throw new MdmlError('value', 'robin: beta = 0 is a Dirichlet condition')
  return { kind: 'robin', alpha, beta, value }
}

/**
 * Periodic boundaries (the first and last nodes are the same point; the last node is kept as a copy of the first).
 *
 * @returns The condition.
 * @example
 * ```ts
 * import { periodic } from 'mdml-engine/dynamics/pde'
 * periodic().kind // => 'periodic'
 * ```
 */
export function periodic(): BoundaryCondition {
  return { kind: 'periodic' }
}

/**
 * A boundary value at time t and a boundary point.
 *
 * @param bc - A condition with a value.
 * @param t - Time.
 * @param x - The boundary point's coordinates.
 * @returns The value (0 for periodic).
 * @example
 * ```ts
 * import { boundaryValue, dirichlet } from 'mdml-engine/dynamics/pde'
 * boundaryValue(dirichlet((t, x) => 2 * t + x), 3, 1) // => 7
 * ```
 */
export function boundaryValue(bc: BoundaryCondition, t: number, ...x: number[]): number {
  if (bc.kind === 'periodic') return 0
  return typeof bc.value === 'number' ? bc.value : bc.value(t, ...x)
}

/**
 * A discretised linear operator with its boundary data: L u + forcing(t) approximates the operator at every node not
 * held by a Dirichlet condition. Dirichlet nodes keep their values in the state (their columns are part of L, their
 * rows are zero) and are listed with their conditions and coordinates; Neumann and Robin data enter the forcing.
 */
export interface DiscreteOperator {
  readonly matrix: CsrMatrix
  readonly forcing: (t: number) => Float64Array
  /** Nodes fixed by Dirichlet conditions, with their conditions and coordinates. */
  readonly fixed: readonly { readonly index: number; readonly bc: BoundaryCondition; readonly at: readonly number[] }[]
  readonly size: number
}

interface Triplets {
  rows: number[]
  cols: number[]
  vals: number[]
}

function add(t: Triplets, r: number, c: number, v: number): void {
  t.rows.push(r)
  t.cols.push(c)
  t.vals.push(v)
}

/** One direction's stencil at node i of an n-node line: entries (column offset, weight) and a ghost forcing. */
function lineStencil(
  i: number,
  n: number,
  h: number,
  k: number,
  low: BoundaryCondition,
  high: BoundaryCondition,
): { entries: [number, number][]; ghost: ((t: number, at: readonly number[]) => number) | undefined } {
  const h2 = h * h
  if (low.kind === 'periodic') {
    const m = n - 1
    const j = i % m
    return {
      entries: [
        [(j - 1 + m) % m, k / h2],
        [j, (-2 * k) / h2],
        [(j + 1) % m, k / h2],
      ],
      ghost: undefined,
    }
  }
  if (i > 0 && i < n - 1) {
    return {
      entries: [
        [i - 1, k / h2],
        [i, (-2 * k) / h2],
        [i + 1, k / h2],
      ],
      ghost: undefined,
    }
  }
  const bc = i === 0 ? low : high
  const inner = i === 0 ? 1 : n - 2
  // Ghost node u_g = u_inner + 2 h du/dn, with du/dn from the condition.
  if (bc.kind === 'neumann') {
    return {
      entries: [
        [i, (-2 * k) / h2],
        [inner, (2 * k) / h2],
      ],
      ghost: (t, at) => (2 * k * h * boundaryValue(bc, t, ...at)) / h2,
    }
  }
  if (bc.kind === 'robin') {
    return {
      entries: [
        [i, (-2 * k) / h2 - (2 * k * h * bc.alpha) / (bc.beta * h2)],
        [inner, (2 * k) / h2],
      ],
      ghost: (t, at) => (2 * k * h * boundaryValue(bc, t, ...at)) / (bc.beta * h2),
    }
  }
  return { entries: [], ghost: undefined }
}

/**
 * The 1-D second-derivative operator d/dx (k d/dx) with constant coefficient k, second order: the 3-point stencil in
 * the interior, ghost points for Neumann and Robin conditions, wrap-around for periodic ones.
 *
 * @param grid - The grid.
 * @param left - The condition at a.
 * @param right - The condition at b.
 * @param k - The diffusion coefficient (default 1).
 * @returns The operator with its boundary forcing.
 * @throws MdmlError with code `value` when only one side is periodic.
 * @concept finite-difference
 * @example
 * ```ts
 * import { dirichlet, grid1d, laplacian1d } from 'mdml-engine/dynamics/pde'
 * laplacian1d(grid1d(0, 1, 5), dirichlet(), dirichlet()).fixed.length // => 2
 * ```
 */
export function laplacian1d(grid: Grid1d, left: BoundaryCondition, right: BoundaryCondition, k = 1): DiscreteOperator {
  if ((left.kind === 'periodic') !== (right.kind === 'periodic')) {
    throw new MdmlError('value', 'laplacian1d: periodic conditions go on both sides')
  }
  const n = grid.n
  const t: Triplets = { rows: [], cols: [], vals: [] }
  const fixed: { index: number; bc: BoundaryCondition; at: number[] }[] = []
  const ghosts: { index: number; at: number[]; g: (t: number, at: readonly number[]) => number }[] = []
  for (let i = 0; i < n; i++) {
    if (
      left.kind !== 'periodic' &&
      ((i === 0 && left.kind === 'dirichlet') || (i === n - 1 && right.kind === 'dirichlet'))
    ) {
      fixed.push({ index: i, bc: i === 0 ? left : right, at: [grid.x[i] ?? 0] })
      continue
    }
    const { entries, ghost } = lineStencil(i, n, grid.dx, k, left, right)
    for (const [c, v] of entries) add(t, i, c, v)
    if (ghost !== undefined) ghosts.push({ index: i, at: [grid.x[i] ?? 0], g: ghost })
  }
  return {
    matrix: fromCoo(t.rows, t.cols, t.vals, [n, n]),
    forcing: (time) => {
      const out = new Float64Array(n)
      for (const g of ghosts) out[g.index] = (out[g.index] ?? 0) + g.g(time, g.at)
      return out
    },
    fixed,
    size: n,
  }
}

/** Conditions on the four sides of a rectangle. */
export interface RectangleConditions {
  readonly left: BoundaryCondition
  readonly right: BoundaryCondition
  readonly bottom: BoundaryCondition
  readonly top: BoundaryCondition
}

/**
 * The 2-D Laplacian (5-point stencil) on a tensor grid, with Dirichlet, Neumann or Robin conditions per side (ghost
 * points; corners take the x-direction condition), or periodic pairs. Boundary values may depend on time and on the
 * boundary point (x, y).
 *
 * @param grid - The grid.
 * @param bcs - The four side conditions.
 * @param k - The diffusion coefficient (default 1).
 * @returns The operator with its boundary forcing.
 * @throws MdmlError with code `value` when only one side of a pair is periodic.
 * @concept finite-difference
 * @example
 * ```ts
 * import { dirichlet, grid2d, laplacian2d } from 'mdml-engine/dynamics/pde'
 * const d = dirichlet()
 * laplacian2d(grid2d([0, 1], 4, [0, 1], 4), { left: d, right: d, bottom: d, top: d }).fixed.length // => 12
 * ```
 */
export function laplacian2d(grid: Grid2d, bcs: RectangleConditions, k = 1): DiscreteOperator {
  if (
    (bcs.left.kind === 'periodic') !== (bcs.right.kind === 'periodic') ||
    (bcs.bottom.kind === 'periodic') !== (bcs.top.kind === 'periodic')
  ) {
    throw new MdmlError('value', 'laplacian2d: periodic conditions go on both sides of a pair')
  }
  const { nx, ny } = grid
  const idx = (i: number, j: number): number => i * ny + j
  const t: Triplets = { rows: [], cols: [], vals: [] }
  const fixed: { index: number; bc: BoundaryCondition; at: number[] }[] = []
  const ghosts: { index: number; at: number[]; g: (t: number, at: readonly number[]) => number }[] = []
  const fixedX = (i: number): BoundaryCondition | undefined =>
    bcs.left.kind === 'dirichlet' && i === 0
      ? bcs.left
      : bcs.right.kind === 'dirichlet' && i === nx - 1
        ? bcs.right
        : undefined
  const fixedY = (j: number): BoundaryCondition | undefined =>
    bcs.bottom.kind === 'dirichlet' && j === 0
      ? bcs.bottom
      : bcs.top.kind === 'dirichlet' && j === ny - 1
        ? bcs.top
        : undefined
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const r = idx(i, j)
      const at = [grid.x.x[i] ?? 0, grid.y.x[j] ?? 0]
      const f = fixedX(i) ?? fixedY(j)
      if (f !== undefined) {
        fixed.push({ index: r, bc: f, at })
        continue
      }
      const sx = lineStencil(i, nx, grid.x.dx, k, bcs.left, bcs.right)
      for (const [c, v] of sx.entries) add(t, r, idx(c, j), v)
      if (sx.ghost !== undefined) ghosts.push({ index: r, at, g: sx.ghost })
      const sy = lineStencil(j, ny, grid.y.dx, k, bcs.bottom, bcs.top)
      for (const [c, v] of sy.entries) add(t, r, idx(i, c), v)
      if (sy.ghost !== undefined) ghosts.push({ index: r, at, g: sy.ghost })
    }
  }
  return {
    matrix: fromCoo(t.rows, t.cols, t.vals, [grid.size, grid.size]),
    forcing: (time) => {
      const out = new Float64Array(grid.size)
      for (const g of ghosts) out[g.index] = (out[g.index] ?? 0) + g.g(time, g.at)
      return out
    },
    fixed,
    size: grid.size,
  }
}

/**
 * Overwrite the Dirichlet nodes of a state with their boundary values at time t.
 *
 * @param u - The nodal values.
 * @param op - The operator holding the conditions.
 * @param t - Time.
 * @returns The corrected values (a copy).
 * @example
 * ```ts
 * import { applyDirichlet, dirichlet, grid1d, laplacian1d } from 'mdml-engine/dynamics/pde'
 * const op = laplacian1d(grid1d(0, 1, 3), dirichlet(1), dirichlet(2))
 * Array.from(applyDirichlet([0, 0, 0], op, 0)) // => [1, 0, 2]
 * ```
 */
export function applyDirichlet(u: ArrayLike<number>, op: DiscreteOperator, t: number): Float64Array {
  const out = Float64Array.from(u)
  for (const f of op.fixed) out[f.index] = boundaryValue(f.bc, t, ...f.at)
  return out
}

/**
 * The nodes of a grid as a tensor (`[n]` for 1-D, `[nx, ny, 2]` coordinates for 2-D).
 *
 * @param grid - A grid.
 * @returns The coordinates.
 * @example
 * ```ts
 * import { grid1d, gridPoints } from 'mdml-engine/dynamics/pde'
 * Array.from(gridPoints(grid1d(0, 1, 3)).data) // => [0, 0.5, 1]
 * ```
 */
export function gridPoints(grid: Grid1d | Grid2d): Tensor {
  if ('a' in grid) return fromStorage(Float64Array.from(grid.x), [grid.n], 'float64')
  const out = new Float64Array(grid.size * 2)
  for (let i = 0; i < grid.nx; i++) {
    for (let j = 0; j < grid.ny; j++) {
      out[(i * grid.ny + j) * 2] = grid.x.x[i] ?? 0
      out[(i * grid.ny + j) * 2 + 1] = grid.y.x[j] ?? 0
    }
  }
  return fromStorage(out, [grid.nx, grid.ny, 2], 'float64')
}
