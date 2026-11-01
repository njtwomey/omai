/**
 * The finite-element method for elliptic and parabolic problems of the form -div(k grad u) + c u = f: assembly of
 * linear and quadratic elements in 1-D and linear triangles in 2-D (with a structured rectangle mesh), the steady solve
 * with Dirichlet values, the heat problem by the theta method, and L2 errors that show the convergence orders.
 */

import { MdmlError } from '../../base/index.ts'
import { type CsrMatrix, fromCoo } from '../../numerics/sparse/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { sparseLu, sparseLuSolve } from '../ode/_sparse-lu.ts'
import { csrApply } from './_linear.ts'
import type { PdeTraceOptions } from './finite-difference.ts'

/**
 * An assembled finite-element system for -div(k grad u) + c u = f: node coordinates (`[n]` in 1-D, `[n, 2]` in 2-D),
 * stiffness K, mass M (unweighted, for time derivatives and norms), the load F (including the reaction term in K) and
 * the boundary nodes.
 */
export interface FemSystem {
  readonly dim: 1 | 2
  readonly nodes: Float64Array
  readonly stiffness: CsrMatrix
  readonly mass: CsrMatrix
  readonly load: Float64Array
  readonly boundary: readonly number[]
  readonly size: number
  /** The nodes of each element (left to right in 1-D; the three vertices of each triangle in 2-D). */
  readonly elements: readonly (readonly number[])[]
  readonly order: 1 | 2
}

/** Coefficients of the problem: diffusion k(x), reaction c(x) and source f(x) (constants or functions). */
export interface FemCoefficients {
  readonly k?: number | ((...x: number[]) => number)
  readonly c?: number | ((...x: number[]) => number)
  readonly f?: number | ((...x: number[]) => number)
}

const coef = (v: number | ((...x: number[]) => number) | undefined, d: number, ...x: number[]): number =>
  v === undefined ? d : typeof v === 'number' ? v : v(...x)

// Gauss-Legendre with 4 points on [-1, 1]: exact to degree 7.
const GAUSS_X = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
const GAUSS_W = [0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385]

function shape1d(order: 1 | 2, xi: number): { n: number[]; dn: number[] } {
  if (order === 1) return { n: [(1 - xi) / 2, (1 + xi) / 2], dn: [-0.5, 0.5] }
  return { n: [(xi * (xi - 1)) / 2, 1 - xi * xi, (xi * (xi + 1)) / 2], dn: [xi - 0.5, -2 * xi, xi + 0.5] }
}

/**
 * Assemble 1-D linear (P1) or quadratic (P2) elements on a mesh of element end points, with 4-point Gauss quadrature
 * for the stiffness, reaction, mass and load integrals. P2 adds a node at each element midpoint (nodes are ordered
 * left to right).
 *
 * @param mesh - Increasing element end points.
 * @param coefficients - k, c and f.
 * @param order - 1 or 2 (default 1).
 * @returns The system; its boundary nodes are the two ends.
 * @throws MdmlError with code `value` for a mesh that is not increasing.
 * @concept finite-element
 * @example
 * ```ts
 * import { assemble1d } from 'mdml-engine/dynamics/pde'
 * assemble1d([0, 0.5, 1], { f: 1 }, 2).size // => 5
 * ```
 */
export function assemble1d(mesh: readonly number[], coefficients: FemCoefficients = {}, order: 1 | 2 = 1): FemSystem {
  if (mesh.length < 2 || mesh.some((x, i) => i > 0 && !(x > (mesh[i - 1] ?? 0)))) {
    throw new MdmlError('value', 'assemble1d: the mesh must be increasing with at least one element')
  }
  const elements = mesh.length - 1
  const n = order * elements + 1
  const nodes = new Float64Array(n)
  for (let e = 0; e < elements; e++) {
    nodes[order * e] = mesh[e] ?? 0
    if (order === 2) nodes[2 * e + 1] = 0.5 * ((mesh[e] ?? 0) + (mesh[e + 1] ?? 0))
  }
  nodes[n - 1] = mesh[elements] ?? 0
  const kr: number[] = []
  const kc: number[] = []
  const kv: number[] = []
  const mv: number[] = []
  const load = new Float64Array(n)
  const elementNodes: number[][] = []
  for (let e = 0; e < elements; e++) {
    const x0 = mesh[e] ?? 0
    const x1 = mesh[e + 1] ?? 0
    const jac = (x1 - x0) / 2
    const local = Array.from({ length: order + 1 }, (_, i) => order * e + i)
    elementNodes.push(local)
    const ke = local.map(() => local.map(() => 0))
    const me = local.map(() => local.map(() => 0))
    for (let q = 0; q < 4; q++) {
      const xi = GAUSS_X[q] ?? 0
      const w = (GAUSS_W[q] ?? 0) * jac
      const x = x0 + (xi + 1) * jac
      const { n: N, dn } = shape1d(order, xi)
      const k = coef(coefficients.k, 1, x)
      const c = coef(coefficients.c, 0, x)
      const f = coef(coefficients.f, 0, x)
      for (let a = 0; a <= order; a++) {
        load[local[a] as number] = (load[local[a] as number] ?? 0) + w * f * (N[a] ?? 0)
        for (let b = 0; b <= order; b++) {
          const dNa = (dn[a] ?? 0) / jac
          const dNb = (dn[b] ?? 0) / jac
          ;(ke[a] as number[])[b] = (ke[a]?.[b] ?? 0) + w * (k * dNa * dNb + c * (N[a] ?? 0) * (N[b] ?? 0))
          ;(me[a] as number[])[b] = (me[a]?.[b] ?? 0) + w * (N[a] ?? 0) * (N[b] ?? 0)
        }
      }
    }
    local.forEach((ga, a) => {
      local.forEach((gb, b) => {
        kr.push(ga)
        kc.push(gb)
        kv.push(ke[a]?.[b] ?? 0)
        mv.push(me[a]?.[b] ?? 0)
      })
    })
  }
  return {
    dim: 1,
    nodes,
    stiffness: fromCoo(kr, kc, kv, [n, n]),
    mass: fromCoo(kr, kc, mv, [n, n]),
    load,
    boundary: [0, n - 1],
    size: n,
    elements: elementNodes,
    order,
  }
}

/** A triangle mesh: points `[np, 2]` (flat), triangles `[nt, 3]` (flat) and boundary node indices. */
export interface TriangleMesh {
  readonly points: Float64Array
  readonly triangles: Int32Array
  readonly boundary: readonly number[]
}

/**
 * A structured triangulation of a rectangle: (nx - 1)(ny - 1) squares each cut into two triangles; node (i, j) has
 * index i * ny + j.
 *
 * @param xRange - [ax, bx].
 * @param nx - Nodes in x (>= 2).
 * @param yRange - [ay, by].
 * @param ny - Nodes in y (>= 2).
 * @returns The mesh.
 * @throws MdmlError with code `value` for fewer than two nodes per side.
 * @example
 * ```ts
 * import { rectangleMesh } from 'mdml-engine/dynamics/pde'
 * rectangleMesh([0, 1], 3, [0, 1], 3).triangles.length / 3 // => 8
 * ```
 */
export function rectangleMesh(
  xRange: readonly [number, number],
  nx: number,
  yRange: readonly [number, number],
  ny: number,
): TriangleMesh {
  if (nx < 2 || ny < 2) throw new MdmlError('value', 'rectangleMesh: at least two nodes per side')
  const points = new Float64Array(nx * ny * 2)
  const boundary: number[] = []
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const k = i * ny + j
      points[2 * k] = xRange[0] + ((xRange[1] - xRange[0]) * i) / (nx - 1)
      points[2 * k + 1] = yRange[0] + ((yRange[1] - yRange[0]) * j) / (ny - 1)
      if (i === 0 || j === 0 || i === nx - 1 || j === ny - 1) boundary.push(k)
    }
  }
  const tri: number[] = []
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const a = i * ny + j
      const b = (i + 1) * ny + j
      const c = (i + 1) * ny + j + 1
      const d = i * ny + j + 1
      tri.push(a, b, c, a, c, d)
    }
  }
  return { points, triangles: Int32Array.from(tri), boundary }
}

/**
 * Assemble linear (P1) triangles: element stiffness from the constant gradients, the exact P1 mass matrix and a
 * three-point (edge-midpoint) rule for the variable coefficients and the load.
 *
 * @param mesh - The triangulation.
 * @param coefficients - k, c and f as functions of (x, y) or constants.
 * @returns The system.
 * @concept finite-element
 * @example
 * ```ts
 * import { assemble2d, rectangleMesh } from 'mdml-engine/dynamics/pde'
 * assemble2d(rectangleMesh([0, 1], 3, [0, 1], 3), { f: 1 }).size // => 9
 * ```
 */
export function assemble2d(mesh: TriangleMesh, coefficients: FemCoefficients = {}): FemSystem {
  const np = mesh.points.length / 2
  const kr: number[] = []
  const kc: number[] = []
  const kv: number[] = []
  const mv: number[] = []
  const load = new Float64Array(np)
  const P = (k: number): [number, number] => [mesh.points[2 * k] ?? 0, mesh.points[2 * k + 1] ?? 0]
  for (let t = 0; t < mesh.triangles.length / 3; t++) {
    const v = [mesh.triangles[3 * t] ?? 0, mesh.triangles[3 * t + 1] ?? 0, mesh.triangles[3 * t + 2] ?? 0]
    const [p0, p1, p2] = v.map(P) as [[number, number], [number, number], [number, number]]
    const det = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    const area = Math.abs(det) / 2
    if (area === 0) throw new MdmlError('value', `assemble2d: triangle ${t} is degenerate`)
    // Gradients of the barycentric coordinates.
    const b = [p1[1] - p2[1], p2[1] - p0[1], p0[1] - p1[1]].map((x) => x / det)
    const c = [p2[0] - p1[0], p0[0] - p2[0], p1[0] - p0[0]].map((x) => x / det)
    // Edge midpoints: exact for quadratics.
    const mids = [
      [0.5, 0.5, 0],
      [0, 0.5, 0.5],
      [0.5, 0, 0.5],
    ]
    let kAvg = 0
    const reaction = [0, 1, 2].map(() => [0, 0, 0])
    for (const m of mids) {
      const x = (m[0] ?? 0) * p0[0] + (m[1] ?? 0) * p1[0] + (m[2] ?? 0) * p2[0]
      const y = (m[0] ?? 0) * p0[1] + (m[1] ?? 0) * p1[1] + (m[2] ?? 0) * p2[1]
      kAvg += coef(coefficients.k, 1, x, y) / 3
      const cv = coef(coefficients.c, 0, x, y)
      const fv = coef(coefficients.f, 0, x, y)
      for (let a = 0; a < 3; a++) {
        load[v[a] as number] = (load[v[a] as number] ?? 0) + (area / 3) * fv * (m[a] ?? 0)
        for (let q = 0; q < 3; q++)
          (reaction[a] as number[])[q] = (reaction[a]?.[q] ?? 0) + (area / 3) * cv * (m[a] ?? 0) * (m[q] ?? 0)
      }
    }
    for (let a = 0; a < 3; a++) {
      for (let q = 0; q < 3; q++) {
        kr.push(v[a] as number)
        kc.push(v[q] as number)
        kv.push(kAvg * area * ((b[a] ?? 0) * (b[q] ?? 0) + (c[a] ?? 0) * (c[q] ?? 0)) + (reaction[a]?.[q] ?? 0))
        mv.push((area / 12) * (a === q ? 2 : 1))
      }
    }
  }
  return {
    dim: 2,
    nodes: Float64Array.from(mesh.points),
    stiffness: fromCoo(kr, kc, kv, [np, np]),
    mass: fromCoo(kr, kc, mv, [np, np]),
    load,
    boundary: mesh.boundary,
    size: np,
    elements: Array.from({ length: mesh.triangles.length / 3 }, (_, t) => [
      mesh.triangles[3 * t] ?? 0,
      mesh.triangles[3 * t + 1] ?? 0,
      mesh.triangles[3 * t + 2] ?? 0,
    ]),
    order: 1,
  }
}

function coordsOf(system: FemSystem, k: number): number[] {
  return system.dim === 1 ? [system.nodes[k] ?? 0] : [system.nodes[2 * k] ?? 0, system.nodes[2 * k + 1] ?? 0]
}

function dirichletRows(
  system: FemSystem,
  matrix: CsrMatrix,
  rhs: Float64Array,
  g: number | ((...x: number[]) => number),
): ReturnType<typeof sparseLu> {
  const fixed = new Set(system.boundary)
  const n = system.size
  const gv = new Float64Array(n)
  for (const k of fixed) gv[k] = coef(g, 0, ...coordsOf(system, k))
  // Move the known boundary values to the right-hand side and put identity rows in their place.
  const lifted = csrApply(matrix, gv)
  const rows: number[] = []
  const cols: number[] = []
  const vals: number[] = []
  const values = (matrix.values as Tensor).data
  for (let r = 0; r < n; r++) {
    if (fixed.has(r)) {
      rows.push(r)
      cols.push(r)
      vals.push(1)
      rhs[r] = gv[r] ?? 0
      continue
    }
    rhs[r] = (rhs[r] ?? 0) - (lifted[r] ?? 0)
    for (let e = matrix.indptr[r] ?? 0; e < (matrix.indptr[r + 1] ?? 0); e++) {
      const c = matrix.indices[e] ?? 0
      if (fixed.has(c)) continue
      rows.push(r)
      cols.push(c)
      vals.push(values[e] ?? 0)
    }
  }
  return sparseLu(n, rows, cols, vals)
}

/**
 * Solve the assembled elliptic problem K u = F with Dirichlet values g on the boundary nodes (the steady state of the
 * problem; for the Poisson equation -Delta u = f take k = 1, c = 0).
 *
 * @param system - From {@link assemble1d} or {@link assemble2d}.
 * @param g - The boundary values (a constant or a function of the node coordinates; default 0).
 * @returns The nodal values.
 * @concept finite-element
 * @example
 * ```ts
 * import { assemble1d, solveFem } from 'mdml-engine/dynamics/pde'
 * // -u'' = 2 on [0, 1], u = 0 at both ends: u = x (1 - x), exact at the nodes for P1.
 * Math.abs((solveFem(assemble1d([0, 0.25, 0.5, 0.75, 1], { f: 2 })).data[2] ?? 0) - 0.25) < 1e-12 // => true
 * ```
 */
export function solveFem(system: FemSystem, g: number | ((...x: number[]) => number) = 0): Tensor {
  const rhs = Float64Array.from(system.load)
  const lu = dirichletRows(system, system.stiffness, rhs, g)
  return fromStorage(sparseLuSolve(lu, rhs), [system.size], 'float64')
}

/** Options of {@link heatFem}. */
export interface HeatFemOptions extends PdeTraceOptions {
  readonly dt: number
  readonly tEnd: number
  /** 1 for backward Euler, 0.5 for Crank-Nicolson (default). */
  readonly theta?: number
  /** Dirichlet values on the boundary (default 0). */
  readonly g?: number | ((...x: number[]) => number)
  readonly every?: number
}

/** The state of {@link heatFemAlgorithm}. */
export interface HeatFemState {
  /** Steps taken. */
  readonly step: number
  /** The time reached and the nodal values there. */
  readonly t: number
  readonly u: Float64Array
  /** The snapshots kept (every n-th step and the last). */
  readonly ts: readonly number[]
  readonly us: readonly Float64Array[]
}

/** The paths {@link heatFemAlgorithm} logs, as a type. */
export type HeatFemPath = 'pde/t' | 'pde/u' | 'pde/norm'

/** What {@link heatFemAlgorithm} logs, by path: the time, the nodal values and their l2 norm. */
export const heatFemContract: Contract<HeatFemState, HeatFemPath> = defineContract({
  'pde/t': scalar({ description: 'the time reached' }),
  'pde/u': tensor({ role: 'state', shape: ['node'], description: 'the nodal values' }),
  'pde/norm': scalar({ description: 'the l2 norm of the nodal values' }),
})

/**
 * The finite-element heat problem M u_t + K u = F as a steppable algorithm, by the theta method: (M + theta dt K) u^{n+1} =
 * (M - (1 - theta) dt K) u^n + dt F, with Dirichlet boundary values; the matrix is factorised once.
 *
 * @param system - The assembled system.
 * @param u0 - Initial nodal values (or a function of the coordinates).
 * @param options - Step, final time, theta, boundary values and thinning.
 * @returns The algorithm; it is done after the last step. It logs the paths of {@link heatFemContract}.
 * @concept finite-element
 * @example
 * ```ts
 * import { assemble1d, heatFemAlgorithm } from 'mdml-engine/dynamics/pde'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const mesh = Array.from({ length: 11 }, (_, i) => i / 10)
 * runAlgorithm(heatFemAlgorithm(assemble1d(mesh), (x) => Math.sin(Math.PI * x), { dt: 0.01, tEnd: 0.05 }), { maxSteps: 5 }).state.t // => 0.05
 * ```
 */
export function heatFemAlgorithm(
  system: FemSystem,
  u0: readonly number[] | ((...x: number[]) => number),
  options: HeatFemOptions,
): Algorithm<HeatFemState, HeatFemPath> {
  const n = system.size
  const theta = options.theta ?? 0.5
  const dt = options.dt
  if (!(dt > 0)) throw new MdmlError('value', 'heatFem: dt must be positive')
  const start = Float64Array.from({ length: n }, (_, k) =>
    typeof u0 === 'function' ? u0(...coordsOf(system, k)) : (u0[k] ?? 0),
  )
  const kRows = (system.stiffness.values as Tensor).data
  const mRows = (system.mass.values as Tensor).data
  // A = M + theta dt K (same sparsity pattern as K and M, which share their triplets).
  const combined = (wm: number, wk: number): CsrMatrix => {
    const rows: number[] = []
    const cols: number[] = []
    const vals: number[] = []
    for (const [mat, w, data] of [
      [system.mass, wm, mRows],
      [system.stiffness, wk, kRows],
    ] as const) {
      for (let r = 0; r < n; r++) {
        for (let e = mat.indptr[r] ?? 0; e < (mat.indptr[r + 1] ?? 0); e++) {
          rows.push(r)
          cols.push(mat.indices[e] ?? 0)
          vals.push(w * (data[e] ?? 0))
        }
      }
    }
    return fromCoo(rows, cols, vals, [n, n])
  }
  const lhs = combined(1, theta * dt)
  const explicitPart = combined(1, -(1 - theta) * dt)
  const g = options.g ?? 0
  const lu = dirichletRows(system, lhs, new Float64Array(n), g)
  const steps = Math.ceil(options.tEnd / dt - 1e-9)
  const every = Math.max(1, options.every ?? 1)
  return {
    name: 'heat-fem',
    config: { dt, theta, tEnd: options.tEnd, steps },
    schema: channels({
      metrics: { t: { description: 'time' } },
      tensors: { u: { role: 'state', description: 'nodal values' } },
    }),
    init: () => ({ step: 0, u: start, t: 0, ts: [0], us: [start] }),
    step: (s, ctx) => {
      const rhs = csrApply(explicitPart, s.u)
      for (let k = 0; k < n; k++) rhs[k] = (rhs[k] ?? 0) + dt * (system.load[k] ?? 0)
      // Lift the boundary values exactly as in the factorised matrix.
      const lifted = new Float64Array(n)
      const fixed = new Set(system.boundary)
      const gv = new Float64Array(n)
      for (const k of fixed) gv[k] = coef(g, 0, ...coordsOf(system, k))
      const lg = csrApply(lhs, gv)
      for (let k = 0; k < n; k++) lifted[k] = fixed.has(k) ? (gv[k] ?? 0) : (rhs[k] ?? 0) - (lg[k] ?? 0)
      const u = sparseLuSolve(lu, lifted)
      const t = ctx.t + 1 === steps ? options.tEnd : s.t + dt
      const keep = (ctx.t + 1) % every === 0 || ctx.t + 1 === steps
      if (ctx.wants('pde/t')) ctx.log('pde/t', t)
      if (ctx.wants('pde/u')) ctx.log('pde/u', u)
      if (ctx.wants('pde/norm')) ctx.log('pde/norm', Math.hypot(...u))
      return { step: s.step + 1, u, t, ts: keep ? [...s.ts, t] : s.ts, us: keep ? [...s.us, u] : s.us }
    },
    done: (s) => (s.step >= steps ? 'converged' : undefined),
    observe: (s) => ({ metrics: { t: s.t }, tensors: { u: { role: 'state', value: s.u } } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HeatFemState,
  }
}

/**
 * The finite-element heat problem M u_t + K u = F by the theta method (see {@link heatFemAlgorithm}): (M + theta dt K) u^{n+1} =
 * (M - (1 - theta) dt K) u^n + dt F, with Dirichlet boundary values; the matrix is factorised once.
 *
 * @param system - The assembled system.
 * @param u0 - Initial nodal values (or a function of the coordinates).
 * @param options - Step, final time, theta, boundary values and thinning.
 * @returns Snapshot times and nodal values `[nt, n]`.
 * @concept finite-element
 * @example
 * ```ts
 * import { assemble1d, heatFem } from 'mdml-engine/dynamics/pde'
 * const mesh = Array.from({ length: 21 }, (_, i) => i / 20)
 * const s = heatFem(assemble1d(mesh), (x) => Math.sin(Math.PI * x), { dt: 0.005, tEnd: 0.1 })
 * Math.abs((s.u.data[s.u.data.length - 11] ?? 0) - Math.exp(-(Math.PI ** 2) * 0.1)) < 2e-3 // => true
 * ```
 */
export function heatFem(
  system: FemSystem,
  u0: readonly number[] | ((...x: number[]) => number),
  options: HeatFemOptions,
): { readonly t: readonly number[]; readonly u: Tensor } {
  const steps = Math.ceil(options.tEnd / options.dt - 1e-9)
  const run = runAlgorithm(heatFemAlgorithm(system, u0, options), { maxSteps: steps, ...options })
  const n = system.size
  const flat = new Float64Array(run.state.us.length * n)
  run.state.us.forEach((u, i) => {
    flat.set(u, i * n)
  })
  return { t: [...run.state.ts], u: fromStorage(flat, [run.state.us.length, n], 'float64') }
}

/**
 * The L2 norm of u_h - u over the domain, integrated element by element (4-point Gauss in 1-D, a 7-point rule on
 * triangles), so it shows the true convergence orders (h^2 for P1, h^3 for P2).
 *
 * @param system - The assembled system.
 * @param u - Nodal values.
 * @param exact - The exact solution as a function of the coordinates.
 * @returns The error norm.
 * @concept finite-element
 * @example
 * ```ts
 * import { assemble1d, femL2Error } from 'mdml-engine/dynamics/pde'
 * femL2Error(assemble1d([0, 1]), [0, 1], (x) => x) // => 0
 * ```
 */
export function femL2Error(
  system: FemSystem,
  u: ArrayLike<number> | Tensor,
  exact: (...x: number[]) => number,
): number {
  const vals = 'data' in u ? (u as Tensor).data : (u as ArrayLike<number>)
  let s = 0
  if (system.dim === 1) {
    for (const el of system.elements) {
      const x0 = system.nodes[el[0] ?? 0] ?? 0
      const x1 = system.nodes[el[el.length - 1] ?? 0] ?? 0
      const jac = (x1 - x0) / 2
      for (let q = 0; q < 4; q++) {
        const xi = GAUSS_X[q] ?? 0
        const { n: N } = shape1d(system.order, xi)
        let uh = 0
        el.forEach((g, a) => {
          uh += (N[a] ?? 0) * (vals[g] ?? 0)
        })
        s += (GAUSS_W[q] ?? 0) * jac * (uh - exact(x0 + (xi + 1) * jac)) ** 2
      }
    }
    return Math.sqrt(s)
  }
  // Strang-Fix 7-point rule (degree 5) in barycentric coordinates.
  const a1 = 0.0597158717
  const b1 = 0.4701420641
  const a2 = 0.7974269853
  const b2 = 0.1012865073
  const rule: [number, number, number, number][] = [
    [1 / 3, 1 / 3, 1 / 3, 0.225],
    [a1, b1, b1, 0.1323941527],
    [b1, a1, b1, 0.1323941527],
    [b1, b1, a1, 0.1323941527],
    [a2, b2, b2, 0.1259391805],
    [b2, a2, b2, 0.1259391805],
    [b2, b2, a2, 0.1259391805],
  ]
  for (const el of system.elements) {
    const p = el.map((g) => [system.nodes[2 * g] ?? 0, system.nodes[2 * g + 1] ?? 0]) as [number, number][]
    const [p0, p1, p2] = p as [[number, number], [number, number], [number, number]]
    const area = Math.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2
    for (const [l0, l1, l2, w] of rule) {
      const x = l0 * p0[0] + l1 * p1[0] + l2 * p2[0]
      const y = l0 * p0[1] + l1 * p1[1] + l2 * p2[1]
      const uh = l0 * (vals[el[0] ?? 0] ?? 0) + l1 * (vals[el[1] ?? 0] ?? 0) + l2 * (vals[el[2] ?? 0] ?? 0)
      s += w * area * (uh - exact(x, y)) ** 2
    }
  }
  return Math.sqrt(s)
}
