/**
 * Differential-algebraic equations: systems mixing differential equations with algebraic constraints. Semi-explicit
 * index-1 DAEs are solved by Radau IIA or BDF with a singular mass matrix, after making the initial algebraic variables
 * consistent by Newton's method.
 */

import { WhiteboxError } from '../../base/index.ts'
import { fsolve } from '../../numerics/roots/index.ts'
import { concat, reshape, sliceAxis, stack, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import type { JacobianOption } from './_newton.ts'
import { floatState, fromNumbers, numbersOf } from './_vec.ts'
import { bdf, radau } from './implicit.ts'
import type { EventSpec, OdeSolution, VectorField } from './problem.ts'
import { type IvpTraceOptions, solveIvp } from './solve-ivp.ts'

/**
 * A semi-explicit DAE of index 1: differential equations y' = f(t, y, z) and algebraic constraints 0 = g(t, y, z), with
 * dg/dz nonsingular along the solution.
 */
export interface SemiExplicitDae {
  readonly f: (t: number, y: Value, z: Value) => Value
  readonly g: (t: number, y: Value, z: Value) => Value
}

/**
 * Consistent initial algebraic variables: solve g(t0, y0, z) = 0 for z by Newton's method from a guess.
 *
 * @param dae - The DAE.
 * @param t0 - Initial time.
 * @param y0 - Initial differential state `[ny]`.
 * @param z0 - Guess for the algebraic state `[nz]`.
 * @param options - Residual tolerance (default 1e-12) and iteration cap (default 50).
 * @returns The consistent z0.
 * @throws WhiteboxError with code `convergence` when Newton fails.
 * @concept differential-algebraic-equation
 * @example
 * ```ts
 * import { consistentInitialConditions } from 'whitebox-engine/dynamics/ode'
 * import { add, sub, sum } from 'whitebox-engine/tensor'
 * const dae = { f: (_t, y) => y, g: (_t, y, z) => sub(add(sum(y), z), 1) }
 * Array.from(consistentInitialConditions(dae, 0, [0.25, 0.25], [0]).data) // => [0.5]
 * ```
 */
export function consistentInitialConditions(
  dae: SemiExplicitDae,
  t0: number,
  y0: TensorLike,
  z0: TensorLike,
  options: { readonly ftol?: number; readonly maxIterations?: number } = {},
): Tensor {
  const y = floatState(y0)
  const z = floatState(z0)
  const r = fsolve((v) => reshape(dae.g(t0, y, reshape(v, z.shape)), [-1]), reshape(z, [-1]), {
    ftol: options.ftol ?? 1e-12,
    maxIterations: options.maxIterations ?? 50,
  })
  if (!r.converged) throw new WhiteboxError('convergence', `consistentInitialConditions: ${r.message}`)
  return fromNumbers(numbersOf(r.x), z.shape)
}

/** Options of {@link solveDae}. */
export interface SolveDaeOptions extends IvpTraceOptions {
  /** Radau IIA (default) or BDF. */
  readonly method?: 'Radau' | 'BDF'
  readonly rtol?: number
  readonly atol?: number
  readonly tEval?: readonly number[]
  readonly events?: EventSpec | readonly EventSpec[]
  readonly jac?: JacobianOption
  readonly maxSteps?: number
  /** Make z0 consistent before integrating (default true). */
  readonly consistent?: boolean
}

/** The solution of a DAE: the ODE solution of the stacked state plus its two parts. */
export interface DaeSolution extends OdeSolution {
  /** Differential states `[nt, ny]`. */
  readonly yDiff: Value
  /** Algebraic states `[nt, nz]`. */
  readonly zAlg: Value
}

/**
 * Solve a semi-explicit index-1 DAE by Radau IIA or BDF with the singular mass matrix diag(I, 0): the algebraic
 * equations are enforced at every stage, and the initial algebraic state is made consistent first.
 *
 * @param dae - The DAE.
 * @param tspan - [t0, t1].
 * @param y0 - Initial differential state `[ny]`.
 * @param z0 - Initial (guess of the) algebraic state `[nz]`.
 * @param options - Method, tolerances, output times, events and tracing.
 * @returns The solution with the differential and algebraic parts.
 * @concept differential-algebraic-equation
 * @example
 * ```ts
 * import { solveDae } from 'whitebox-engine/dynamics/ode'
 * import { neg, sub, toFlat } from 'whitebox-engine/tensor'
 * // y' = -z, 0 = z - y: y = e^{-t}.
 * const s = solveDae({ f: (_t, _y, z) => neg(z), g: (_t, y, z) => sub(z, y) }, [0, 1], [1], [0], { rtol: 1e-8, atol: 1e-10, tEval: [1] })
 * Math.abs((toFlat(s.yDiff)[0] ?? 0) - Math.exp(-1)) < 1e-6 // => true
 * ```
 */
export function solveDae(
  dae: SemiExplicitDae,
  tspan: readonly [number, number],
  y0: TensorLike,
  z0: TensorLike,
  options: SolveDaeOptions = {},
): DaeSolution {
  const y = reshape(floatState(y0), [-1])
  const zGuess = reshape(floatState(z0), [-1])
  const z = options.consistent === false ? zGuess : consistentInitialConditions(dae, tspan[0], y, zGuess)
  const ny = y.shape[0] ?? 0
  const nz = z.shape[0] ?? 0
  const n = ny + nz
  const mass = new Float64Array(n * n)
  for (let i = 0; i < ny; i++) mass[i * n + i] = 1
  const field: VectorField = (t, u) => {
    const yy = sliceAxis(u, 0, 0, ny)
    const zz = sliceAxis(u, 0, ny, n)
    return concat([reshape(dae.f(t, yy, zz), [-1]), reshape(dae.g(t, yy, zz), [-1])], 0)
  }
  const implicitOptions = {
    rtol: options.rtol ?? 1e-6,
    atol: options.atol ?? 1e-9,
    mass: fromNumbers(mass, [n, n]),
    ...(options.jac === undefined ? {} : { jac: options.jac }),
  }
  const sol = solveIvp(field, tspan, concat([y, z], 0), {
    method: options.method === 'BDF' ? bdf(implicitOptions) : radau(implicitOptions),
    ...(options.tEval === undefined ? {} : { tEval: options.tEval }),
    ...(options.events === undefined ? {} : { events: options.events }),
    ...(options.maxSteps === undefined ? {} : { maxSteps: options.maxSteps }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  const split = (lo: number, hi: number): Value =>
    sol.ys.length === 0
      ? fromNumbers(new Float64Array(0), [0, hi - lo])
      : stack(sol.ys.map((u) => sliceAxis(u, 0, lo, hi)))
  return { ...sol, yDiff: split(0, ny), zAlg: split(ny, n) }
}
