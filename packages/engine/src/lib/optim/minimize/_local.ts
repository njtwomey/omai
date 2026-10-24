// The local search shared by the global methods (not exported from the barrel).
import { fromStorage, toFlat, type Value } from '../../tensor/index.ts'
import type { StepContext } from '../../trace/index.ts'
import type { LocalMethod } from './global.ts'
import { lbfgsb } from './lbfgsb.ts'
import type { Bounds, FlatObjective, VectorMethodOptions } from './objective.ts'

/**
 * The default local method of the global methods: L-BFGS-B, as scipy's basin hopping, dual annealing and differential
 * evolution polishing use (any {@link LocalMethod} can be passed instead, such as Nelder-Mead for non-smooth
 * objectives); its tolerances are tighter than scipy's defaults.
 */
export const defaultLocal: LocalMethod = (f, x0, options = {}) =>
  lbfgsb(f, x0, { gradTol: 1e-8, ftol: 1e-12, ...options })

/** Run a local method on the flat objective from a flat point, as a child run of the current step. */
export function localSearch(
  obj: FlatObjective,
  method: LocalMethod,
  x: Float64Array,
  options: VectorMethodOptions & { readonly bounds?: Bounds },
  ctx: StepContext | undefined,
): { readonly x: Float64Array; readonly fun: number; readonly nfev: number; readonly ok: boolean } {
  const r = method((v) => obj.ofValue(v as Value), fromStorage(Float64Array.from(x), [x.length], 'float64'), {
    ...options,
    ...(ctx === undefined ? {} : { parent: ctx.run, key: ctx.key }),
  })
  const xs = Float64Array.from(toFlat(r.x as Value))
  return { x: xs, fun: r.fun, nfev: r.nfev, ok: r.status !== 'failed' && r.status !== 'diverged' }
}
