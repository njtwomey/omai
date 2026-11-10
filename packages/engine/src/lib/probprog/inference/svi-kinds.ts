/**
 * The variational methods beyond `sviMethod` by kind, as a table merged into `methodKinds`: a way to say which of them an
 * app offers on its registered models (`['svi-stl', 'pathfinder']`), never a way for the library to pick one. The amortised
 * method is not here: it needs the features of the data and the sites to encode, which are not knobs.
 */

import type { MethodModel } from './_support.ts'
import type { InferenceMethod } from './method.ts'
import { pathfinderMethod } from './pathfinder-method.ts'
import {
  renyiSviMethod,
  stickingTheLandingMethod,
  traceEnumSviMethod,
  traceGraphSviMethod,
} from './svi-extra-methods.ts'
import { mixtureSviMethod, structuredSviMethod } from './svi-guide-methods.ts'
import { conjugateNaturalSviMethod, naturalGradientSviMethod } from './svi-natural-method.ts'

// biome-ignore lint/suspicious/noExplicitAny: a table of methods is generic over their states and paths
type AnyMethod = InferenceMethod<any, any, any, any>

/** A function that makes a variational method for a model. */
export type VariationalKind = (model: MethodModel, options?: { readonly name?: string }) => AnyMethod

/**
 * The variational methods by kind: `svi-trace-graph`, `svi-trace-enum`, `svi-stl`, `svi-renyi`, `svi-structured`,
 * `svi-mixture`, `svi-natural`, `svi-conjugate-natural` and `pathfinder`.
 *
 * @example
 * ```ts
 * import { sviExtraKinds } from 'omai-engine/probprog/inference'
 * Object.keys(sviExtraKinds).includes('pathfinder') // => true
 * ```
 */
export const sviExtraKinds: Readonly<Record<string, VariationalKind>> = {
  'svi-trace-graph': traceGraphSviMethod,
  'svi-trace-enum': traceEnumSviMethod,
  'svi-stl': stickingTheLandingMethod,
  'svi-renyi': renyiSviMethod,
  'svi-structured': structuredSviMethod,
  'svi-mixture': mixtureSviMethod,
  'svi-natural': naturalGradientSviMethod,
  'svi-conjugate-natural': conjugateNaturalSviMethod,
  pathfinder: pathfinderMethod,
}
