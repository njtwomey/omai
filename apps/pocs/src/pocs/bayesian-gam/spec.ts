import type { AnyInferenceMethod } from 'omai-engine/probprog/inference'
import {
  type BayesianGamFitOptions,
  type BayesianGamOptions,
  bayesianGam,
  bayesianGamMethod,
} from 'omai-engine/probprog/models'

/**
 * What a run is: the rows, the model's options, the inference method and the fit's grid. A worker is asked for a method by
 * name, and a name can only carry text, so the problem travels as the name (`bgam:` and the JSON of the spec); both sides
 * build the method from it with {@link methodOf}, so the page and the worker hold the same method.
 */
export interface Spec {
  readonly kind: string
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly options: BayesianGamOptions
  readonly fit: BayesianGamFitOptions
}

const PREFIX = 'bgam:'
const made = new Map<string, AnyInferenceMethod>()

/** The name of a spec's method. */
export const nameOf = (spec: Spec): string => PREFIX + JSON.stringify(spec)

/** The method a name stands for (made once per name), or undefined for a name that is not a spec's. */
export function methodOf(name: string): AnyInferenceMethod | undefined {
  if (!name.startsWith(PREFIX)) return undefined
  let method = made.get(name)
  if (method === undefined) {
    const spec = JSON.parse(name.slice(PREFIX.length)) as Spec
    method = bayesianGamMethod(spec.kind, bayesianGam([...spec.x], [...spec.y], spec.options), { name, fit: spec.fit })
    made.set(name, method)
    if (made.size > 8) made.delete(made.keys().next().value as string)
  }
  return method
}
