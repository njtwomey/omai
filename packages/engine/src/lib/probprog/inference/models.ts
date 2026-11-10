/**
 * How a model reaches a method that is run by name, for instance in a worker. A method's knobs are only numbers, strings and
 * flags (`Method.config` is a `Space` and never carries a function or a spec), so the model cannot ride in a request. It is
 * registered by name where the method is: a `ModelRegistry` holds named models (a `ModelDefinition`, whose spec is JSON
 * and whose custom functions are code, with the data it is bound to), the worker's module builds one and lists the methods
 * it offers with `inferenceMethods`, each named `<kind>/<model>` (`nuts/radon`), and a page asks for it by that name with
 * only knob values. Data that varies from run to run is the registry's job too: `bind` rebinds a registered model to
 * JSON data (nested numbers), so a page that has the data sends it and the worker builds the method from the registered
 * spec and that data, trusting no spec from the wire. Nothing registers by import; an app builds its registry.
 */

import { OmaiError } from '../../base/index.ts'
import type { Config } from '../../trace/index.ts'
import type { BindOptions, BoundModel, DataInput } from '../model/bind.ts'
import type { ModelDefinition } from '../model/definition.ts'
import type { MethodModel } from './_support.ts'
import { abcRejectionMethod, abcSmcMethod } from './abc-methods.ts'
import { emMethod } from './em-method.ts'
import { chainMethod, conjugateMethod, enumerationMethod, gaussianMethod } from './exact-methods.ts'
import { gibbsMethod } from './gibbs-method.ts'
import { importanceMethod } from './importance-method.ts'
import { laplaceMethod } from './laplace-method.ts'
import type { InferenceMethod } from './method.ts'
import { nutsMethod } from './nuts-method.ts'
import { particleFilterMethod, temperedSmcMethod } from './smc-methods.ts'
import { sviMethod } from './svi-method.ts'

/** A model a registry holds: the definition (spec and custom functions) and the data it is bound to by default. */
export interface ModelEntry {
  readonly definition: ModelDefinition
  /** The data for the model's data nodes (nested numbers: JSON), possibly none for a model with no data nodes. */
  readonly data?: DataInput
  readonly options?: BindOptions
}

/** Named models, built by an app and handed to the module that makes its methods. */
export interface ModelRegistry {
  /** The names, in the order given. */
  names(): readonly string[]
  /** The registered model bound to its own data. @throws OmaiError with code `value` listing the names that exist. */
  need(name: string): BoundModel
  /**
   * The registered model bound to other data (JSON: nested arrays of numbers), over its own for the nodes not given.
   *
   * @throws OmaiError for an unknown name or data the spec rejects.
   */
  bind(name: string, data: DataInput): BoundModel
}

/**
 * A registry of named models.
 *
 * @param entries - The models by name.
 * @returns The registry.
 * @concept model-spec
 * @index function Models by name for a worker: `modelRegistry({ radon: { definition, data } })`.
 * @example
 * ```ts
 * import { modelRegistry } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), b.data('y', { dims: ['row'] })) })
 * const registry = modelRegistry({ m: { definition: m, data: { y: [1, 2] } } })
 * [registry.names(), registry.need('m').sizes, registry.bind('m', { y: [1, 2, 3] }).sizes] // => [['m'], { row: 2 }, { row: 3 }]
 * ```
 */
export function modelRegistry(entries: Readonly<Record<string, ModelEntry>>): ModelRegistry {
  const need = (name: string): ModelEntry => {
    const e = entries[name]
    if (e === undefined)
      throw new OmaiError('value', `no model called '${name}' (known: ${Object.keys(entries).join(', ') || 'none'})`)
    return e
  }
  return {
    names: () => Object.keys(entries),
    need: (name) => {
      const e = need(name)
      return e.definition.bind(e.data ?? {}, e.options)
    },
    bind: (name, data) => {
      const e = need(name)
      return e.definition.bind({ ...(e.data ?? {}), ...data }, e.options)
    },
  }
}

/** An inference method of any state, configuration and path type, as a list of the methods an app offers holds it. */
// biome-ignore lint/suspicious/noExplicitAny: a list of methods is generic over their states and paths
export type AnyInferenceMethod = InferenceMethod<any, any, any, any>

/** A function that makes a method for a model. */
export type MethodMaker = (model: MethodModel, options?: { readonly name?: string }) => AnyInferenceMethod

// The one-step methods take their name for their paths; a registry name only renames the method.
const renamed =
  (make: (model: MethodModel) => AnyInferenceMethod): MethodMaker =>
  (model, options) => {
    const method = make(model)
    return options?.name === undefined ? method : { ...method, name: options.name }
  }

/**
 * The inference methods by kind, as a table to pick from: a way to say which methods an app offers (`['nuts', 'svi']`), never
 * a way for the library to pick one for a model.
 */
export const methodKinds: Readonly<Record<string, MethodMaker>> = {
  nuts: nutsMethod,
  svi: sviMethod,
  gibbs: gibbsMethod,
  laplace: laplaceMethod,
  em: emMethod,
  importance: importanceMethod,
  'tempered-smc': temperedSmcMethod,
  'particle-filter': particleFilterMethod,
  'abc-smc': abcSmcMethod,
  conjugate: renamed(conjugateMethod),
  gaussian: renamed(gaussianMethod),
  enumeration: renamed(enumerationMethod),
  chain: renamed(chainMethod),
  'abc-rejection': renamed(abcRejectionMethod),
}

/**
 * The methods an app offers on its registered models, as a list for `methodRegistry` of the producer: one method per
 * (kind, model), named `<kind>/<model>`, with the model bound to the registered data. The worker's module calls this once
 * and a page asks for a method by name with only knob values.
 *
 * @param models - The registered models.
 * @param kinds - The kinds on offer (names of {@link methodKinds}; default every kind).
 * @returns The methods.
 * @throws OmaiError with code `value` for a kind that is not in the table.
 * @concept inference-engines
 * @index function The methods an app offers on its registered models, named kind/model, for a worker.
 * @example
 * ```ts
 * import { inferenceMethods, modelRegistry } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), b.data('y', { dims: ['row'] })) })
 * inferenceMethods(modelRegistry({ m: { definition: m, data: { y: [1, 2] } } }), ['nuts', 'svi']).map((x) => x.name) // => ['nuts/m', 'svi/m']
 * ```
 */
export function inferenceMethods(
  models: ModelRegistry,
  kinds: readonly string[] = Object.keys(methodKinds),
): AnyInferenceMethod[] {
  const out: AnyInferenceMethod[] = []
  for (const model of models.names()) for (const kind of kinds) out.push(inferenceMethodFor(models, kind, model))
  return out
}

/**
 * One method on one registered model, at its own data or at data given (JSON): what a worker builds when a page sends the
 * data of a run, with `<kind>/<model>` as its name.
 *
 * @param models - The registered models.
 * @param kind - A name of {@link methodKinds}.
 * @param model - A registered model.
 * @param data - Data for the model's data nodes, over the registered data.
 * @returns The method.
 * @throws OmaiError with code `value` for an unknown kind or model, or data the spec rejects.
 * @concept inference-engines
 * @example
 * ```ts
 * import { inferenceMethodFor, modelRegistry } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), b.data('y', { dims: ['row'] })) })
 * inferenceMethodFor(modelRegistry({ m: { definition: m, data: { y: [1, 2] } } }), 'laplace', 'm', { y: [3, 4, 5] }).name // => 'laplace/m'
 * ```
 */
export function inferenceMethodFor(
  models: ModelRegistry,
  kind: string,
  model: string,
  data?: DataInput,
): AnyInferenceMethod {
  const make = methodKinds[kind]
  if (make === undefined)
    throw new OmaiError(
      'value',
      `no inference method of kind '${kind}' (known: ${Object.keys(methodKinds).join(', ')})`,
    )
  const bound = data === undefined ? models.need(model) : models.bind(model, data)
  return make(bound, { name: `${kind}/${model}` })
}

/** The configuration a worker request carries, as the knobs of a method. */
export type MethodConfig = Config
