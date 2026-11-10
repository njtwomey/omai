/**
 * A model as a value you hold: `ModelDefinition` pairs a `ModelSpec` (pure JSON) with the `custom` functions the spec
 * names (kept beside it because functions are not data), and carries the verbs of the model layer as methods: `bind`
 * data to it, `condition` latents on values, `do` interventions, and `toJSON`. `fromJSON` makes one back from a
 * serialised spec, supplying its custom functions by name; this is what lets a worker, a docs page and a saved example
 * hold the same model.
 *
 * @index structure ModelDefinition: a spec with its custom functions, and the methods bind, condition, do and toJSON.
 */

import { OmaiError } from '../../base/index.ts'
import type { Value } from '../../tensor/index.ts'
import { type BindOptions, type BoundModel, bind, type DataInput } from './bind.ts'
import { validateSpec } from './check.ts'
import { conditionSpec, doSpec, type SiteConstants } from './compose.ts'
import type { NamedValue } from './operators.ts'
import type { ModelSpec } from './spec.ts'

/**
 * An opaque function a model calls by name (the escape hatch of the registry): it receives its arguments with their
 * dimension names, and returns a value laid out as the `dims` and event axes the call declared. It works with every
 * sampling engine and with the compiled program, and is invisible to message passing and static analysis.
 */
export type CustomFn = (args: readonly NamedValue[]) => Value

/** The custom functions of a model, by the names its spec lists. */
export type CustomFns = Readonly<Record<string, CustomFn>>

/**
 * A model spec with its custom functions and the verbs of the model layer. Immutable: `condition` and `do` return new
 * definitions.
 *
 * @concept model-spec
 */
export class ModelDefinition {
  readonly spec: ModelSpec
  readonly customs: CustomFns

  constructor(spec: ModelSpec, customs: CustomFns = {}) {
    this.spec = spec
    this.customs = customs
  }

  get name(): string {
    return this.spec.name
  }

  /** Bind data to the model's data nodes (see `bind`). */
  bind(data: DataInput, options: BindOptions = {}): BoundModel {
    return bind(this, data, options)
  }

  /** The model with latent nodes conditioned on values: each becomes an observed node (see `conditionSpec`). */
  condition(values: SiteConstants): ModelDefinition {
    return new ModelDefinition(conditionSpec(this.spec, values), this.customs)
  }

  /** The model with latent nodes intervened on (Pearl's do): each becomes a constant (see `doSpec`). */
  do(values: SiteConstants): ModelDefinition {
    return new ModelDefinition(doSpec(this.spec, values), this.customs)
  }

  /** The spec, which is what `JSON.stringify` writes. */
  toJSON(): ModelSpec {
    return this.spec
  }
}

/**
 * A model definition from a serialised spec (the parsed JSON written by `toJSON`), checked end to end. A spec that
 * calls `custom` functions needs them supplied by name.
 *
 * @param json - The spec as parsed JSON (or an object of the right shape).
 * @param customs - The custom functions the spec names.
 * @returns The definition.
 * @throws OmaiError with code `value` for a spec that fails `validateSpec`, or a custom function it names that is not
 *   supplied.
 * @concept model-spec
 * @example
 * ```ts
 * import { defineModel, fromJSON } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.latent('x', normal(0, 1)) })
 * fromJSON(JSON.parse(JSON.stringify(m))).spec.nodes.length // => 1
 * ```
 */
export function fromJSON(json: unknown, customs: CustomFns = {}): ModelDefinition {
  const spec = json as ModelSpec
  validateSpec(spec)
  for (const name of spec.customs)
    if (customs[name] === undefined)
      throw new OmaiError('value', `the spec calls the custom function '${name}'; supply it`, { name })
  return new ModelDefinition(spec, customs)
}
