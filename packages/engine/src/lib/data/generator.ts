/**
 * The dataset generator contract: a generator describes its knobs as a typed `Space` and maps a key and knob values to
 * data together with the truth of how it was made. `generate` runs one with defaults and a reproducible key,
 * `describeGenerator` flattens its knobs for documentation and controls, and `asGenerated` wraps a real dataset so
 * modifiers apply to it too.
 *
 * @index structure Dataset generator contract: knobs as a Space, key to data with truth.
 */

import { MdmlError } from '../base/index.ts'
import type { Context } from '../context/index.ts'
import { resolveKey } from '../context/index.ts'
import type { Key } from '../random/index.ts'
import { contains, type DictSpace, defaults, type Space, type SpaceValue } from '../space/index.ts'
import type { Dataset, DatasetKind } from './schema/common.ts'
import type { Truth } from './truth.ts'

/** Knob values of a generator: a value of its knob space. */
export type Knobs = Readonly<Record<string, SpaceValue>>

/** A dataset together with the truth of how it was generated. */
export interface Generated<D extends Dataset = Dataset> {
  readonly data: D
  readonly truth: Truth
}

/**
 * A dataset generator: knobs described by a typed `Space` (so they can be sampled, tuned or shown as controls), and a
 * function from a key and knobs to data with its truth.
 *
 * @concept datasets
 */
export interface DatasetGenerator<D extends Dataset = Dataset> {
  readonly name: string
  readonly kind: DatasetKind
  readonly description: string
  readonly knobs: DictSpace
  generate(key: Key, knobs: Knobs): Generated<D>
}

/** Options of {@link generate}. */
export interface GenerateOptions {
  /** Knob values; missing knobs take the space's defaults. */
  readonly knobs?: Knobs
  readonly key?: Key
  readonly ctx?: Context
}

/**
 * Generate a dataset: knobs default from the generator's space, and the key comes from `key`, the context, or the
 * default context (named after the generator), so the same seed always gives the same data. The data's metadata
 * records the generator, the knobs and the key.
 *
 * @param generator - The generator.
 * @param options - Knobs, key or context.
 * @returns The data and its truth.
 * @throws MdmlError with code `value` when a knob is unknown or outside its space.
 * @concept datasets
 * @index function Run a generator with a key and knob values.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { friedman1 } from 'mdml-engine/datasets'
 * generate(friedman1, { knobs: { n: 50 } }).data.x.shape // => [50, 10]
 * ```
 */
export function generate<D extends Dataset>(
  generator: DatasetGenerator<D>,
  options: GenerateOptions = {},
): Generated<D> {
  const given = options.knobs ?? {}
  for (const [name, value] of Object.entries(given)) {
    const space = generator.knobs.spaces[name]
    if (space === undefined) throw new MdmlError('value', `${generator.name}: unknown knob ${name}`)
    if (!contains(space, value))
      throw new MdmlError('value', `${generator.name}: knob ${name} = ${String(value)} is outside its space`)
  }
  const knobs = { ...(defaults(generator.knobs) as Knobs), ...given }
  const k = resolveKey(options, `dataset/${generator.name}`)
  const out = generator.generate(k, knobs)
  const meta = out.data.meta
  return {
    data: {
      ...out.data,
      meta: { ...meta, key: k.words, recipe: meta.recipe ?? { generator: generator.name, knobs } },
    },
    truth: out.truth,
  }
}

/** One knob of a generator, flattened for display. */
export interface KnobDescription {
  readonly name: string
  readonly kind: Space['kind']
  readonly label?: string
  readonly description?: string
  readonly default?: SpaceValue
  readonly low?: number
  readonly high?: number
  readonly options?: readonly (string | number | boolean)[]
}

/** What {@link describeGenerator} returns. */
export interface GeneratorDescription {
  readonly name: string
  readonly kind: DatasetKind
  readonly description: string
  readonly knobs: readonly KnobDescription[]
}

/**
 * A plain description of a generator and its knobs (name, type, range, default), for documentation and controls.
 *
 * @param generator - The generator.
 * @returns The description.
 * @example
 * ```ts
 * import { describeGenerator } from 'mdml-engine/data'
 * import { friedman1 } from 'mdml-engine/datasets'
 * describeGenerator(friedman1).knobs.map((k) => k.name) // => ['n', 'nFeatures', 'noise']
 * ```
 */
export function describeGenerator(generator: DatasetGenerator): GeneratorDescription {
  const defs = defaults(generator.knobs) as Knobs
  const knobs = Object.entries(generator.knobs.spaces).map(([name, s]): KnobDescription => {
    const range = s.kind === 'real' || s.kind === 'int' ? { low: s.low, high: s.high } : {}
    const value = defs[name]
    return {
      name,
      kind: s.kind,
      ...(s.label === undefined ? {} : { label: s.label }),
      ...(s.description === undefined ? {} : { description: s.description }),
      ...(value === undefined ? {} : { default: value }),
      ...range,
      ...(s.kind === 'choice' ? { options: s.options } : {}),
    }
  })
  return { name: generator.name, kind: generator.kind, description: generator.description, knobs }
}

/**
 * A generated pair for data that has no generator (a real dataset), so modifiers can be applied to it.
 *
 * @param data - The dataset.
 * @param truth - What is known (default nothing).
 * @returns `{ data, truth }`.
 * @example
 * ```ts
 * import { asGenerated, xy } from 'mdml-engine/data'
 * asGenerated(xy([[1]], [2])).truth // => {}
 * ```
 */
export function asGenerated<D extends Dataset>(data: D, truth: Truth = {}): Generated<D> {
  return { data, truth }
}
