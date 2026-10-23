import type { Context } from '../context/index.ts'
import { resolveKey } from '../context/index.ts'
import type { Key } from '../random/index.ts'
import { type DictSpace, defaults, type SpaceValue } from '../space/index.ts'
import type { DatasetKind } from './schema/common.ts'
import type { TabularData } from './schema/tabular.ts'
import type { Truth } from './truth.ts'

/** Knob values of a generator: a value of its knob space. */
export type Knobs = Readonly<Record<string, SpaceValue>>

/**
 * A dataset generator: knobs described by a typed `Space` (so they can be sampled, tuned or shown as controls), and a
 * function from a key and knobs to data with its truth.
 *
 * @concept datasets
 */
export interface DatasetGenerator<D extends TabularData = TabularData> {
  readonly name: string
  readonly kind: DatasetKind
  readonly description: string
  readonly knobs: DictSpace
  generate(key: Key, knobs: Knobs): { readonly data: D; readonly truth: Truth }
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
 * default context (named after the generator), so the same seed always gives the same data.
 *
 * @param generator - The generator.
 * @param options - Knobs, key or context.
 * @returns The data and its truth.
 * @concept datasets
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { friedman1 } from 'whitebox-engine/methods/datasets'
 * generate(friedman1, { knobs: { n: 50 } }).data.x.shape // => [50, 10]
 * ```
 */
export function generate<D extends TabularData>(
  generator: DatasetGenerator<D>,
  options: GenerateOptions = {},
): { readonly data: D; readonly truth: Truth } {
  const knobs = { ...(defaults(generator.knobs) as Knobs), ...(options.knobs ?? {}) }
  return generator.generate(resolveKey(options, `dataset/${generator.name}`), knobs)
}
