/**
 * The one builder of metric specifications: a file of metrics fixes the category, the kind of input and the concept
 * once, and each metric then names itself with its direction and range.
 */

import type { MetricDirection, MetricSpec } from '../base/index.ts'

/** The direction and range a file of metrics uses when a metric does not say. */
export interface SpecDefaults {
  /** Default direction (default `'higher'`). */
  readonly direction?: MetricDirection
  /** Default range (default unbounded). */
  readonly range?: readonly [number, number]
}

/**
 * A spec builder for a file of metrics: the category, inputs and concept are fixed, the name, direction and range come
 * per metric. The keys come out in the order name, direction, range, inputs, category, concept.
 *
 * @param category - The metric category.
 * @param inputs - What the metrics read.
 * @param concept - The concept page they belong to.
 * @param defaults - Direction and range when a metric gives none.
 * @returns The builder `(name, direction?, range?) => MetricSpec`.
 */
export function specOf(
  category: string,
  inputs: MetricSpec['inputs'],
  concept: string,
  defaults: SpecDefaults = {},
): (name: string, direction?: MetricDirection, range?: readonly [number, number]) => MetricSpec {
  const { direction: d = 'higher', range: r = [-Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY] } = defaults
  return (name, direction = d, range = r) => ({ name, direction, range, inputs, category, concept })
}
