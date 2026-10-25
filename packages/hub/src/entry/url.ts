/**
 * A hub demo's state in the URL: `stateToQuery` writes only what differs from the entry's demo state (dataset, seed,
 * dataset knobs and parameters) as query parameters, and `stateFromQuery` reads them back, ignoring unknown names and
 * out-of-range values, so a configured demo can be shared as a link.
 */

import { datasetCollections, datasetGenerators } from 'mdml-engine/datasets'
import { contains, type DictSpace, type SpaceValue } from 'mdml-engine/space'
import type { HubEntry } from './entry.ts'
import { datasetState, demoState, type HubState } from './run.ts'

const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b)
const text = (v: SpaceValue): string => (typeof v === 'string' ? v : JSON.stringify(v))
const parse = (s: string): unknown => {
  try {
    return JSON.parse(s)
  } catch {
    return s
  }
}

function knobSpace(collection: string, dataset: string): DictSpace | undefined {
  const e = datasetCollections[collection]?.entries.find((x) => x.name === dataset)
  return e !== undefined && 'generator' in e ? datasetGenerators[e.generator]?.knobs : undefined
}

/**
 * A state as URL query parameters, keeping only what differs from the entry's demo state: `data` (collection and
 * dataset), `seed`, `k.<knob>` for dataset knobs and the parameters by name.
 *
 * @param entry - The entry.
 * @param state - The state.
 * @returns The query, without `?` (empty for the demo state).
 * @example
 * ```ts
 * import { catalogue, demoState, stateToQuery } from 'mdml-hub'
 * const lasso = catalogue.byId.get('sklearn/linear-model/lasso')
 * lasso === undefined ? '' : stateToQuery(lasso, { ...demoState(lasso), seed: 3 }) // => 'seed=3'
 * ```
 */
export function stateToQuery(entry: HubEntry, state: HubState): string {
  const demo = demoState(entry)
  const q = new URLSearchParams()
  if (state.collection !== demo.collection || state.dataset !== demo.dataset)
    q.set('data', `${state.collection}/${state.dataset}`)
  if (state.seed !== 0) q.set('seed', String(state.seed))
  const base: Readonly<Record<string, SpaceValue>> =
    state.collection === ''
      ? {}
      : state.collection === demo.collection && state.dataset === demo.dataset
        ? demo.knobs
        : datasetState(state.collection, state.dataset).knobs
  for (const [k, v] of Object.entries(state.knobs)) if (!same(v, base[k])) q.set(`k.${k}`, text(v))
  for (const [k, v] of Object.entries(state.params)) if (!same(v, demo.params[k])) q.set(k, text(v))
  return q.toString()
}

/**
 * The state URL query parameters describe (see {@link stateToQuery}), from the entry's demo state: unknown datasets,
 * knobs and parameters, and values outside their spaces, are ignored.
 *
 * @param entry - The entry.
 * @param query - The query, with or without `?`.
 * @returns The state.
 * @example
 * ```ts
 * import { catalogue, stateFromQuery } from 'mdml-hub'
 * const lasso = catalogue.byId.get('sklearn/linear-model/lasso')
 * lasso === undefined ? 0 : stateFromQuery(lasso, '?seed=3&alpha=0.5').params['alpha'] // => 0.5
 * ```
 */
export function stateFromQuery(entry: HubEntry, query: string): HubState {
  const q = new URLSearchParams(query)
  const demo = demoState(entry)
  let data: Pick<HubState, 'collection' | 'dataset' | 'seed' | 'knobs'> = demo
  const named = q.get('data')
  if (named !== null) {
    const cut = named.lastIndexOf('/')
    const collection = named.slice(0, cut)
    const dataset = named.slice(cut + 1)
    if (entry.datasets.includes(collection) && (collection !== demo.collection || dataset !== demo.dataset))
      try {
        data = datasetState(collection, dataset)
      } catch {
        // An unknown dataset: keep the demo's.
      }
  }
  const seed = Number(q.get('seed') ?? 0)
  const knobs: Record<string, SpaceValue> = { ...data.knobs }
  const space = knobSpace(data.collection, data.dataset)
  const params: Record<string, SpaceValue> = { ...demo.params }
  for (const [k, raw] of q) {
    const v = parse(raw) as SpaceValue
    if (k.startsWith('k.')) {
      const s = space?.spaces[k.slice(2)]
      if (s !== undefined && contains(s, v)) knobs[k.slice(2)] = v
    } else {
      const s = entry.params.spaces[k]
      if (s !== undefined && contains(s, v)) params[k] = v
    }
  }
  return { ...data, seed: Number.isInteger(seed) && seed >= 0 ? seed : 0, knobs, params }
}
