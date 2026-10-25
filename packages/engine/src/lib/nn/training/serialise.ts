/**
 * Saving and loading a module's parameters and buffers: a JSON form with dtype and shape per tensor, text round trips,
 * and `loadTorchStateDict`, which loads a PyTorch state dict exported as tensor JSON (the modules use PyTorch's names
 * and layouts, so weights load as they are).
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { isTensor, type Tensor, type TensorJson, tensorFromJson, tensorToJson, treePaths } from '../../tensor/index.ts'
import type { Module, Params } from '../module/module.ts'

/** The serialised form of a module's state: format tag, module kind, and tensors by path. */
export interface SavedState {
  readonly format: 'mdml.state/1'
  readonly kind: string
  /** Parameters by path, as tensor JSON (dtype, shape, flat data; non-finite numbers as strings). */
  readonly params: Readonly<Record<string, TensorJson>>
  /** Buffers by path. */
  readonly buffers: Readonly<Record<string, TensorJson>>
}

const encodeTree = (tree: Params): Record<string, TensorJson> => {
  const out: Record<string, TensorJson> = {}
  for (const [path, v] of Object.entries(treePaths(tree))) {
    if (!isTensor(v)) throw new MdmlError('value', `save: ${path} is traced`)
    out[path] = tensorToJson(v)
  }
  return out
}

/**
 * Serialise a module's parameters and buffers (JSON with dtype and shape per tensor), for saving to disk or sending to
 * a worker.
 *
 * @param module - The module.
 * @returns The saved state.
 * @concept serialisation
 * @example
 * ```ts
 * import { saveState, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * Object.keys(saveState(new Linear(2, 1, { key: key(0) })).params) // => ['bias', 'weight']
 * ```
 */
export function saveState(module: Module<never, unknown>): SavedState {
  return {
    format: 'mdml.state/1',
    kind: module.kind,
    params: encodeTree(module.params),
    buffers: encodeTree(module.buffers),
  }
}

const isTensorJson = (x: unknown): x is TensorJson =>
  typeof x === 'object' && x !== null && 'dtype' in x && 'shape' in x && 'data' in x

/**
 * Load a saved state into a module of the same architecture (paths and shapes must match).
 *
 * @param module - The module to fill.
 * @param state - From {@link saveState} (or parsed from its JSON text).
 * @param options - `strict` (default true): every parameter and buffer must be present.
 * @returns The module.
 * @throws MdmlError with code `schema` for a different format and `value` or `shape` for mismatches.
 * @example
 * ```ts
 * import { loadState, saveState, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const a = new Linear(2, 1, { key: key(0) })
 * loadState(new Linear(2, 1, { key: key(1) }), saveState(a)).stateDict()['weight']?.data[0] === a.stateDict()['weight']?.data[0] // => true
 * ```
 */
export function loadState<M extends Module<never, unknown>>(
  module: M,
  state: SavedState | Json,
  options: { readonly strict?: boolean } = {},
): M {
  const s = state as Partial<SavedState>
  if (s.format !== 'mdml.state/1' || s.params === undefined)
    throw new MdmlError('schema', 'loadState: not a mdml.state/1 document')
  const dict: Record<string, Tensor> = {}
  for (const [k, v] of Object.entries({ ...s.params, ...(s.buffers ?? {}) })) dict[k] = tensorFromJson(v)
  return module.loadStateDict(dict, options)
}

/**
 * The JSON text of a module's state, and back: `stateToText(m)` and `stateFromText(m, text)`.
 *
 * @param module - The module.
 * @returns JSON text.
 * @example
 * ```ts
 * import { stateToText, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * JSON.parse(stateToText(new Linear(1, 1, { key: key(0) }))).format // => 'mdml.state/1'
 * ```
 */
export function stateToText(module: Module<never, unknown>): string {
  return JSON.stringify(saveState(module))
}

/**
 * Load a module's state from JSON text written by {@link stateToText}.
 *
 * @param module - The module to fill.
 * @param text - The JSON text.
 * @returns The module.
 * @example
 * ```ts
 * import { stateFromText, stateToText, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * stateFromText(new Linear(1, 1), stateToText(new Linear(1, 1, { key: key(0) }))).kind // => 'Linear'
 * ```
 */
export function stateFromText<M extends Module<never, unknown>>(module: M, text: string): M {
  return loadState(module, JSON.parse(text) as Json)
}

/** Options of {@link loadTorchStateDict}. */
export interface TorchLoadOptions {
  /** Rename keys before loading (for example to strip a `module.` prefix). */
  readonly rename?: (key: string) => string
  /** Keys to skip. */
  readonly ignore?: (key: string) => boolean
  readonly strict?: boolean
}

/**
 * Load a PyTorch state dict exported as tensor JSON (the oracle's fixture encoding, `{dtype, shape, data}` per key):
 * modules here use PyTorch's names and layouts, so weights load as they are, after optional renaming.
 *
 * @param module - The module to fill.
 * @param stateDict - Tensor JSON by PyTorch key.
 * @param options - Renaming, ignored keys and strictness.
 * @returns The module.
 * @example
 * ```ts
 * import { loadTorchStateDict, Linear } from 'mdml-engine/nn'
 * const w = { dtype: 'float64', shape: [1, 1], data: [2] } as const
 * const b = { dtype: 'float64', shape: [1], data: [0] } as const
 * loadTorchStateDict(new Linear(1, 1), { 'module.weight': w, 'module.bias': b }, { rename: (k) => k.replace('module.', '') }).forward([[3]]).shape // => [1, 1]
 * ```
 */
export function loadTorchStateDict<M extends Module<never, unknown>>(
  module: M,
  stateDict: Readonly<Record<string, unknown>>,
  options: TorchLoadOptions = {},
): M {
  const dict: Record<string, Tensor> = {}
  for (const [k, v] of Object.entries(stateDict)) {
    if (options.ignore?.(k) === true) continue
    if (!isTensorJson(v)) throw new MdmlError('schema', `loadTorchStateDict: ${k} is not tensor JSON`)
    dict[options.rename === undefined ? k : options.rename(k)] = tensorFromJson(v)
  }
  return module.loadStateDict(dict, { strict: options.strict ?? true })
}
