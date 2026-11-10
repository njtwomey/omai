/**
 * Binding data to a model: `bind` checks the arrays given for a spec's data nodes against what the spec declares (every
 * required node present, the rank each node's dimensions imply, one size per named dimension across all nodes, the
 * support each node declares, integer codes inside the dimension they index, no stray NaN) and returns a `BoundModel`:
 * the spec, the sizes of its dimensions and the data as float64 tensors. `withData` rebinds some nodes (new covariates for
 * a prediction, `null` to unbind a response) without touching the latents; `bindDataset` binds the columns of a library
 * dataset; `boundToJSON` and `boundFromJSON` carry a bound model to a worker. A NaN in the data of an observed node is a
 * missing value, handled by that node's `missing` policy at compile time.
 *
 * @index function bind: check data against a spec's data nodes and dimensions and make a BoundModel.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { asTensor, cast, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { nodeExprs } from './analysis-exprs.ts'
import { validateSpec } from './check.ts'
import type { CustomFns } from './definition.ts'
import type { DataNode, ExprSpec, MissingPolicy, ModelSpec } from './spec.ts'

/** Data for a model's data nodes, by node name: numbers, nested arrays or tensors. `null` leaves a node unbound. */
export type DataInput = Readonly<Record<string, TensorLike | null>>

/** Subsampling: the dimension to take a random subset of per evaluation, and the subset's size. */
export interface BatchOption {
  readonly dim: string
  readonly size: number
}

/** Options of {@link bind}. */
export interface BindOptions {
  /** Sizes for dimensions no data pins down (a latent-only dimension without a fixed size). */
  readonly sizes?: Readonly<Record<string, number>>
  /** Replace the policy for missing (NaN) entries of observed nodes, by node name. */
  readonly missing?: Readonly<Record<string, MissingPolicy>>
  /** Subsample a dimension: log densities of the plates over it are rescaled by size over batch size. */
  readonly batch?: BatchOption
}

/** What `bind` records about an observed node's missing entries. */
export interface MissingInfo {
  readonly policy: MissingPolicy
  readonly count: number
}

/** What `bind` needs of a model: a spec and its custom functions (a `ModelDefinition` is one). */
export interface ModelLike {
  readonly spec: ModelSpec
  readonly customs: CustomFns
}

/**
 * A model with its data: the spec, the size of every named dimension, the data as float64 tensors (NaN for missing),
 * the data nodes left unbound, the batch option and the missing-value policy of each observed node.
 *
 * @concept model-spec
 */
export interface BoundModel extends ModelLike {
  readonly sizes: Readonly<Record<string, number>>
  readonly data: Readonly<Record<string, Tensor>>
  readonly unbound: readonly string[]
  readonly batch?: BatchOption
  /** By observed node name. */
  readonly missing: Readonly<Record<string, MissingInfo>>
  /** The options this model was bound with. */
  readonly options: BindOptions
  /**
   * The model with some data nodes rebound (new covariates for prediction) and the rest kept; `null` unbinds a node.
   * Sizes are inferred again from all bound data, so rebinding the rows of a covariate means unbinding or rebinding the
   * response of the same dimension.
   */
  withData(data: DataInput): BoundModel
}

const shapeText = (s: readonly number[]): string => `[${s.join(', ')}]`

function indexText(flat: number, shape: readonly number[]): string {
  const idx: number[] = []
  let rest = flat
  for (let i = shape.length - 1; i >= 0; i--) {
    const n = shape[i] as number
    idx.unshift(rest % n)
    rest = Math.floor(rest / n)
  }
  return idx.length === 0 ? '' : `[${idx.join(', ')}]`
}

function checkSupport(
  node: DataNode,
  values: readonly number[],
  shape: readonly number[],
  sizes: Readonly<Record<string, number>>,
): void {
  const bad = (i: number, why: string): never => {
    throw new OmaiError('value', `data '${node.name}'${indexText(i, shape)} is ${values[i]}: ${why}`, {
      name: node.name,
      index: i,
    })
  }
  const rule = node.support
  const levels = node.indexes === undefined ? undefined : sizes[node.indexes]
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number
    if (Number.isNaN(v)) continue
    if (!Number.isFinite(v)) bad(i, 'data must be finite')
    switch (rule) {
      case 'binary':
        if (v !== 0 && v !== 1) bad(i, 'a binary variable must be 0 or 1')
        break
      case 'count':
        if (!Number.isInteger(v) || v < 0) bad(i, 'a count must be a non-negative integer')
        break
      case 'positive':
        if (!(v > 0)) bad(i, 'must be positive')
        break
      case 'nonNegative':
        if (!(v >= 0)) bad(i, 'must be non-negative')
        break
      case 'probability':
        if (!(v >= 0 && v <= 1)) bad(i, 'a probability must lie in [0, 1]')
        break
      case 'index':
        if (!Number.isInteger(v) || v < 0) bad(i, 'an index must be a non-negative integer')
        break
      default:
        break
    }
    if (node.indexes !== undefined && levels !== undefined && (!Number.isInteger(v) || v < 0 || v >= levels))
      bad(i, `an index into '${node.indexes}' must be an integer in [0, ${levels})`)
  }
}

function constSizes(e: ExprSpec, out: (dim: string, size: number, who: string) => void): void {
  switch (e.kind) {
    case 'const': {
      let v: Json = e.value
      for (const d of e.dims ?? []) {
        if (!Array.isArray(v)) break
        out(d, v.length, 'a constant')
        v = v[0] ?? 0
      }
      break
    }
    case 'op':
    case 'custom':
      for (const a of e.args) constSizes(a, out)
      break
    default:
      break
  }
}

/** The data nodes an observed node's value reads directly (the ones that may hold NaN for missing). */
function observedData(spec: ModelSpec): Map<string, string[]> {
  const out = new Map<string, string[]>()
  for (const n of spec.nodes)
    if (n.kind === 'observed' && n.value.kind === 'ref')
      out.set(n.value.name, [...(out.get(n.value.name) ?? []), n.name])
  return out
}

function make(model: ModelLike, inputs: Readonly<Record<string, Tensor>>, options: BindOptions): BoundModel {
  const { spec } = model
  const dataNodes = new Map<string, DataNode>()
  for (const n of spec.nodes) if (n.kind === 'data') dataNodes.set(n.name, n)
  const sizes: Record<string, number> = {}
  const sizeFrom = new Map<string, string>()
  const setSize = (dim: string, size: number, who: string): void => {
    const known = sizes[dim]
    if (known !== undefined && known !== size)
      throw new OmaiError(
        'shape',
        `dimension '${dim}' has size ${known} in ${sizeFrom.get(dim)} but ${size} in ${who} (to predict for new rows, rebind or unbind every data node of '${dim}' with withData)`,
        { dim, sizes: [known, size] },
      )
    sizes[dim] = size
    if (!sizeFrom.has(dim)) sizeFrom.set(dim, who)
  }
  for (const d of spec.dims) if (d.size !== undefined) setSize(d.name, d.size, `range('${d.name}')`)
  for (const [dim, size] of Object.entries(options.sizes ?? {})) setSize(dim, size, 'the sizes option')
  for (const n of spec.nodes)
    for (const e of nodeExprs(n)) constSizes(e, (d, s, who) => setSize(d, s, `${who} in '${n.name}'`))

  const unbound: string[] = []
  const missingRequired: string[] = []
  for (const node of dataNodes.values()) {
    const t = inputs[node.name]
    if (t === undefined) {
      unbound.push(node.name)
      if (node.optional !== true) missingRequired.push(node.name)
      continue
    }
    const expected = node.dims.length + node.eventRank
    if (t.shape.length !== expected)
      throw new OmaiError(
        'shape',
        `data '${node.name}' has shape ${shapeText(t.shape)}, but the node has ${node.dims.length} named dimension(s) [${node.dims.join(', ')}] and ${node.eventRank} event axis(es)`,
        { name: node.name, shape: t.shape },
      )
    for (const [i, d] of node.dims.entries()) setSize(d, t.shape[i] as number, `data '${node.name}'`)
  }
  if (missingRequired.length > 0)
    throw new OmaiError(
      'value',
      `no data for ${missingRequired.map((m) => `'${m}'`).join(', ')} (mark a data node optional, or unbind it with null, to leave it out)`,
      {
        names: missingRequired,
      },
    )
  for (const d of spec.dims)
    if (d.labels !== undefined && sizes[d.name] !== undefined && d.labels.length !== sizes[d.name])
      throw new OmaiError('shape', `dimension '${d.name}' has ${d.labels.length} labels but size ${sizes[d.name]}`)
  const unsized = spec.dims.filter((d) => sizes[d.name] === undefined).map((d) => d.name)
  if (unsized.length > 0)
    throw new OmaiError(
      'shape',
      `the size of ${unsized.map((d) => `'${d}'`).join(', ')} is not known: give range() a size, bind data over it, or pass options.sizes`,
      { dims: unsized },
    )

  const mayBeMissing = observedData(spec)
  const missing: Record<string, MissingInfo> = {}
  for (const n of spec.nodes)
    if (n.kind === 'observed')
      missing[n.name] = { policy: options.missing?.[n.name] ?? n.missing ?? 'impute', count: 0 }
  const data: Record<string, Tensor> = {}
  for (const node of dataNodes.values()) {
    const t = inputs[node.name]
    if (t === undefined) continue
    const flat = toFlat(t)
    checkSupport(node, flat, t.shape, sizes)
    const nan = flat.reduce((c, v) => c + (Number.isNaN(v) ? 1 : 0), 0)
    if (nan > 0) {
      const users = mayBeMissing.get(node.name)
      if (users === undefined)
        throw new OmaiError(
          'value',
          `data '${node.name}' has ${nan} NaN entries, but only the data of observed nodes may be missing`,
          {
            name: node.name,
          },
        )
      for (const u of users) {
        const info = missing[u] as MissingInfo
        if (info.policy === 'error')
          throw new OmaiError(
            'value',
            `data '${node.name}' has ${nan} missing (NaN) entries and observed '${u}' has policy 'error'`,
            {
              name: node.name,
            },
          )
        missing[u] = { ...info, count: nan }
      }
    }
    data[node.name] = t
  }
  if (options.batch !== undefined) {
    const { dim, size } = options.batch
    const full = sizes[dim]
    if (full === undefined) throw new OmaiError('value', `batch: '${dim}' is not a dimension of the model`)
    if (!Number.isInteger(size) || size < 1 || size > full)
      throw new OmaiError('value', `batch: the size must be an integer in [1, ${full}], got ${size}`)
  }
  const bound: BoundModel = {
    spec,
    customs: model.customs,
    sizes,
    data,
    unbound,
    ...(options.batch === undefined ? {} : { batch: options.batch }),
    missing,
    options,
    withData: (more) => {
      const next: Record<string, Tensor> = { ...inputs }
      for (const [name, v] of Object.entries(more)) {
        if (!dataNodes.has(name)) throw unknownData(spec, name)
        if (v === null) delete next[name]
        else next[name] = toFloat(v)
      }
      return make(model, next, options)
    },
  }
  return bound
}

function toFloat(x: TensorLike): Tensor {
  return primalOf(cast(asTensor(x), 'float64')) as Tensor
}

function unknownData(spec: ModelSpec, name: string): OmaiError {
  const names = spec.nodes.filter((n) => n.kind === 'data').map((n) => n.name)
  return new OmaiError(
    'value',
    `'${name}' is not a data node of ${spec.name} (its data nodes are ${names.join(', ') || 'none'})`,
    {
      name,
    },
  )
}

/**
 * Bind data to a model: check it against the spec's data nodes and named dimensions and make a `BoundModel`.
 *
 * @param model - The model definition (or anything with a `spec` and `customs`).
 * @param data - Arrays by data node name (`null` or absent leaves an optional node unbound).
 * @param options - Sizes for dimensions without data, missing-value policies, and a batch dimension.
 * @returns The bound model.
 * @throws OmaiError with code `value` for an unknown or missing data node, a value outside a node's support, an index
 *   outside its dimension, a stray NaN; and `shape` for a rank that does not match the node's dimensions, a dimension
 *   with two sizes, or a dimension whose size nothing gives.
 * @concept model-spec
 * @index function Bind data to a ModelSpec with checks of dims, sizes and supports.
 * @example
 * ```ts
 * import { defineModel, bind } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => {
 *   const y = b.data('y', { dims: ['row'] })
 *   b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), y)
 * })
 * bind(m, { y: [1, 2, 3] }).sizes // => { row: 3 }
 * ```
 */
export function bind(model: ModelLike, data: DataInput, options: BindOptions = {}): BoundModel {
  validateSpec(model.spec)
  const inputs: Record<string, Tensor> = {}
  for (const [name, v] of Object.entries(data)) {
    if (!model.spec.nodes.some((n) => n.kind === 'data' && n.name === name)) throw unknownData(model.spec, name)
    if (v !== null) inputs[name] = toFloat(v)
  }
  return make(model, inputs, options)
}

/**
 * Bind the fields of a library dataset to a model's data nodes: `{ X: 'x', y: 'y' }` binds the dataset's `x` to the
 * node `X`. A field may be a path into the dataset (`columns.age`); a table column is read as its values.
 *
 * @param model - The model definition.
 * @param dataset - A library dataset (regression, classification, ...), or any object of tensors.
 * @param fields - For each data node name, the dataset field that holds its data.
 * @param options - Binding options.
 * @returns The bound model.
 * @throws OmaiError with code `value` for a field the dataset lacks, plus everything `bind` throws.
 * @concept model-spec
 * @example
 * ```ts
 * import { bindDataset, defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * import { tensor } from 'omai-engine/tensor'
 * const m = defineModel('m', (b) => { const y = b.data('y', { dims: ['row'] }); b.observe('y', normal(0, 1), y) })
 * bindDataset(m, { y: tensor([1, 2, 3]) }, { y: 'y' }).sizes // => { row: 3 }
 * ```
 */
export function bindDataset(
  model: ModelLike,
  dataset: object,
  fields: Readonly<Record<string, string>>,
  options: BindOptions = {},
): BoundModel {
  const input: Record<string, TensorLike> = {}
  for (const [node, path] of Object.entries(fields)) {
    let cur: unknown = dataset
    for (const part of path.split('.')) {
      cur = (cur as Record<string, unknown> | undefined)?.[part]
      if (cur === undefined)
        throw new OmaiError('value', `the dataset has no field '${path}' (for data node '${node}')`, { path })
    }
    const col = cur as { readonly values?: TensorLike }
    input[node] = (col.values ?? cur) as TensorLike
  }
  return bind(model, input, options)
}

/** A bound model as JSON: the spec, the data (NaN as null) and the options, for a worker or a saved example. */
export interface BoundModelJson {
  readonly schema: 'omai.bound-model/1'
  readonly spec: ModelSpec
  readonly data: {
    readonly [name: string]: { readonly shape: readonly number[]; readonly values: readonly (number | null)[] }
  }
  readonly options: {
    readonly sizes?: { readonly [dim: string]: number }
    readonly missing?: { readonly [node: string]: MissingPolicy }
    readonly batch?: BatchOption
  }
}

/**
 * A bound model as JSON, so it can cross to a worker (the custom functions it names are supplied by the receiver).
 *
 * @param bound - The bound model.
 * @returns The JSON value.
 * @concept model-spec
 * @example
 * ```ts
 * import { bind, boundToJSON, defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { const y = b.data('y', { dims: ['row'] }); b.observe('y', normal(0, 1), y) })
 * boundToJSON(bind(m, { y: [1, Number.NaN] })).data['y'] // => { shape: [2], values: [1, null] }
 * ```
 */
export function boundToJSON(bound: BoundModel): BoundModelJson {
  const data: Record<string, { shape: readonly number[]; values: (number | null)[] }> = {}
  for (const [name, t] of Object.entries(bound.data))
    data[name] = { shape: t.shape, values: toFlat(t).map((v) => (Number.isNaN(v) ? null : v)) }
  const { sizes, missing, batch } = bound.options
  return {
    schema: 'omai.bound-model/1',
    spec: bound.spec,
    data,
    options: {
      ...(sizes === undefined ? {} : { sizes }),
      ...(missing === undefined ? {} : { missing }),
      ...(batch === undefined ? {} : { batch }),
    },
  }
}

/**
 * A bound model from {@link boundToJSON}'s output.
 *
 * @param json - The parsed JSON.
 * @param customs - The custom functions the spec names.
 * @returns The bound model.
 * @throws OmaiError with code `value` for a wrong schema tag, an invalid spec or a missing custom function.
 * @concept model-spec
 * @example
 * ```ts
 * import { bind, boundFromJSON, boundToJSON, defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { const y = b.data('y', { dims: ['row'] }); b.observe('y', normal(0, 1), y) })
 * boundFromJSON(JSON.parse(JSON.stringify(boundToJSON(bind(m, { y: [1, 2] }))))).sizes // => { row: 2 }
 * ```
 */
export function boundFromJSON(json: BoundModelJson, customs: CustomFns = {}): BoundModel {
  if (json.schema !== 'omai.bound-model/1')
    throw new OmaiError('value', `unsupported bound model schema '${String(json.schema)}'`)
  validateSpec(json.spec)
  for (const name of json.spec.customs)
    if (customs[name] === undefined)
      throw new OmaiError('value', `the model calls the custom function '${name}'; supply it`, { name })
  const inputs: Record<string, Tensor> = {}
  for (const [name, d] of Object.entries(json.data))
    inputs[name] = fromStorage(
      Float64Array.from(d.values, (v) => (v === null ? Number.NaN : v)),
      d.shape,
      'float64',
    )
  return make({ spec: json.spec, customs }, inputs, json.options)
}
