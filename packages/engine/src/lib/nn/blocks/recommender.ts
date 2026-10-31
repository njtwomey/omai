/**
 * Recommender blocks over categorical features and ids: the factorisation machine (pairwise interactions through
 * factor vectors, in the linear-time form), its field-aware (FFM), deep (DeepFM) and wide-and-deep relatives, neural
 * collaborative filtering (generalised matrix factorisation and an MLP over user and item embeddings, fused), the
 * two-tower retrieval model with in-batch logits and the log-Q correction, the deep cross network, and the DLRM-style
 * block that crosses dense and embedded sparse features by pairwise dot products.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  einsum,
  expandDims,
  fromStorage,
  matmul,
  mul,
  reciprocal,
  reshape,
  sliceAxis,
  sqrt,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { relu } from '../functional/activations.ts'
import { embedding } from '../functional/linear.ts'
import { normalInit } from '../init.ts'
import { Composite, mlp, Sequential } from '../module/containers.ts'
import { Embedding, Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, Module, type Params } from '../module/module.ts'

/** The input of the feature-interaction models: one feature index per field, with optional feature values. */
export interface FeatureInput {
  /** Feature indices `[B, F]` into a table of `numFeatures` rows (field `p` of every row at column `p`). */
  readonly ids: TensorLike
  /** Values `[B, F]` that scale each feature (default 1: a one-hot field). */
  readonly values?: TensorLike
}

const prepareFeatures = (x: unknown): FeatureInput => {
  const input = x as FeatureInput
  return input.ids === undefined ? { ids: x as TensorLike } : input
}

const valuesOf = (input: FeatureInput, ids: Value): Value =>
  input.values === undefined
    ? fromStorage(new Float64Array(ids.shape.reduce((a, b) => a * b, 1)).fill(1), ids.shape, 'float64')
    : asTensor(input.values)

/**
 * The factorisation-machine score from gathered rows (Rendle 2010): `w0 + sum_i w_i x_i + 1/2 sum_f [(sum_i v_if x_i)^2
 * - sum_i v_if^2 x_i^2]`, which equals the explicit sum over pairs `sum_{i<j} <v_i, v_j> x_i x_j` in linear time.
 *
 * @param bias - The global bias `w0` (scalar).
 * @param linear - The linear weights of the active features `[B, F]`.
 * @param factors - The factor vectors of the active features `[B, F, k]`.
 * @param values - The feature values `[B, F]`.
 * @returns The scores `[B]`.
 * @example
 * ```ts
 * import { fmScore } from 'mdml-engine/nn/blocks'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(fmScore(0, [[0, 0]], [[[1, 2], [3, 4]]], [[1, 1]])) // => [11]
 * ```
 */
export function fmScore(bias: TensorLike, linear: TensorLike, factors: TensorLike, values: TensorLike): Value {
  const x = asTensor(values)
  const v = asTensor(factors)
  const weighted = mul(v, expandDims(x, -1))
  const squareOfSum = sum(mul(sum(weighted, 1), sum(weighted, 1)), -1)
  const sumOfSquares = sum(mul(weighted, weighted), [1, 2])
  return add(add(bias, sum(mul(linear, x), -1)), mul(sub(squareOfSum, sumOfSquares), 0.5))
}

/** Options of the factorisation machines. */
export interface FmOptions {
  /** Standard deviation of the initial factors (default 0.01). */
  readonly std?: number
  readonly key?: Key
}

/**
 * A factorisation machine over `numFeatures` features: a global bias `bias`, linear weights `linear` `[N, 1]` and
 * factor vectors `factors` `[N, k]`. The input has one active feature per field (`ids` `[B, F]`, with optional
 * `values`); the output is the score (a logit for click prediction, a rating for regression) `[B]`. Matrix
 * factorisation is the special case of two fields, user and item.
 *
 * @reference Rendle (2010), Factorization machines.
 * @example
 * ```ts
 * import { FactorisationMachine } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new FactorisationMachine(20, 4, { key: key(0) }).forward({ ids: [[0, 10], [3, 12]] }).shape // => [2]
 * ```
 */
export class FactorisationMachine extends Module<FeatureInput, Value> {
  readonly kind = 'FactorisationMachine'
  readonly numFeatures: number
  readonly factorCount: number
  private readonly std: number

  constructor(numFeatures: number, factors: number, options: FmOptions = {}) {
    super()
    this.numFeatures = numFeatures
    this.factorCount = factors
    this.std = options.std ?? 0.01
    this.params = this.initParams(initKey('FactorisationMachine', options.key))
  }

  protected override prepare(x: unknown): unknown {
    return prepareFeatures(x)
  }

  override initParams(key: Key): Params {
    return {
      bias: zeros([]),
      linear: zeros([this.numFeatures, 1]),
      factors: normalInit(foldIn(key, 0), [this.numFeatures, this.factorCount], 0, this.std),
    }
  }

  apply(params: Params, input: FeatureInput): Value {
    const ids = asTensor(input.ids, 'int32')
    return fmScore(
      params['bias'] as Value,
      reshape(embedding(ids, params['linear'] as Value), ids.shape),
      embedding(ids, params['factors'] as Value),
      valuesOf(input, ids),
    )
  }

  /**
   * The pairwise interaction of every pair of fields in a row, `<v_i, v_j> x_i x_j` as `[B, F, F]` with zeros on and
   * below the diagonal, so that summing it gives the interaction part of the score.
   */
  interactions(params: Params, input: FeatureInput): Value {
    const ids = asTensor(input.ids, 'int32')
    const x = valuesOf(input, ids)
    const v = mul(embedding(ids, params['factors'] as Value), expandDims(x, -1))
    const gram = einsum('bik,bjk->bij', v, v)
    const f = ids.shape[1] ?? 0
    return mul(gram, upperPairs(f))
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }

  override extraRepr(): string {
    return `features=${this.numFeatures}, factors=${this.factorCount}`
  }
}

/** The strictly upper triangular 0/1 mask `[n, n]` (pairs `i < j`). */
function upperPairs(n: number): Tensor {
  const m = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m[i * n + j] = 1
  return fromStorage(m, [n, n], 'float64')
}

/**
 * A field-aware factorisation machine (Juan et al. 2016): every feature has one factor vector per field, and the
 * interaction of features `i` (in field `p`) and `j` (in field `q`) uses `v_{i,q}` and `v_{j,p}`:
 * `sum_{p<q} <v_{i_p,q}, v_{i_q,p}> x_p x_q`. Parameters `bias`, `linear` `[N, 1]` and `factors` `[N, F, k]`.
 *
 * @reference Juan, Zhuang, Chin and Lin (2016), Field-aware factorization machines for CTR prediction.
 * @example
 * ```ts
 * import { FieldAwareFm } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new FieldAwareFm(20, 2, 3, { key: key(0) }).forward({ ids: [[0, 10]] }).shape // => [1]
 * ```
 */
export class FieldAwareFm extends Module<FeatureInput, Value> {
  readonly kind = 'FieldAwareFm'
  readonly numFeatures: number
  readonly fields: number
  readonly factorCount: number
  private readonly std: number

  constructor(numFeatures: number, fields: number, factors: number, options: FmOptions = {}) {
    super()
    this.numFeatures = numFeatures
    this.fields = fields
    this.factorCount = factors
    this.std = options.std ?? 0.01
    this.params = this.initParams(initKey('FieldAwareFm', options.key))
  }

  protected override prepare(x: unknown): unknown {
    return prepareFeatures(x)
  }

  override initParams(key: Key): Params {
    return {
      bias: zeros([]),
      linear: zeros([this.numFeatures, 1]),
      factors: normalInit(foldIn(key, 0), [this.numFeatures, this.fields, this.factorCount], 0, this.std),
    }
  }

  apply(params: Params, input: FeatureInput): Value {
    const ids = asTensor(input.ids, 'int32')
    const [b, f] = ids.shape as [number, number]
    if (f !== this.fields) throw new MdmlError('shape', `FieldAwareFm: expected ${this.fields} fields, got ${f}`)
    const x = valuesOf(input, ids)
    const table = reshape(params['factors'] as Value, [this.numFeatures, f * this.factorCount])
    // v[b, p, q, :] is the vector of the feature in field p for interacting with field q.
    const v = reshape(embedding(ids, table), [b, f, f, this.factorCount])
    const pair = sum(mul(v, swapaxes(v, 1, 2)), -1)
    const scaled = mul(mul(pair, expandDims(x, 2)), expandDims(x, 1))
    const interaction = sum(mul(scaled, upperPairs(f)), [1, 2])
    const linear = sum(mul(reshape(embedding(ids, params['linear'] as Value), ids.shape), x), -1)
    return add(add(params['bias'] as Value, linear), interaction)
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }
}

/** Options of {@link DeepFm}. */
export interface DeepFmOptions extends FmOptions {
  /** The hidden widths of the deep part (default `[32, 32]`). */
  readonly hidden?: readonly number[]
}

/**
 * DeepFM (Guo et al. 2017): the factorisation machine and a multilayer perceptron over the concatenated field
 * embeddings share the same factor vectors; the output is the sum of the FM score and the deep logit.
 *
 * @reference Guo, Tang, Ye, Li and He (2017), DeepFM: a factorization-machine based neural network for CTR prediction.
 * @example
 * ```ts
 * import { DeepFm } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new DeepFm(20, 3, 4, { key: key(0) }).forward({ ids: [[0, 10, 15]] }).shape // => [1]
 * ```
 */
export class DeepFm extends Composite<FeatureInput, Value> {
  readonly kind = 'DeepFm'
  readonly numFeatures: number
  readonly fields: number
  readonly factorCount: number

  constructor(numFeatures: number, fields: number, factors: number, options: DeepFmOptions = {}) {
    const key = initKey('DeepFm', options.key)
    super(
      { deep: mlp([fields * factors, ...(options.hidden ?? [32, 32]), 1], { key: foldIn(key, 1) }) },
      {
        bias: zeros([]),
        linear: zeros([numFeatures, 1]),
        factors: normalInit(foldIn(key, 0), [numFeatures, factors], 0, options.std ?? 0.01),
      },
    )
    this.numFeatures = numFeatures
    this.fields = fields
    this.factorCount = factors
  }

  protected override prepare(x: unknown): unknown {
    return prepareFeatures(x)
  }

  apply(params: Params, input: FeatureInput, ctx: ApplyContext = {}): Value {
    const ids = asTensor(input.ids, 'int32')
    const x = valuesOf(input, ids)
    const factors = embedding(ids, params['factors'] as Value)
    const fm = fmScore(
      params['bias'] as Value,
      reshape(embedding(ids, params['linear'] as Value), ids.shape),
      factors,
      x,
    )
    const p = this.parts(params, ctx)
    const flat = reshape(mul(factors, expandDims(x, -1)), [ids.shape[0] ?? 0, this.fields * this.factorCount])
    const deep = reshape(p.run<Value, Value>('deep', flat), [-1])
    p.done()
    return add(fm, deep)
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }
}

/** Options of {@link WideAndDeep}. */
export interface WideAndDeepOptions {
  /** The embedding width of the deep part (default 8). */
  readonly embeddingDim?: number
  /** The hidden widths of the deep part (default `[32, 16]`). */
  readonly hidden?: readonly number[]
  readonly key?: Key
}

/**
 * Wide and deep (Cheng et al. 2016): a linear model over the (one-hot) features, the memorisation path, plus a
 * multilayer perceptron over learned field embeddings, the generalisation path; the output is the sum of their logits.
 *
 * @reference Cheng et al. (2016), Wide & deep learning for recommender systems.
 * @example
 * ```ts
 * import { WideAndDeep } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new WideAndDeep(20, 3, { key: key(0) }).forward({ ids: [[0, 10, 15]] }).shape // => [1]
 * ```
 */
export class WideAndDeep extends Composite<FeatureInput, Value> {
  readonly kind = 'WideAndDeep'
  readonly numFeatures: number
  readonly fields: number
  readonly embeddingDim: number

  constructor(numFeatures: number, fields: number, options: WideAndDeepOptions = {}) {
    const key = initKey('WideAndDeep', options.key)
    const dim = options.embeddingDim ?? 8
    super(
      {
        embedding: new Embedding(numFeatures, dim, { key: foldIn(key, 0) }),
        deep: mlp([fields * dim, ...(options.hidden ?? [32, 16]), 1], { key: foldIn(key, 1) }),
      },
      { bias: zeros([]), wide: zeros([numFeatures, 1]) },
    )
    this.numFeatures = numFeatures
    this.fields = fields
    this.embeddingDim = dim
  }

  protected override prepare(x: unknown): unknown {
    return prepareFeatures(x)
  }

  apply(params: Params, input: FeatureInput, ctx: ApplyContext = {}): Value {
    const ids = asTensor(input.ids, 'int32')
    const x = valuesOf(input, ids)
    const p = this.parts(params, ctx)
    const embedded = mul(p.run<Value, Value>('embedding', ids), expandDims(x, -1))
    const deep = reshape(
      p.run<Value, Value>('deep', reshape(embedded, [ids.shape[0] ?? 0, this.fields * this.embeddingDim])),
      [-1],
    )
    p.done()
    const wide = sum(mul(reshape(embedding(ids, params['wide'] as Value), ids.shape), x), -1)
    return add(add(params['bias'] as Value, wide), deep)
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }
}

/** The input of {@link NeuralCF}: a user and an item index for each interaction. */
export interface InteractionInput {
  readonly user: TensorLike
  readonly item: TensorLike
}

/** Which branches of {@link NeuralCF} are used. */
export type NcfMode = 'neumf' | 'gmf' | 'mlp'

/** Options of {@link NeuralCF}. */
export interface NeuralCfOptions {
  /** `gmf`: the generalised matrix factorisation branch only; `mlp`: the MLP branch only; `neumf`: both (default). */
  readonly mode?: NcfMode
  /** The embedding width of the GMF branch (default 8). */
  readonly gmfDim?: number
  /** The embedding width of each of the user and item embeddings of the MLP branch (default 8). */
  readonly mlpDim?: number
  /** The hidden widths of the MLP tower after the concatenated embeddings (default `[16, 8]`). */
  readonly hidden?: readonly number[]
  readonly key?: Key
}

/**
 * Neural collaborative filtering (He et al. 2017, NeuMF): a generalised matrix factorisation branch (the elementwise
 * product of user and item embeddings) and an MLP branch (the concatenated embeddings through a tower), their outputs
 * concatenated and mapped to one logit by a linear layer. With the `gmf` mode and a fixed all-ones output weight this
 * is matrix factorisation; `mlp` drops the product path.
 *
 * @reference He, Liao, Zhang, Nie, Hu and Chua (2017), Neural collaborative filtering.
 * @example
 * ```ts
 * import { NeuralCF } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new NeuralCF(10, 20, { key: key(0) }).forward({ user: [0, 1, 2], item: [5, 6, 7] }).shape // => [3]
 * ```
 */
export class NeuralCF extends Composite<InteractionInput, Value> {
  readonly kind = 'NeuralCF'
  readonly users: number
  readonly items: number
  readonly mode: NcfMode
  readonly gmfDim: number
  readonly mlpDim: number
  readonly tower: readonly number[]

  constructor(users: number, items: number, options: NeuralCfOptions = {}) {
    const key = initKey('NeuralCF', options.key)
    const mode = options.mode ?? 'neumf'
    const gmfDim = options.gmfDim ?? 8
    const mlpDim = options.mlpDim ?? 8
    const tower = options.hidden ?? [16, 8]
    const gmf = mode !== 'mlp'
    const deep = mode !== 'gmf'
    const fused = (gmf ? gmfDim : 0) + (deep ? (tower.at(-1) ?? 2 * mlpDim) : 0)
    super({
      ...(gmf
        ? {
            userGmf: new Embedding(users, gmfDim, { key: foldIn(key, 0) }),
            itemGmf: new Embedding(items, gmfDim, { key: foldIn(key, 1) }),
          }
        : {}),
      ...(deep
        ? {
            userMlp: new Embedding(users, mlpDim, { key: foldIn(key, 2) }),
            itemMlp: new Embedding(items, mlpDim, { key: foldIn(key, 3) }),
            tower: mlp([2 * mlpDim, ...tower], { key: foldIn(key, 4) }),
          }
        : {}),
      output: new Linear(fused, 1, { key: foldIn(key, 5) }),
    })
    this.users = users
    this.items = items
    this.mode = mode
    this.gmfDim = gmfDim
    this.mlpDim = mlpDim
    this.tower = tower
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: InteractionInput, ctx: ApplyContext = {}): Value {
    const user = asTensor(input.user, 'int32')
    const item = asTensor(input.item, 'int32')
    const p = this.parts(params, ctx)
    const branches: Value[] = []
    if (this.mode !== 'mlp') {
      branches.push(mul(p.run<Value, Value>('userGmf', user), p.run<Value, Value>('itemGmf', item)))
    }
    if (this.mode !== 'gmf') {
      const joined = concat([p.run<Value, Value>('userMlp', user), p.run<Value, Value>('itemMlp', item)], -1)
      // The tower's last layer is followed by a ReLU, as the original.
      branches.push(relu(p.run<Value, Value>('tower', joined)))
    }
    const out = reshape(p.run<Value, Value>('output', concat(branches, -1)), [-1])
    p.done()
    return out
  }

  /** The user embeddings used by the GMF branch (or the MLP branch when there is no GMF) `[users, dim]`. */
  userEmbeddings(params: Params): Value {
    return ((params[this.mode === 'mlp' ? 'userMlp' : 'userGmf'] as Params)['weight'] ?? zeros([0])) as Value
  }

  /** The item embeddings of the same branch as {@link NeuralCF.userEmbeddings}. */
  itemEmbeddings(params: Params): Value {
    return ((params[this.mode === 'mlp' ? 'itemMlp' : 'itemGmf'] as Params)['weight'] ?? zeros([0])) as Value
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }
}

/** The input of {@link TwoTower}: a user and an item (indices or feature rows, whatever the towers read). */
export interface TwoTowerInput {
  readonly user: TensorLike
  readonly item: TensorLike
}

/** The output of {@link TwoTower}. */
export interface TwoTowerOutput {
  /** The user tower's embeddings `[B, d]`. */
  readonly user: Value
  /** The item tower's embeddings `[B, d]`. */
  readonly item: Value
  /** The in-batch logits `[B, B]`: entry `(i, j)` scores user `i` against item `j`, divided by the temperature. */
  readonly logits: Value
}

/** Options of {@link TwoTower}. */
export interface TwoTowerOptions {
  /** Normalise both embeddings to unit length (cosine similarity, default true). */
  readonly normalise?: boolean
  /** The temperature dividing the logits (default 1). */
  readonly temperature?: number
}

/**
 * A two-tower retrieval model: a user tower and an item tower (any modules) map their inputs to embeddings that meet
 * in a dot product. The in-batch logits `[B, B]` score every user against every item of the batch, the positives on
 * the diagonal and the other items serving as negatives (the sampled softmax of Yi et al. 2019, see
 * {@link logQCorrection}). Item embeddings are what a nearest-neighbour index stores.
 *
 * @reference Yi et al. (2019), Sampling-bias-corrected neural modeling for large corpus item recommendations.
 * @example
 * ```ts
 * import { TwoTower, embeddingTower } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * const model = new TwoTower(embeddingTower(10, 4, [8], key(0)), embeddingTower(20, 4, [8], key(1)))
 * model.forward({ user: [0, 1, 2], item: [5, 6, 7] }).logits.shape // => [3, 3]
 * ```
 */
export class TwoTower extends Composite<TwoTowerInput, TwoTowerOutput> {
  readonly kind = 'TwoTower'
  readonly normalise: boolean
  readonly temperature: number

  constructor(userTower: Module, itemTower: Module, options: TwoTowerOptions = {}) {
    super({ userTower, itemTower })
    this.normalise = options.normalise ?? true
    this.temperature = options.temperature ?? 1
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  /** The item embeddings of a batch of items (what the index stores). */
  encodeItems(params: Params, item: TensorLike): Value {
    return this.unit(this.part('itemTower').apply((params['itemTower'] ?? {}) as Params, asTensor(item)))
  }

  /** The user embeddings of a batch of users. */
  encodeUsers(params: Params, user: TensorLike): Value {
    return this.unit(this.part('userTower').apply((params['userTower'] ?? {}) as Params, asTensor(user)))
  }

  private unit(e: Value): Value {
    return this.normalise ? normaliseRows(e) : e
  }

  apply(params: Params, input: TwoTowerInput): TwoTowerOutput {
    const user = this.encodeUsers(params, input.user)
    const item = this.encodeItems(params, input.item)
    return { user, item, logits: mul(matmul(user, swapaxes(item, 0, 1)), 1 / this.temperature) }
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, s[0] ?? 1]
  }
}

function normaliseRows(e: Value): Value {
  const norm = sqrt(sum(mul(e, e), -1, true))
  return mul(e, reciprocal(add(norm, 1e-12)))
}

/**
 * A tower of an id embedding followed by a perceptron: `Embedding(vocabulary, width)` then `mlp([width, ...hidden,
 * width])` (just the embedding when `hidden` is empty).
 *
 * @param vocabulary - The number of ids.
 * @param width - The embedding and output width.
 * @param hidden - The perceptron's hidden widths.
 * @param key - An initialisation key.
 * @returns The tower.
 * @example
 * ```ts
 * import { embeddingTower } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * embeddingTower(10, 4, [8], key(0)).forward([1, 2]).shape // => [2, 4]
 * ```
 */
export function embeddingTower(vocabulary: number, width: number, hidden: readonly number[], key?: Key): Module {
  const e = new Embedding(vocabulary, width, key === undefined ? {} : { key: foldIn(key, 0) })
  return hidden.length === 0
    ? e
    : new Sequential(e, ...mlp([width, ...hidden, width], key === undefined ? {} : { key: foldIn(key, 1) }).layers)
}

/**
 * The log-Q correction of a sampled softmax (Yi et al. 2019): subtracting the log of each item's probability of being
 * in a batch from its column of in-batch logits, so frequent items do not get an unearned advantage as negatives.
 *
 * @param logits - In-batch logits `[B, B]` (users by items).
 * @param logProbability - The log sampling probability of each item `[B]` (for example log of its popularity share).
 * @returns The corrected logits `[B, B]`.
 * @example
 * ```ts
 * import { logQCorrection } from 'mdml-engine/nn/blocks'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(logQCorrection([[1, 1], [1, 1]], [0, -1])) // => [[1, 2], [1, 2]]
 * ```
 */
export function logQCorrection(logits: TensorLike, logProbability: TensorLike): Value {
  return sub(logits, expandDims(asTensor(logProbability), 0))
}

/** Options of {@link CrossNetwork}. */
export interface CrossNetworkOptions {
  readonly key?: Key
}

/**
 * The cross network of DCN (Wang et al. 2017): `x_{l+1} = x_0 (w_l . x_l) + b_l + x_l`, an explicit feature crossing
 * of growing degree: `l` layers contain every product of up to `l + 1` input features, with `2 d` parameters per layer.
 *
 * @reference Wang, Fu, Fu and Wang (2017), Deep & cross network for ad click predictions.
 * @example
 * ```ts
 * import { CrossNetwork } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new CrossNetwork(4, 2, { key: key(0) }).forward([[1, 2, 3, 4]]).shape // => [1, 4]
 * ```
 */
export class CrossNetwork extends Module {
  readonly kind = 'CrossNetwork'
  readonly dim: number
  readonly layers: number

  constructor(dim: number, layers: number, options: CrossNetworkOptions = {}) {
    super()
    this.dim = dim
    this.layers = layers
    this.params = this.initParams(initKey('CrossNetwork', options.key))
  }

  override initParams(key: Key): Params {
    const out: Record<string, unknown> = {}
    for (let l = 0; l < this.layers; l++)
      out[String(l)] = {
        weight: normalInit(foldIn(key, l), [this.dim], 0, 1 / Math.sqrt(this.dim)),
        bias: zeros([this.dim]),
      }
    return out
  }

  apply(params: Params, x: Value): Value {
    let h = x
    for (let l = 0; l < this.layers; l++) {
      const p = params[String(l)] as Params
      const gate = sum(mul(h, p['weight'] as Value), -1, true)
      h = add(add(mul(x, gate), p['bias'] as Value), h)
    }
    return h
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** The input of {@link Dlrm}: dense features and one id per sparse feature. */
export interface DlrmInput {
  /** Dense features `[B, D]`. */
  readonly dense: TensorLike
  /** Sparse feature ids `[B, S]`, column `s` indexing table `s`. */
  readonly sparse: TensorLike
}

/** Options of {@link Dlrm}. */
export interface DlrmOptions {
  /** The bottom MLP's hidden widths (default `[16]`); its output width is the embedding width. */
  readonly bottomHidden?: readonly number[]
  /** The top MLP's hidden widths (default `[32, 16]`). */
  readonly topHidden?: readonly number[]
  readonly key?: Key
}

/**
 * The pairwise dot-product interaction of DLRM: given the `n` vectors `[B, n, d]` (the bottom MLP output and the sparse
 * embeddings), the inner products of all distinct pairs, the strictly lower triangle of the Gram matrix, `[B, n (n - 1) / 2]`.
 *
 * @param vectors - The vectors `[B, n, d]`.
 * @returns The pairwise products.
 * @example
 * ```ts
 * import { dotInteraction } from 'mdml-engine/nn/blocks'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(dotInteraction([[[1, 0], [1, 2], [0, 3]]])) // => [[1, 0, 6]]
 * ```
 */
export function dotInteraction(vectors: TensorLike): Value {
  const v = asTensor(vectors)
  const n = v.shape[1] ?? 0
  const gram = einsum('bid,bjd->bij', v, v)
  const rows: Value[] = []
  for (let i = 1; i < n; i++) for (let j = 0; j < i; j++) rows.push(sliceTwo(gram, i, j))
  return rows.length === 0 ? zeros([v.shape[0] ?? 0, 0]) : concat(rows, -1)
}

function sliceTwo(gram: Value, i: number, j: number): Value {
  return reshape(sliceAxis(sliceAxis(gram, 1, i, i + 1), 2, j, j + 1), [-1, 1])
}

/**
 * A DLRM-style click model (Naumov et al. 2019): the dense features pass through a bottom MLP to a vector of the
 * embedding width, each sparse feature is looked up in its own embedding table, all these vectors are crossed by
 * pairwise dot products ({@link dotInteraction}), and the products joined with the bottom output go through a top MLP
 * to a logit.
 *
 * @reference Naumov et al. (2019), Deep learning recommendation model for personalization and recommendation systems.
 * @example
 * ```ts
 * import { Dlrm } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new Dlrm(3, [10, 20], 4, { key: key(0) }).forward({ dense: [[0.1, 0.2, 0.3]], sparse: [[1, 5]] }).shape // => [1]
 * ```
 */
export class Dlrm extends Composite<DlrmInput, Value> {
  readonly kind = 'Dlrm'
  readonly denseDim: number
  readonly cardinalities: readonly number[]
  readonly width: number

  constructor(denseDim: number, cardinalities: readonly number[], width: number, options: DlrmOptions = {}) {
    const key = initKey('Dlrm', options.key)
    const vectors = cardinalities.length + 1
    const crossed = (vectors * (vectors - 1)) / 2
    const parts: Record<string, Module<never, unknown>> = {
      bottom: mlp([denseDim, ...(options.bottomHidden ?? [16]), width], { key: foldIn(key, 0) }) as never,
      top: mlp([width + crossed, ...(options.topHidden ?? [32, 16]), 1], { key: foldIn(key, 1) }) as never,
    }
    for (const [s, size] of cardinalities.entries())
      parts[`table${s}`] = new Embedding(size, width, { key: foldIn(key, 10 + s) }) as never
    super(parts as never)
    this.denseDim = denseDim
    this.cardinalities = cardinalities
    this.width = width
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: DlrmInput, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const dense = p.run<Value, Value>('bottom', asTensor(input.dense))
    const sparse = asTensor(input.sparse, 'int32')
    const columns = this.cardinalities.map((_, s) =>
      p.run<Value, Value>(`table${s}`, reshape(sliceAxis(sparse, 1, s, s + 1), [-1])),
    )
    const stacked = stack([dense, ...columns], 1)
    const out = reshape(p.run<Value, Value>('top', concat([dense, dotInteraction(stacked)], -1)), [-1])
    p.done()
    return out
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1]
  }
}
