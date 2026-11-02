/**
 * Knowledge-graph embeddings: entities and relations as vectors with a scoring function for triples `(head, relation,
 * tail)`. TransE (head plus relation near tail), DistMult (a trilinear product, symmetric in head and tail), ComplEx
 * (the same in complex space, whose conjugate breaks the symmetry) and RotatE (the relation a rotation of each complex
 * coordinate), each trained on a margin loss or a logistic negative-sampling loss against corrupted triples drawn once
 * from a key. The loss is a `SmoothProblem` over the flat entity and relation matrices, so the optimiser is a slot
 * (the stochastic `gradient` mode by default); evaluation ranks every entity as the head and the tail of each test
 * triple in the filtered setting (other true triples removed from the candidates) and reports MRR, mean rank and Hits@k.
 *
 * @index function Knowledge-graph embeddings: TransE, DistMult, ComplEx, RotatE.
 */

import { MdmlError } from '../base/index.ts'
import type { BatchObjective, SmoothProblem, VectorParams } from '../optim/index.ts'
import { categorical, foldIn, type Key, uniform } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  abs,
  add,
  cos,
  fromStorage,
  item,
  maximum,
  mul,
  neg,
  reshape,
  sin,
  sliceAxis,
  softplus,
  sqrt,
  square,
  sub,
  sum,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type ModeTable, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { solversWith } from './word2vec.ts'

/** The scoring function. */
export type KgModel = 'transe' | 'distmult' | 'complex' | 'rotate'

/** The loss against corrupted triples. */
export type KgLoss = 'margin' | 'negative-sampling'

/** Options of {@link kgProblem}. */
export interface KgOptions {
  /** The scoring function (default `distmult`). */
  readonly model?: KgModel
  /** The (complex) dimension (default 16): entities have this many coordinates, twice that in the complex models. */
  readonly dimension?: number
  /** `margin` (hinge on score differences, default) or `negative-sampling` (logistic against shifted scores). */
  readonly loss?: KgLoss
  /** The margin (default 1; for negative sampling, the shift gamma). */
  readonly margin?: number
  /** Corrupted triples per positive, half with the head replaced and half the tail (default 4). */
  readonly negatives?: number
  /** TransE's distance norm, 1 or 2 (default 1). */
  readonly norm?: 1 | 2
  /** An L2 penalty on the embeddings used by each batch (default 0). */
  readonly l2?: number
}

/** Triples as parallel index arrays. */
export interface Triples {
  readonly heads: ArrayLike<number>
  readonly relations: ArrayLike<number>
  readonly tails: ArrayLike<number>
}

/** A knowledge-graph embedding problem. */
export interface KgProblem extends SmoothProblem {
  readonly model: KgModel
  readonly entities: number
  readonly relations: number
  readonly dimension: number
  /** Coordinates per entity row and per relation row. */
  readonly entityWidth: number
  readonly relationWidth: number
  readonly options: Required<Pick<KgOptions, 'model' | 'dimension' | 'loss' | 'margin' | 'negatives' | 'norm' | 'l2'>>
}

const widths = (model: KgModel, d: number): { entity: number; relation: number } =>
  model === 'complex'
    ? { entity: 2 * d, relation: 2 * d }
    : model === 'rotate'
      ? { entity: 2 * d, relation: d }
      : { entity: d, relation: d }

/**
 * The score of one triple (larger is more plausible) from the embedding rows, in plain numbers.
 *
 * @param model - The scoring function.
 * @param h - The head's row.
 * @param r - The relation's row.
 * @param t - The tail's row.
 * @param norm - TransE's norm (default 1).
 * @returns The score.
 * @concept knowledge-graph-embedding
 * @example
 * ```ts
 * import { kgScore } from 'mdml-engine/embedding'
 * kgScore('transe', [1, 0], [0, 1], [1, 1]) // => 0
 * ```
 */
export function kgScore(
  model: KgModel,
  h: ArrayLike<number>,
  r: ArrayLike<number>,
  t: ArrayLike<number>,
  norm: 1 | 2 = 1,
): number {
  let s = 0
  if (model === 'transe') {
    for (let i = 0; i < h.length; i++) {
      const x = (h[i] ?? 0) + (r[i] ?? 0) - (t[i] ?? 0)
      s += norm === 1 ? Math.abs(x) : x * x
    }
    return -(norm === 1 ? s : Math.sqrt(s + 1e-12))
  }
  if (model === 'distmult') {
    for (let i = 0; i < h.length; i++) s += (h[i] ?? 0) * (r[i] ?? 0) * (t[i] ?? 0)
    return s
  }
  const d = h.length / 2
  for (let i = 0; i < d; i++) {
    const hr = h[i] ?? 0
    const hi = h[d + i] ?? 0
    const tr = t[i] ?? 0
    const ti = t[d + i] ?? 0
    if (model === 'complex') {
      const rr = r[i] ?? 0
      const ri = r[d + i] ?? 0
      s += hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr
    } else {
      const c = Math.cos(r[i] ?? 0)
      const n = Math.sin(r[i] ?? 0)
      s -= Math.hypot(hr * c - hi * n - tr, hr * n + hi * c - ti)
    }
  }
  return s
}

/** The same score on batches of rows `[n, width]`, in differentiable tensor operations. */
function scoreTensor(model: KgModel, H: Value, R: Value, T: Value, d: number, norm: 1 | 2): Value {
  if (model === 'transe') {
    const x = sub(add(H, R), T)
    return norm === 1 ? neg(sum(abs(x), -1)) : neg(sqrt(add(sum(square(x), -1), 1e-12)))
  }
  if (model === 'distmult') return sum(mul(mul(H, R), T), -1)
  const part = (x: Value, a: number): Value => sliceAxis(x, 1, a * d, (a + 1) * d)
  const [hr, hi, tr, ti] = [part(H, 0), part(H, 1), part(T, 0), part(T, 1)] as [Value, Value, Value, Value]
  if (model === 'complex') {
    const [rr, ri] = [part(R, 0), part(R, 1)] as [Value, Value]
    const terms = sub(add(add(mul(mul(hr, rr), tr), mul(mul(hi, rr), ti)), mul(mul(hr, ri), ti)), mul(mul(hi, ri), tr))
    return sum(terms, -1)
  }
  const c = cos(R)
  const n = sin(R)
  const re = sub(sub(mul(hr, c), mul(hi, n)), tr)
  const im = sub(add(mul(hr, n), mul(hi, c)), ti)
  return neg(sum(sqrt(add(add(square(re), square(im)), 1e-12)), -1))
}

/**
 * The knowledge-graph embedding loss as a {@link SmoothProblem} over the flat vector `[entity matrix, relation
 * matrix]`: the mean over training triples of the margin loss `max(0, margin - score(pos) + score(neg))` averaged over
 * the corrupted triples, or the logistic loss `softplus(-(margin + score(pos))) + mean softplus(margin + score(neg))`
 * written for scores that are larger when plausible (RotatE's negative-sampling loss with its distance as a negative
 * score), plus an optional L2 penalty on the rows a batch uses. Corrupted triples replace the head or the tail by a
 * uniformly drawn entity, drawn once, so the loss is a fixed function of the parameters. The start is uniform in
 * `[-1 / sqrt(d), 1 / sqrt(d)]`, with RotatE's phases uniform in `[-pi, pi]`.
 *
 * @param train - The training triples.
 * @param entities - The number of entities.
 * @param relations - The number of relations.
 * @param options - Scoring function, dimension, loss, margin, negatives, norm and L2.
 * @param key - The key for the corruptions and the start.
 * @returns The problem.
 * @throws MdmlError with code `value` for no triples, an index out of range or a bad option.
 * @concept knowledge-graph-embedding
 * @example
 * ```ts
 * import { kgProblem } from 'mdml-engine/embedding'
 * import { key } from 'mdml-engine/random'
 * const p = kgProblem({ heads: [0, 1], relations: [0, 0], tails: [1, 2] }, 3, 1, { model: 'complex', dimension: 2 }, key(0))
 * [p.entityWidth, p.relationWidth, (p.x0 as Float64Array).length] // => [4, 4, 16]
 * ```
 */
export function kgProblem(
  train: Triples,
  entities: number,
  relations: number,
  options: KgOptions,
  key: Key,
): KgProblem {
  const o = {
    model: options.model ?? 'distmult',
    dimension: options.dimension ?? 16,
    loss: options.loss ?? 'margin',
    margin: options.margin ?? 1,
    negatives: options.negatives ?? 4,
    norm: options.norm ?? 1,
    l2: options.l2 ?? 0,
  } as KgProblem['options']
  const n = train.heads.length
  if (n === 0) throw new MdmlError('value', 'kgProblem: needs at least one triple')
  if (!(o.dimension >= 1) || !(o.negatives >= 1))
    throw new MdmlError('value', 'kgProblem: dimension and negatives must be at least 1')
  for (let i = 0; i < n; i++) {
    const h = train.heads[i] as number
    const r = train.relations[i] as number
    const t = train.tails[i] as number
    if (!(h >= 0 && h < entities && t >= 0 && t < entities && r >= 0 && r < relations))
      throw new MdmlError('value', `kgProblem: triple ${i} is out of range`)
  }
  const { entity: we, relation: wr } = widths(o.model, o.dimension)
  const k = o.negatives
  // Corrupted triples: the first half of each positive's negatives replace the head, the rest the tail.
  const corrupt = Int32Array.from(categorical(foldIn(key, 0), new Array<number>(entities).fill(1), [n * k]).data)
  const H = Int32Array.from(train.heads as ArrayLike<number>)
  const Rl = Int32Array.from(train.relations as ArrayLike<number>)
  const Tl = Int32Array.from(train.tails as ArrayLike<number>)
  const x0 = new Float64Array(entities * we + relations * wr)
  const u = uniform(foldIn(key, 1), [x0.length]).data
  const scale = 1 / Math.sqrt(o.dimension)
  for (let i = 0; i < x0.length; i++) x0[i] = (2 * (u[i] ?? 0.5) - 1) * scale
  if (o.model === 'rotate') for (let i = entities * we; i < x0.length; i++) x0[i] = (2 * (u[i] ?? 0.5) - 1) * Math.PI
  const objective: BatchObjective = (params, batch) => {
    const x = params as Value
    const idx = batch ?? Array.from({ length: n }, (_, i) => i)
    const b = idx.length
    const E = reshape(sliceAxis(x, 0, 0, entities * we), [entities, we])
    const Rm = reshape(sliceAxis(x, 0, entities * we, entities * we + relations * wr), [relations, wr])
    const ints = (a: Int32Array | number[]): Value => fromStorage(Int32Array.from(a), [a.length], 'int32')
    const hp = ints(idx.map((e) => H[e] as number))
    const rp = ints(idx.map((e) => Rl[e] as number))
    const tp = ints(idx.map((e) => Tl[e] as number))
    const pos = scoreTensor(o.model, take(E, hp, 0), take(Rm, rp, 0), take(E, tp, 0), o.dimension, o.norm)
    // The negatives of each positive, flattened: [b * k].
    const hn: number[] = []
    const tn: number[] = []
    const rn: number[] = []
    for (const e of idx)
      for (let j = 0; j < k; j++) {
        const c = corrupt[e * k + j] as number
        const head = j < k / 2
        hn.push(head ? c : (H[e] as number))
        tn.push(head ? (Tl[e] as number) : c)
        rn.push(Rl[e] as number)
      }
    const negs = scoreTensor(
      o.model,
      take(E, ints(hn), 0),
      take(Rm, ints(rn), 0),
      take(E, ints(tn), 0),
      o.dimension,
      o.norm,
    )
    const negM = reshape(negs, [b, k])
    const posM = reshape(pos, [b, 1])
    let loss: Value
    if (o.loss === 'margin') loss = sum(maximum(add(sub(negM, posM), o.margin), 0))
    else loss = add(sum(softplus(neg(add(pos, o.margin)))), mul(sum(softplus(add(negM, o.margin))), 1 / k))
    let total = mul(loss, 1 / (b * (o.loss === 'margin' ? k : 1)))
    if (o.l2 > 0) {
      const used = add(add(sum(square(take(E, hp, 0))), sum(square(take(E, tp, 0)))), sum(square(take(Rm, rp, 0))))
      total = add(total, mul(used, o.l2 / b))
    }
    return total
  }
  return {
    objective,
    x0,
    batches: { n },
    model: o.model,
    entities,
    relations,
    dimension: o.dimension,
    entityWidth: we,
    relationWidth: wr,
    options: o,
  }
}

/**
 * The scores of triples under a problem's scoring function at parameters `x`, through the differentiable tensor
 * path the loss uses (it agrees with {@link kgScore} on the same rows).
 *
 * @param problem - The problem.
 * @param x - The flat parameters.
 * @param triples - The triples to score.
 * @returns One score per triple.
 * @concept knowledge-graph-embedding
 * @example
 * ```ts
 * import { kgProblem, kgScoreBatch } from 'mdml-engine/embedding'
 * import { key } from 'mdml-engine/random'
 * const p = kgProblem({ heads: [0], relations: [0], tails: [1] }, 2, 1, { model: 'transe', dimension: 2 }, key(0))
 * Array.from(kgScoreBatch(p, new Float64Array(8), [[0, 0, 1]])) // => [-0]
 * ```
 */
export function kgScoreBatch(
  problem: KgProblem,
  x: ArrayLike<number>,
  triples: readonly (readonly [number, number, number])[],
): Float64Array {
  const { entities, relations, entityWidth: we, relationWidth: wr } = problem
  const flat = fromStorage(Float64Array.from(x), [x.length], 'float64')
  const E = reshape(sliceAxis(flat, 0, 0, entities * we), [entities, we])
  const Rm = reshape(sliceAxis(flat, 0, entities * we, entities * we + relations * wr), [relations, wr])
  const col = (c: number): Value =>
    fromStorage(
      Int32Array.from(triples, (t) => t[c] as number),
      [triples.length],
      'int32',
    )
  const s = scoreTensor(
    problem.model,
    take(E, col(0), 0),
    take(Rm, col(1), 0),
    take(E, col(2), 0),
    problem.dimension,
    problem.options.norm,
  )
  return Float64Array.from(toFlat(s))
}

/** The solvers of the knowledge-graph loss: the library's smooth solvers, the stochastic `gradient` mode the default. */
export const kgSolvers: ModeTable<KgProblem, VectorParams> = solversWith<KgProblem>({}, 'gradient')

/** A trained knowledge-graph embedding. */
export interface KgEmbedding {
  readonly model: KgModel
  readonly norm: 1 | 2
  readonly entities: readonly Float64Array[]
  readonly relations: readonly Float64Array[]
  /** The objective over all training triples at the end. */
  readonly loss: number
  readonly steps: number
  /** The score of a triple. */
  score(h: number, r: number, t: number): number
}

/**
 * The embedding that a flat parameter vector of a problem stands for.
 *
 * @param problem - The problem.
 * @param x - The flat parameters.
 * @param extra - The loss and steps to record.
 * @returns The embedding with its scoring function.
 * @concept knowledge-graph-embedding
 * @example
 * ```ts
 * import { kgEmbedding, kgProblem } from 'mdml-engine/embedding'
 * import { key } from 'mdml-engine/random'
 * const p = kgProblem({ heads: [0], relations: [0], tails: [1] }, 2, 1, { dimension: 2 }, key(0))
 * kgEmbedding(p, p.x0 as Float64Array).entities.length // => 2
 * ```
 */
export function kgEmbedding(
  problem: KgProblem,
  x: ArrayLike<number>,
  extra: { readonly loss?: number; readonly steps?: number } = {},
): KgEmbedding {
  const rows = (off: number, count: number, w: number): Float64Array[] =>
    Array.from({ length: count }, (_, r) => Float64Array.from({ length: w }, (_, c) => x[off + r * w + c] ?? 0))
  const entities = rows(0, problem.entities, problem.entityWidth)
  const relations = rows(problem.entities * problem.entityWidth, problem.relations, problem.relationWidth)
  const { model } = problem
  const norm = problem.options.norm
  return {
    model,
    norm,
    entities,
    relations,
    loss: extra.loss ?? Number.NaN,
    steps: extra.steps ?? 0,
    score: (h, r, t) =>
      kgScore(model, entities[h] as Float64Array, relations[r] as Float64Array, entities[t] as Float64Array, norm),
  }
}

/** Hyperparameters of {@link trainKg}. */
export interface KgHyper extends KgOptions {
  /** The solver, a value of {@link kgSolvers}'s space (default the stochastic `gradient` mode). */
  readonly solver?: SpaceValue
  /** The step budget (default 1000). */
  readonly maxSteps?: number
}

/**
 * Train a knowledge-graph embedding: build the problem for the triples and run the solver of the slot on it.
 *
 * @param train - The training triples.
 * @param entities - The number of entities.
 * @param relations - The number of relations.
 * @param hyper - The model, loss and solver.
 * @param key - The key for the corruptions, the start and the batches.
 * @param run - Run options (`tracer`, `policy`).
 * @returns The embedding.
 * @concept knowledge-graph-embedding
 * @example
 * ```ts
 * import { trainKg } from 'mdml-engine/embedding'
 * import { key } from 'mdml-engine/random'
 * const m = trainKg({ heads: [0, 1], relations: [0, 0], tails: [1, 2] }, 3, 1, { dimension: 4, maxSteps: 5 }, key(0))
 * m.entities.length // => 3
 * ```
 */
export function trainKg(
  train: Triples,
  entities: number,
  relations: number,
  hyper: KgHyper,
  key: Key,
  run: Partial<Omit<RunOptions, 'maxSteps' | 'seed' | 'key'>> = {},
): KgEmbedding {
  const problem = kgProblem(train, entities, relations, hyper, foldIn(key, 0))
  const solver = hyper.solver ?? { kind: 'gradient', learningRate: 0.05, batch: 64, optimiser: { kind: 'adam' } }
  const result = runAlgorithm(kgSolvers.make(problem, solver, foldIn(key, 1)), {
    ...run,
    key: foldIn(key, 2),
    maxSteps: hyper.maxSteps ?? 1000,
  })
  const x = kgSolvers.params(solver, result.state).x
  const loss = item(problem.objective(fromStorage(Float64Array.from(x), [x.length], 'float64')) as Value)
  return kgEmbedding(problem, x, { loss, steps: result.steps })
}

/** The outcome of ranking the true entity among all candidates for a set of test triples. */
export interface KgRanking {
  /** The filtered rank of the true tail of each triple, then of the true head of each triple. */
  readonly ranks: readonly number[]
  /** Mean reciprocal rank. */
  readonly mrr: number
  /** Mean rank. */
  readonly meanRank: number
  /** The share of ranks at most k, for each k asked. */
  readonly hits: Readonly<Record<number, number>>
}

/**
 * Filtered link-prediction ranking (Bordes et al. 2013): for each test triple, score every entity as the tail (and
 * again as the head), remove the other entities that complete a known true triple, and take the rank of the true
 * entity (ties share the average rank). Reports the mean reciprocal rank, the mean rank and Hits@k over both
 * directions.
 *
 * @param embedding - The trained embedding.
 * @param test - The triples to rank.
 * @param known - Every true triple (train, validation, test), for filtering; default the test triples.
 * @param options - The ks of Hits@k (default 1, 3, 10), `heads: false` ranks tails only.
 * @returns The ranks and the summary metrics.
 * @concept knowledge-graph-embedding
 * @example
 * ```ts
 * import { rankKg } from 'mdml-engine/embedding'
 * const emb = { model: 'distmult' as const, norm: 1 as const, entities: [Float64Array.of(1), Float64Array.of(2), Float64Array.of(-1)], relations: [Float64Array.of(1)], loss: 0, steps: 0, score: (h: number, r: number, t: number) => [1, 2, -1][h]! * [1][r]! * [1, 2, -1][t]! }
 * rankKg(emb, [[0, 0, 1]], [[0, 0, 1]], { heads: false }).ranks // => [1]
 * ```
 */
export function rankKg(
  embedding: Pick<KgEmbedding, 'score' | 'entities'>,
  test: readonly (readonly [number, number, number])[],
  known: readonly (readonly [number, number, number])[] = test,
  options: { readonly ks?: readonly number[]; readonly heads?: boolean } = {},
): KgRanking {
  const n = embedding.entities.length
  const truth = new Set(known.map(([h, r, t]) => `${h},${r},${t}`))
  const ranks: number[] = []
  const rankOf = (scores: Float64Array, trueIndex: number, excluded: (e: number) => boolean): number => {
    const target = scores[trueIndex] as number
    let greater = 0
    let equal = 0
    for (let e = 0; e < n; e++) {
      if (e === trueIndex || excluded(e)) continue
      const s = scores[e] as number
      if (s > target) greater++
      else if (s === target) equal++
    }
    return 1 + greater + equal / 2
  }
  for (const [h, r, t] of test) {
    const tailScores = Float64Array.from({ length: n }, (_, e) => embedding.score(h, r, e))
    ranks.push(rankOf(tailScores, t, (e) => truth.has(`${h},${r},${e}`)))
  }
  if (options.heads !== false)
    for (const [h, r, t] of test) {
      const headScores = Float64Array.from({ length: n }, (_, e) => embedding.score(e, r, t))
      ranks.push(rankOf(headScores, h, (e) => truth.has(`${e},${r},${t}`)))
    }
  const ks = options.ks ?? [1, 3, 10]
  const m = ranks.length
  return {
    ranks,
    mrr: ranks.reduce((a, r) => a + 1 / r, 0) / m,
    meanRank: ranks.reduce((a, r) => a + r, 0) / m,
    hits: Object.fromEntries(ks.map((k) => [k, ranks.filter((r) => r <= k).length / m])),
  }
}
