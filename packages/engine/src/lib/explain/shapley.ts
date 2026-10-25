/**
 * The Shapley family of attributions: exact Shapley and interaction values of any coalition game, the interventional
 * and conditional (Gaussian, empirical) value functions of a model, Kernel SHAP, permutation-sampling Shapley values,
 * Tree SHAP (path-dependent) with its interaction values for single trees and ensembles, linear SHAP (independent and
 * correlated), Owen values and partition SHAP over feature hierarchies, and SAGE's global importance.
 */
import { popcount, WhiteboxError } from '../base/index.ts'
import { type CoalitionalGame, coalitionalGame, shapleyInteractionIndex, shapleyValue } from '../games/index.ts'
import { keyFor, runOptionsFor } from '../learn/_shared.ts'
import { type FitOptions, type LearnData, requireTargets } from '../learn/estimator.ts'
import { type MatrixView, matrixView } from '../learn/validation.ts'
import { choleskyJitter, pinv, solve } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal, permutation, randint } from '../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'

/** A cooperative game: the value of each coalition (a membership mask over the players). */
export type CoalitionValue = (coalition: readonly boolean[]) => number

/** The most players the exact (exponential) computations accept. */
export const MAX_EXACT_PLAYERS = 20

function maskOf(bits: number, n: number): boolean[] {
  return Array.from({ length: n }, (_, i) => ((bits >> i) & 1) === 1)
}

/**
 * The game of a coalition value function, as games' `CoalitionalGame`: every coalition's value by bit mask, shifted by
 * v(empty) (games fix v(empty) = 0; Shapley and interaction values depend only on differences).
 */
function gameOf(value: CoalitionValue, n: number, who: string): CoalitionalGame {
  if (!(Number.isInteger(n) && n >= 1 && n <= MAX_EXACT_PLAYERS)) {
    throw new WhiteboxError('value', `${who}: exact computation needs 1 to ${MAX_EXACT_PLAYERS} players, got ${n}`)
  }
  const table = new Float64Array(1 << n)
  for (let b = 0; b < table.length; b++) table[b] = value(maskOf(b, n))
  const empty = table[0] ?? 0
  for (let b = 0; b < table.length; b++) table[b] = (table[b] ?? 0) - empty
  return coalitionalGame(n, table)
}

/**
 * Exact Shapley values of a cooperative game (Shapley 1953): each player's average marginal contribution over all
 * orders, sum over S of |S|! (n - |S| - 1)! / n! (v(S + i) - v(S)). Exponential in the players (at most
 * {@link MAX_EXACT_PLAYERS}). The values sum to v(N) - v(empty) (efficiency).
 *
 * @param value - The coalition value function.
 * @param nPlayers - The number of players.
 * @returns One value per player.
 * @concept shapley-values
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * // A glove game: player 0 owns a left glove, players 1 and 2 right gloves; a pair is worth 1.
 * explain.shapleyValues((s) => (s[0] && (s[1] || s[2]) ? 1 : 0), 3).map((v) => Math.round(v * 1e12) / 1e12) // => [0.666666666667, 0.166666666667, 0.166666666667]
 * ```
 */
export function shapleyValues(value: CoalitionValue, nPlayers: number): number[] {
  return shapleyValue(gameOf(value, nPlayers, 'shapleyValues'))
}

/**
 * Shapley interaction values (Grabisch and Roubens 1999; Lundberg et al. 2020's SHAP interaction values): for i != j,
 * half the Shapley interaction index sum over S of |S|! (n - |S| - 2)! / (2 (n - 1)!) times v(S + ij) - v(S + i) -
 * v(S + j) + v(S); the diagonal holds the main effects phi_i minus the row's interactions, so every row sums to phi_i
 * and the whole matrix to v(N) - v(empty).
 *
 * @param value - The coalition value function.
 * @param nPlayers - The number of players.
 * @returns The n x n matrix, row-major as nested arrays.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * // v(S) = 1 when both players are in: all the value is interaction.
 * explain.shapleyInteractionValues((s) => (s[0] && s[1] ? 1 : 0), 2) // => [[0, 0.5], [0.5, 0]]
 * ```
 */
export function shapleyInteractionValues(value: CoalitionValue, nPlayers: number): number[][] {
  const index = shapleyInteractionIndex(gameOf(value, nPlayers, 'shapleyInteractionValues'))
  return index.map((row, i) => {
    const out = row.map((v, j) => (j === i ? 0 : v / 2))
    let rest = 0
    for (const v of out) rest += v
    out[i] = (row[i] ?? 0) - rest
    return out
  })
}

/** A prediction function of a feature matrix: `[m, d]` to `[m]` (or the first column of `[m, k]`). */
export type PredictFunction = (x: Tensor) => Tensor

function evalRows(predict: PredictFunction, data: Float64Array, m: number, d: number): Float64Array {
  const out = predict(fromStorage(data, [m, d], 'float64'))
  const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const v = toFlat(out)
  return Float64Array.from({ length: m }, (_, i) => v[i * k] ?? 0)
}

/**
 * The interventional value function of a model at one point (the game KernelSHAP and the exact explainers solve):
 * v(S) = mean over background rows b of f(x_S, b_rest).
 *
 * @param predict - The model.
 * @param x - The explained row `[d]`.
 * @param background - Background rows `[m, d]`.
 * @returns The coalition value function.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const sum = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + x.data[2 * i + 1]))
 * explain.interventionalValue(sum, [1, 1], [[0, 0], [2, 0]])([true, false]) // => 1
 * ```
 */
export function interventionalValue(predict: PredictFunction, x: TensorLike, background: TensorLike): CoalitionValue {
  const xv = rowValues(x)
  const bg = matrixView(background, 'interventionalValue')
  if (xv.length !== bg.d)
    throw new WhiteboxError('shape', 'interventionalValue: x and the background have different features')
  return (s) => {
    const data = Float64Array.from(bg.data)
    for (let r = 0; r < bg.n; r++) for (let j = 0; j < bg.d; j++) if (s[j] === true) data[r * bg.d + j] = xv[j] ?? 0
    const p = evalRows(predict, data, bg.n, bg.d)
    return p.reduce((a, b) => a + b, 0) / bg.n
  }
}

function rowValues(x: TensorLike): number[] {
  return toFlat(asTensor(x))
}

/** Options of {@link kernelShap}. */
export interface KernelShapOptions {
  /** Coalitions evaluated (default 2d + 2048; every coalition when that is at least 2^d - 2, which is exact). */
  readonly nSamples?: number
  readonly key?: Key
}

/** Shapley values of a set of rows, and the base value E f(background). */
export interface ShapExplanation {
  /** values[row][feature]. */
  readonly values: readonly (readonly number[])[]
  readonly baseValue: number
}

/**
 * Kernel SHAP (Lundberg and Lee 2017; shap.KernelExplainer): Shapley values of the interventional game of each row as
 * the weighted least-squares fit of coalition values with the Shapley kernel (M - 1) / (C(M, s) s (M - s)), the
 * efficiency constraint imposed by eliminating one feature. With enough samples every coalition is enumerated and the
 * result is exact; otherwise coalition sizes are filled in complementary pairs from the smallest and largest, then
 * sampled. Features equal to every background row get 0 (shap's varying-feature rule).
 *
 * @param predict - The model.
 * @param x - Rows to explain `[n, d]`.
 * @param background - Background rows `[m, d]`.
 * @param options - nSamples and key.
 * @returns The values and the base value.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 2 * x.data[2 * i] - x.data[2 * i + 1]))
 * explain.kernelShap(linear, [[1, 1]], [[0, 0], [2, 2]]).values // => [[0, 0]]
 * ```
 */
export function kernelShap(
  predict: PredictFunction,
  x: TensorLike,
  background: TensorLike,
  options: KernelShapOptions = {},
): ShapExplanation {
  const xs = matrixView(x, 'kernelShap')
  const bg = matrixView(background, 'kernelShap', { nFeatures: xs.d })
  const baseValue = evalRows(predict, Float64Array.from(bg.data), bg.n, bg.d).reduce((a, b) => a + b, 0) / bg.n
  const key = options.key ?? keyFor({}, 'kernelShap')
  const values = Array.from({ length: xs.n }, (_, r) => {
    const row = xs.data.subarray(r * xs.d, (r + 1) * xs.d)
    const varying = Array.from(row, (v, j) => {
      for (let b = 0; b < bg.n; b++) if (bg.data[b * bg.d + j] !== v) return true
      return false
    })
    const idx = varying.flatMap((v, j) => (v ? [j] : []))
    const M = idx.length
    const out = new Array<number>(xs.d).fill(0)
    const game = interventionalValue(predict, Array.from(row), fromStorage(bg.data, [bg.n, bg.d], 'float64'))
    const fx = game(new Array<boolean>(xs.d).fill(true))
    if (M === 0) return out
    if (M === 1) {
      out[idx[0] ?? 0] = fx - baseValue
      return out
    }
    const masks = coalitionsFor(M, options.nSamples ?? 2 * xs.d + 2048, foldIn(key, r))
    const expand = (m: readonly boolean[]): boolean[] => {
      const s = new Array<boolean>(xs.d).fill(false)
      idx.forEach((j, t) => {
        s[j] = m[t] === true
      })
      return s
    }
    // Weighted least squares for phi_1..phi_{M-1}, with phi_M = (fx - base) - sum of the others.
    const p = M - 1
    const g = new Float64Array(p * p)
    const h = new Float64Array(p)
    const total = fx - baseValue
    for (const { mask, weight } of masks) {
      const y = game(expand(mask)) - baseValue - (mask[M - 1] === true ? total : 0)
      const z = Array.from({ length: p }, (_, t) => (mask[t] === true ? 1 : 0) - (mask[M - 1] === true ? 1 : 0))
      for (let a = 0; a < p; a++) {
        h[a] = (h[a] ?? 0) + weight * (z[a] ?? 0) * y
        for (let b = 0; b < p; b++) g[a * p + b] = (g[a * p + b] ?? 0) + weight * (z[a] ?? 0) * (z[b] ?? 0)
      }
    }
    const phi = toFlat(solve(fromStorage(g, [p, p], 'float64'), fromStorage(h, [p], 'float64')))
    phi.forEach((v, t) => {
      out[idx[t] ?? 0] = v
    })
    out[idx[M - 1] ?? 0] = total - phi.reduce((a, b) => a + b, 0)
    return out
  })
  return { values, baseValue }
}

function binomial(n: number, k: number): number {
  let r = 1
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i
  return r
}

/** The coalitions and kernel weights Kernel SHAP evaluates (all, when the budget allows). */
function coalitionsFor(M: number, budget: number, key: Key): { readonly mask: boolean[]; readonly weight: number }[] {
  const all = 2 ** M - 2
  const kernel = (s: number): number => (M - 1) / (binomial(M, s) * s * (M - s))
  if (M <= 30 && budget >= all) {
    const out: { mask: boolean[]; weight: number }[] = []
    for (let b = 1; b < (1 << M) - 1; b++) out.push({ mask: maskOf(b, M), weight: kernel(popcount(b)) })
    return out
  }
  // Complete sizes in complementary pairs (s and M - s) while the budget allows, then sample the rest.
  const out: { mask: boolean[]; weight: number }[] = []
  let left = budget
  const sizes = Array.from({ length: Math.floor((M - 1) / 2) }, (_, i) => i + 1)
  const done = new Set<number>()
  for (const s of sizes) {
    const count = binomial(M, s) * (s === M - s ? 1 : 2)
    if (count > left) break
    for (let b = 1; b < (1 << M) - 1; b++) {
      const c = popcount(b)
      if (c === s || c === M - s) out.push({ mask: maskOf(b, M), weight: kernel(c) })
    }
    done.add(s)
    done.add(M - s)
    left -= count
  }
  const rest = Array.from({ length: M - 1 }, (_, i) => i + 1).filter((s) => !done.has(s))
  if (rest.length > 0 && left > 0) {
    const mass = rest.reduce((a, s) => a + kernel(s) * binomial(M, s), 0)
    const draws = toFlat(randint(foldIn(key, 0), [left], 0, 1 << 30))
    for (let t = 0; t < left; t++) {
      let u = ((draws[t] ?? 0) / 2 ** 30) * mass
      let s = rest[rest.length - 1] ?? 1
      for (const c of rest) {
        u -= kernel(c) * binomial(M, c)
        if (u <= 0) {
          s = c
          break
        }
      }
      const perm = toFlat(permutation(foldIn(key, 1 + t), M))
      const mask = new Array<boolean>(M).fill(false)
      for (let i = 0; i < s; i++) mask[perm[i] ?? 0] = true
      out.push({ mask, weight: mass / left / binomial(M, s) })
    }
  }
  return out
}

/** Options of {@link samplingShapley}. */
export interface SamplingShapleyOptions extends FitOptions {
  /** Random feature orders per row (default 200). */
  readonly nPermutations?: number
}

/**
 * Permutation-sampling Shapley values (Strumbelj and Kononenko 2014; shap.SamplingExplainer): average each feature's
 * marginal contribution over random feature orders, each order paired with a random background row. Unbiased; a
 * traced algorithm with one step per permutation.
 *
 * @param predict - The model.
 * @param x - The row to explain `[d]`.
 * @param background - Background rows `[m, d]`.
 * @param options - nPermutations, key and tracing.
 * @returns The estimates and their standard errors.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 2 * x.data[2 * i] + x.data[2 * i + 1]))
 * explain.samplingShapley(linear, [1, 1], [[0, 0]], { nPermutations: 10 }).values // => [2, 1]
 * ```
 */
export function samplingShapley(
  predict: PredictFunction,
  x: TensorLike,
  background: TensorLike,
  options: SamplingShapleyOptions = {},
): { readonly values: readonly number[]; readonly standardErrors: readonly number[] } {
  const xv = rowValues(x)
  const bg = matrixView(background, 'samplingShapley', { nFeatures: xv.length })
  const d = xv.length
  const nPerm = options.nPermutations ?? 200
  if (!(Number.isInteger(nPerm) && nPerm >= 1))
    throw new WhiteboxError('value', 'samplingShapley: nPermutations must be >= 1')
  const key = keyFor(options, 'samplingShapley')
  type State = { readonly sum: Float64Array; readonly sumSq: Float64Array; readonly count: number }
  const algorithm: Algorithm<State> = {
    name: 'sampling-shapley',
    config: { features: d, nPermutations: nPerm },
    schema: channels({ metrics: { count: { description: 'permutations so far' } } }),
    init: () => ({ sum: new Float64Array(d), sumSq: new Float64Array(d), count: 0 }),
    step: (s) => {
      const k = foldIn(key, s.count)
      const order = toFlat(permutation(foldIn(k, 0), d))
      const b = toFlat(randint(foldIn(k, 1), [1], 0, bg.n))[0] ?? 0
      // Evaluate the d + 1 points along the order in one batch.
      const pts = new Float64Array((d + 1) * d)
      const cur = Float64Array.from(bg.data.subarray(b * d, (b + 1) * d))
      pts.set(cur, 0)
      order.forEach((j, t) => {
        cur[j] = xv[j] ?? 0
        pts.set(cur, (t + 1) * d)
      })
      const f = evalRows(predict, pts, d + 1, d)
      const sum = Float64Array.from(s.sum)
      const sumSq = Float64Array.from(s.sumSq)
      order.forEach((j, t) => {
        const c = (f[t + 1] ?? 0) - (f[t] ?? 0)
        sum[j] = (sum[j] ?? 0) + c
        sumSq[j] = (sumSq[j] ?? 0) + c * c
      })
      return { sum, sumSq, count: s.count + 1 }
    },
    done: (s) => (s.count >= nPerm ? 'completed' : undefined),
    observe: (s) => ({ metrics: { count: s.count } }),
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'samplingShapley'), maxSteps: nPerm })
  const values = Array.from(state.sum, (v) => v / state.count)
  const standardErrors = Array.from(state.sumSq, (v, j) => {
    const m = values[j] ?? 0
    return state.count < 2 ? Number.NaN : Math.sqrt(Math.max(0, v / state.count - m * m) / (state.count - 1))
  })
  return { values, standardErrors }
}

/**
 * A binary decision tree as arrays (sklearn's `tree_`): node i splits on `feature[i]` at `threshold[i]` (going left
 * when x <= threshold), leaves have `childrenLeft[i] = -1`; `value[i]` is the node's prediction and `cover[i]` its
 * (weighted) number of training samples.
 */
export interface TreeModel {
  readonly childrenLeft: readonly number[]
  readonly childrenRight: readonly number[]
  readonly feature: readonly number[]
  readonly threshold: readonly number[]
  readonly value: readonly number[]
  readonly cover: readonly number[]
}

/**
 * The prediction of a tree for each row.
 *
 * @param tree - The tree.
 * @param x - Rows `[n, d]`.
 * @returns Predictions `[n]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * toFlat(explain.treePredict(stump, [[0], [1]])) // => [0, 1]
 * ```
 */
export function treePredict(tree: TreeModel, x: TensorLike): Tensor {
  const m = matrixView(x, 'treePredict')
  return fromStorage(
    Float64Array.from({ length: m.n }, (_, i) => tree.value[leafOf(tree, m, i)] ?? 0),
    [m.n],
    'float64',
  )
}

function leafOf(tree: TreeModel, m: MatrixView, i: number): number {
  let node = 0
  for (let guard = 0; (tree.childrenLeft[node] ?? -1) >= 0 && guard < tree.value.length; guard++) {
    const f = tree.feature[node] ?? 0
    node =
      (m.data[i * m.d + f] ?? 0) <= (tree.threshold[node] ?? 0)
        ? (tree.childrenLeft[node] ?? 0)
        : (tree.childrenRight[node] ?? 0)
  }
  return node
}

interface PathElement {
  d: number
  z: number
  o: number
  w: number
}

/**
 * Tree SHAP (Lundberg, Erion and Lee 2018, algorithm 2; shap.TreeExplainer with tree_path_dependent perturbation):
 * exact Shapley values of the game v(S) = E[f(x) | x_S] where missing features follow the training data down the
 * tree (by cover), in time polynomial in the depth. The values sum to f(x) - E f.
 *
 * @param tree - The tree.
 * @param x - Rows to explain `[n, d]`.
 * @returns The values (`[n][d]`) and the base value (the cover-weighted mean leaf).
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * explain.treeShap(stump, [[1, 7]]) // => { values: [[0.5, 0]], baseValue: 0.5 }
 * ```
 */
export function treeShap(tree: TreeModel, x: TensorLike): ShapExplanation {
  const m = matrixView(x, 'treeShap')
  const baseValue = expectedLeaf(tree, 0)
  const values = Array.from({ length: m.n }, (_, i) => {
    const phi = new Array<number>(m.d).fill(0)
    const extend = (path: PathElement[], pz: number, po: number, pi: number): PathElement[] => {
      const l = path.length
      const out = path.map((e) => ({ ...e }))
      out.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 })
      for (let k = l - 1; k >= 0; k--) {
        const ek = out[k] as PathElement
        const ek1 = out[k + 1] as PathElement
        ek1.w += (po * ek.w * (k + 1)) / (l + 1)
        ek.w = (pz * ek.w * (l - k)) / (l + 1)
      }
      return out
    }
    const unwind = (path: PathElement[], idx: number): PathElement[] => {
      const l = path.length - 1
      const out = path.map((e) => ({ ...e }))
      const target = out[idx] as PathElement
      let n = (out[l] as PathElement).w
      for (let k = l - 1; k >= 0; k--) {
        const ek = out[k] as PathElement
        if (target.o !== 0) {
          const t = ek.w
          ek.w = (n * (l + 1)) / ((k + 1) * target.o)
          n = t - (ek.w * target.z * (l - k)) / (l + 1)
        } else {
          ek.w = (ek.w * (l + 1)) / (target.z * (l - k))
        }
      }
      for (let k = idx; k < l; k++) {
        const a = out[k] as PathElement
        const b = out[k + 1] as PathElement
        a.d = b.d
        a.z = b.z
        a.o = b.o
      }
      out.pop()
      return out
    }
    const unwoundSum = (path: PathElement[], idx: number): number => {
      const l = path.length - 1
      const target = path[idx] as PathElement
      let total = 0
      if (target.o !== 0) {
        let n = (path[l] as PathElement).w
        for (let k = l - 1; k >= 0; k--) {
          const tmp = (n * (l + 1)) / ((k + 1) * target.o)
          total += tmp
          n = (path[k] as PathElement).w - (tmp * target.z * (l - k)) / (l + 1)
        }
      } else {
        for (let k = l - 1; k >= 0; k--) total += ((path[k] as PathElement).w * (l + 1)) / (target.z * (l - k))
      }
      return total
    }
    const recurse = (node: number, path0: PathElement[], pz: number, po: number, pi: number, depth: number): void => {
      if (depth > tree.value.length) throw new WhiteboxError('value', 'treeShap: the tree has a cycle')
      const path = extend(path0, pz, po, pi)
      const left = tree.childrenLeft[node] ?? -1
      if (left < 0) {
        for (let k = 1; k < path.length; k++) {
          const e = path[k] as PathElement
          phi[e.d] = (phi[e.d] ?? 0) + unwoundSum(path, k) * (e.o - e.z) * (tree.value[node] ?? 0)
        }
        return
      }
      const right = tree.childrenRight[node] ?? -1
      const f = tree.feature[node] ?? 0
      const goLeft = (m.data[i * m.d + f] ?? 0) <= (tree.threshold[node] ?? 0)
      const hot = goLeft ? left : right
      const cold = goLeft ? right : left
      const cover = tree.cover[node] ?? 1
      let iz = 1
      let io = 1
      let p = path
      const k = p.findIndex((e, t) => t > 0 && e.d === f)
      if (k > 0) {
        iz = (p[k] as PathElement).z
        io = (p[k] as PathElement).o
        p = unwind(p, k)
      }
      recurse(hot, p, (iz * (tree.cover[hot] ?? 0)) / cover, io, f, depth + 1)
      recurse(cold, p, (iz * (tree.cover[cold] ?? 0)) / cover, 0, f, depth + 1)
    }
    recurse(0, [], 1, 1, -1, 0)
    return phi
  })
  return { values, baseValue }
}

function expectedLeaf(tree: TreeModel, node: number): number {
  const left = tree.childrenLeft[node] ?? -1
  if (left < 0) return tree.value[node] ?? 0
  const right = tree.childrenRight[node] ?? -1
  const c = tree.cover[node] ?? 1
  return ((tree.cover[left] ?? 0) * expectedLeaf(tree, left) + (tree.cover[right] ?? 0) * expectedLeaf(tree, right)) / c
}

/**
 * The path-dependent conditional expectation E[f(x) | x_S] of a tree (Lundberg et al. 2018, algorithm 1): follow x
 * on features in S, and both branches weighted by cover elsewhere. It is the game Tree SHAP solves.
 *
 * @param tree - The tree.
 * @param x - One row `[d]`.
 * @returns The coalition value function.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * explain.treeConditionalValue(stump, [1])([false]) // => 0.5
 * ```
 */
export function treeConditionalValue(tree: TreeModel, x: TensorLike): CoalitionValue {
  const xv = rowValues(x)
  return (s) => {
    const go = (node: number, depth: number): number => {
      if (depth > tree.value.length) throw new WhiteboxError('value', 'treeConditionalValue: the tree has a cycle')
      const left = tree.childrenLeft[node] ?? -1
      if (left < 0) return tree.value[node] ?? 0
      const right = tree.childrenRight[node] ?? -1
      const f = tree.feature[node] ?? 0
      if (s[f] === true) return go((xv[f] ?? 0) <= (tree.threshold[node] ?? 0) ? left : right, depth + 1)
      const c = tree.cover[node] ?? 1
      return ((tree.cover[left] ?? 0) * go(left, depth + 1) + (tree.cover[right] ?? 0) * go(right, depth + 1)) / c
    }
    return go(0, 0)
  }
}

/**
 * SHAP interaction values of a tree (shap.TreeExplainer.shap_interaction_values): the Shapley interaction values of
 * the path-dependent game, by exact enumeration over the features (at most {@link MAX_EXACT_PLAYERS}).
 *
 * @param tree - The tree.
 * @param x - Rows `[n, d]`.
 * @returns interactions[row][i][j].
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * explain.treeShapInteractions(stump, [[1, 7]]) // => [[[0.5, 0], [0, 0]]]
 * ```
 */
export function treeShapInteractions(tree: TreeModel, x: TensorLike): number[][][] {
  const m = matrixView(x, 'treeShapInteractions')
  return Array.from({ length: m.n }, (_, i) =>
    shapleyInteractionValues(treeConditionalValue(tree, Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))), m.d),
  )
}

/**
 * Tree SHAP for an ensemble that averages (random forests) or sums (boosting, with a scale such as the learning rate)
 * its trees: the values of the trees combined the same way.
 *
 * @param trees - The trees.
 * @param x - Rows `[n, d]`.
 * @param options - `'mean'` (default) or `'sum'` and a scale (default 1) applied to the sum.
 * @returns The values and base value of the ensemble.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * explain.ensembleTreeShap([stump, stump], [[1]], { combine: 'sum' }).values // => [[1]]
 * ```
 */
export function ensembleTreeShap(
  trees: readonly TreeModel[],
  x: TensorLike,
  options: { readonly combine?: 'mean' | 'sum'; readonly scale?: number } = {},
): ShapExplanation {
  if (trees.length === 0) throw new WhiteboxError('value', 'ensembleTreeShap: no trees')
  const parts = trees.map((t) => treeShap(t, x))
  const factor = (options.combine === 'sum' ? 1 : 1 / trees.length) * (options.scale ?? 1)
  const first = parts[0] as ShapExplanation
  return {
    values: first.values.map((row, i) =>
      row.map((_, j) => factor * parts.reduce((s, p) => s + (p.values[i]?.[j] ?? 0), 0)),
    ),
    baseValue: factor * parts.reduce((s, p) => s + p.baseValue, 0),
  }
}

/** A feature hierarchy (shap's partition tree): a feature index, or a group of sub-hierarchies. */
export type FeatureHierarchy = number | readonly FeatureHierarchy[]

function leavesOf(h: FeatureHierarchy): number[] {
  return typeof h === 'number' ? [h] : h.flatMap(leavesOf)
}

function factorial(k: number): number {
  let r = 1
  for (let i = 2; i <= k; i++) r *= i
  return r
}

/**
 * Owen values (Owen 1977; Winter 1989 for nested levels): the Shapley value of a game with a coalition structure.
 * Groups are first ordered among their siblings, then players within their group, so a player's value averages its
 * marginal contributions over orders that keep every group together; with one group of all players (or all players as
 * singletons) it is the Shapley value. shap's Partition explainer computes them over a feature clustering.
 *
 * @param value - The coalition value function.
 * @param hierarchy - The players' hierarchy (nested arrays of player indices covering 0..n-1 once each).
 * @returns One value per player.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * // v = 1 when player 0 joins player 1 or 2. Grouping 0 with 1 shifts value from 2 to them.
 * const v = (s) => (s[0] && (s[1] || s[2]) ? 1 : 0)
 * explain.owenValues(v, [[0, 1], 2]).map((x) => Math.round(x * 1e12) / 1e12) // => [0.75, 0.25, 0]
 * ```
 */
export function owenValues(value: CoalitionValue, hierarchy: FeatureHierarchy): number[] {
  const leaves = leavesOf(hierarchy)
  const n = leaves.length
  if (n > MAX_EXACT_PLAYERS) throw new WhiteboxError('value', `owenValues: at most ${MAX_EXACT_PLAYERS} players`)
  const seen = new Set(leaves)
  if (seen.size !== n || leaves.some((j) => !(Number.isInteger(j) && j >= 0 && j < n)))
    throw new WhiteboxError('value', 'owenValues: the hierarchy must hold each player 0..n-1 exactly once')
  const cache = new Map<number, number>()
  const v = (bits: number): number => {
    const known = cache.get(bits)
    if (known !== undefined) return known
    const r = value(maskOf(bits, n))
    cache.set(bits, r)
    return r
  }
  const bitsOf = (h: FeatureHierarchy): number => leavesOf(h).reduce((b, j) => b | (1 << j), 0)
  // The path from the root to each player: at each level the siblings and which of them holds the player.
  const paths = new Map<number, { readonly children: readonly FeatureHierarchy[]; readonly at: number }[]>()
  const walk = (h: FeatureHierarchy, path: { children: readonly FeatureHierarchy[]; at: number }[]): void => {
    if (typeof h === 'number') {
      paths.set(h, path)
      return
    }
    h.forEach((c, k) => {
      walk(c, [...path, { children: h, at: k }])
    })
  }
  walk(hierarchy, [])
  return Array.from({ length: n }, (_, i) => {
    const path = paths.get(i) ?? []
    let total = 0
    const go = (level: number, bits: number, weight: number): void => {
      if (level === path.length) {
        total += weight * (v(bits | (1 << i)) - v(bits))
        return
      }
      const { children, at } = path[level] as { children: readonly FeatureHierarchy[]; at: number }
      const others = children.filter((_, k) => k !== at).map(bitsOf)
      const c = children.length
      for (let r = 0; r < 1 << others.length; r++) {
        let b = bits
        let size = 0
        others.forEach((o, k) => {
          if ((r >> k) & 1) {
            b |= o
            size += 1
          }
        })
        go(level + 1, b, (weight * factorial(size) * factorial(c - size - 1)) / factorial(c))
      }
    }
    go(0, 0, 1)
    return total
  })
}

/**
 * A feature hierarchy from a scipy linkage matrix (rows `[a, b, distance, count]`, merged cluster ids from n up), as
 * shap's Partition masker reads its clustering.
 *
 * @param linkage - The `[n - 1, 4]` linkage rows.
 * @param n - The number of features.
 * @returns The binary hierarchy.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.hierarchyFromLinkage([[0, 1, 0.1, 2], [2, 3, 0.5, 3]], 3) // => [2, [0, 1]]
 * ```
 */
export function hierarchyFromLinkage(linkage: readonly (readonly number[])[], n: number): FeatureHierarchy {
  if (n === 1) return 0
  if (linkage.length !== n - 1) throw new WhiteboxError('shape', `hierarchyFromLinkage: ${n - 1} merges expected`)
  const node = (id: number): FeatureHierarchy => {
    if (id < n) return id
    const row = linkage[id - n]
    if (row === undefined) throw new WhiteboxError('value', `hierarchyFromLinkage: no cluster ${id}`)
    return [node(row[0] ?? 0), node(row[1] ?? 0)]
  }
  return node(2 * n - 2)
}

function meanAndCovariance(bg: MatrixView): { mu: number[]; sigma: number[][] } {
  const mu = Array.from({ length: bg.d }, (_, j) => {
    let s = 0
    for (let r = 0; r < bg.n; r++) s += bg.data[r * bg.d + j] ?? 0
    return s / bg.n
  })
  const sigma = Array.from({ length: bg.d }, (_, a) =>
    Array.from({ length: bg.d }, (_, b) => {
      let s = 0
      for (let r = 0; r < bg.n; r++)
        s += ((bg.data[r * bg.d + a] ?? 0) - (mu[a] ?? 0)) * ((bg.data[r * bg.d + b] ?? 0) - (mu[b] ?? 0))
      return s / Math.max(1, bg.n - 1)
    }),
  )
  return { mu, sigma }
}

/**
 * Average-linkage clustering of features by 1 - |correlation| over data, as a hierarchy for {@link owenValues} and
 * {@link partitionShap} (shap clusters features before its Partition explainer; correlated features end up together).
 *
 * @param x - Data `[n, d]`.
 * @returns The binary hierarchy.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * // Columns 0 and 2 move together; 1 is unrelated.
 * explain.correlationHierarchy([[1, 5, 1], [2, 1, 2.1], [3, 4, 2.9], [4, 2, 4.2]]) // => [1, [0, 2]]
 * ```
 */
export function correlationHierarchy(x: TensorLike): FeatureHierarchy {
  const m = matrixView(x, 'correlationHierarchy')
  const { sigma } = meanAndCovariance(m)
  const corr = (a: number, b: number): number => {
    const den = Math.sqrt((sigma[a]?.[a] ?? 0) * (sigma[b]?.[b] ?? 0))
    return den === 0 ? 0 : (sigma[a]?.[b] ?? 0) / den
  }
  let clusters: { members: number[]; tree: FeatureHierarchy }[] = Array.from({ length: m.d }, (_, j) => ({
    members: [j],
    tree: j,
  }))
  while (clusters.length > 1) {
    let best = [0, 1]
    let bestD = Number.POSITIVE_INFINITY
    for (let a = 0; a < clusters.length; a++)
      for (let b = a + 1; b < clusters.length; b++) {
        const ca = clusters[a]?.members ?? []
        const cb = clusters[b]?.members ?? []
        let s = 0
        for (const u of ca) for (const w of cb) s += 1 - Math.abs(corr(u, w))
        const dist = s / (ca.length * cb.length)
        if (dist < bestD - 1e-12) {
          bestD = dist
          best = [a, b]
        }
      }
    const [a = 0, b = 1] = best
    const A = clusters[a] as { members: number[]; tree: FeatureHierarchy }
    const B = clusters[b] as { members: number[]; tree: FeatureHierarchy }
    // The smaller cluster first, ties by the lowest feature, so the result does not depend on the scan order.
    const [first, second] =
      A.members.length < B.members.length ||
      (A.members.length === B.members.length && Math.min(...A.members) < Math.min(...B.members))
        ? [A, B]
        : [B, A]
    const merged = { members: [...first.members, ...second.members], tree: [first.tree, second.tree] }
    clusters = clusters.filter((_, k) => k !== a && k !== b)
    clusters.push(merged)
  }
  return (clusters[0] as { tree: FeatureHierarchy }).tree
}

/**
 * Partition SHAP (shap.PartitionExplainer and ExactExplainer with a Partition masker): the Owen values of each row's
 * interventional game over a feature hierarchy, so correlated features (clustered together) are explained as groups
 * first, which keeps the explanation from crediting impossible combinations; the values still sum to f(x) - E f.
 *
 * @param predict - The model.
 * @param x - Rows to explain `[n, d]`.
 * @param background - Background rows `[m, d]`.
 * @param hierarchy - The feature hierarchy (default {@link correlationHierarchy} of the background).
 * @returns The values and the base value.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 2 * x.data[2 * i] - x.data[2 * i + 1]))
 * // An additive model: Owen values equal Shapley values w (x - E x).
 * explain.partitionShap(linear, [[1, 1]], [[0, 0], [2, 4]], [0, 1]).values // => [[0, 1]]
 * ```
 */
export function partitionShap(
  predict: PredictFunction,
  x: TensorLike,
  background: TensorLike,
  hierarchy?: FeatureHierarchy,
): ShapExplanation {
  const xs = matrixView(x, 'partitionShap')
  const bg = matrixView(background, 'partitionShap', { nFeatures: xs.d })
  const h = hierarchy ?? correlationHierarchy(fromStorage(bg.data, [bg.n, bg.d], 'float64'))
  if (leavesOf(h).length !== xs.d)
    throw new WhiteboxError('shape', 'partitionShap: the hierarchy must cover every feature')
  const baseValue = evalRows(predict, Float64Array.from(bg.data), bg.n, bg.d).reduce((a, b) => a + b, 0) / bg.n
  const bgT = fromStorage(bg.data, [bg.n, bg.d], 'float64')
  const values = Array.from({ length: xs.n }, (_, r) =>
    owenValues(interventionalValue(predict, Array.from(xs.data.subarray(r * xs.d, (r + 1) * xs.d)), bgT), h),
  )
  return { values, baseValue }
}

/** Options of {@link linearShap}. */
export interface LinearShapOptions {
  /**
   * `interventional` (default; shap's Independent masker): features missing from a coalition take background values
   * independently, phi = w (x - E x). `correlation`: they follow the Gaussian conditional given the coalition's values
   * (exact Shapley values over the 2^d coalitions).
   */
  readonly perturbation?: 'interventional' | 'correlation'
}

/**
 * Linear SHAP (shap.LinearExplainer): exact Shapley values of a linear model w x + b. With interventional
 * (independent) perturbation they are w_j (x_j - E x_j); with correlation-dependent perturbation the value of a
 * coalition is the model at the Gaussian conditional mean of the missing features given the present ones (the
 * background's mean and covariance), which spreads credit onto correlated features.
 *
 * @param coef - The weights `[d]`.
 * @param intercept - The intercept.
 * @param x - Rows to explain `[n, d]`.
 * @param background - Background rows `[m, d]` (their mean, and covariance for `correlation`).
 * @param options - perturbation.
 * @returns The values and the base value w E x + b.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.linearShap([2, -1], 0.5, [[3, 1]], [[1, 1], [3, 3]]).values // => [[2, 1]]
 * ```
 */
export function linearShap(
  coef: readonly number[],
  intercept: number,
  x: TensorLike,
  background: TensorLike,
  options: LinearShapOptions = {},
): ShapExplanation {
  const xs = matrixView(x, 'linearShap', { nFeatures: coef.length })
  const bg = matrixView(background, 'linearShap', { nFeatures: coef.length })
  const { mu, sigma } = meanAndCovariance(bg)
  const d = xs.d
  const baseValue = intercept + coef.reduce((s, w, j) => s + w * (mu[j] ?? 0), 0)
  if ((options.perturbation ?? 'interventional') === 'interventional') {
    return {
      values: Array.from({ length: xs.n }, (_, r) =>
        coef.map((w, j) => w * ((xs.data[r * d + j] ?? 0) - (mu[j] ?? 0))),
      ),
      baseValue,
    }
  }
  const values = Array.from({ length: xs.n }, (_, r) => {
    const row = Array.from(xs.data.subarray(r * d, (r + 1) * d))
    const v: CoalitionValue = (s) => {
      const m = gaussianConditionalMean(row, s, mu, sigma)
      return intercept + coef.reduce((acc, w, j) => acc + w * (m[j] ?? 0), 0)
    }
    return shapleyValues(v, d)
  })
  return { values, baseValue }
}

/** E[x | x_S = row_S] under N(mu, sigma): row on S, the regression of the rest on S elsewhere. */
function gaussianConditionalMean(
  row: readonly number[],
  s: readonly boolean[],
  mu: readonly number[],
  sigma: readonly (readonly number[])[],
): number[] {
  const d = row.length
  const inS = Array.from({ length: d }, (_, j) => j).filter((j) => s[j] === true)
  const out = mu.map((v, j) => (s[j] === true ? (row[j] ?? 0) : v))
  if (inS.length === 0 || inS.length === d) return out
  const ss = inS.map((a) => inS.map((b) => sigma[a]?.[b] ?? 0))
  const inv = toArray(pinv(ss)) as number[][]
  const dev = inS.map((a) => (row[a] ?? 0) - (mu[a] ?? 0))
  const z = inv.map((r) => r.reduce((acc, v, k) => acc + v * (dev[k] ?? 0), 0))
  for (let j = 0; j < d; j++) {
    if (s[j] === true) continue
    out[j] = (mu[j] ?? 0) + inS.reduce((acc, a, k) => acc + (sigma[j]?.[a] ?? 0) * (z[k] ?? 0), 0)
  }
  return out
}

/** Options of {@link conditionalValue}. */
export interface ConditionalValueOptions {
  /**
   * `gaussian` (Aas, Jullum and Loland 2021's Gaussian approach: sample the missing features from their conditional
   * normal given the present ones) or `empirical` (weight background rows by their Mahalanobis closeness on the present
   * features). Default `gaussian`.
   */
  readonly method?: 'gaussian' | 'empirical'
  /** Conditional samples per coalition (gaussian, default 100). */
  readonly nSamples?: number
  /** The kernel bandwidth of the empirical weights (default 0.4). */
  readonly sigma?: number
  readonly key?: Key
}

/**
 * The observational (conditional) value function of a model at a point: v(S) = E[f(x) | x_S] with the missing
 * features drawn from their distribution given the present ones, against {@link interventionalValue}'s independent
 * draw. Conditional Shapley values (Aas, Jullum and Loland 2021) stay on the data manifold and credit correlated
 * features jointly; interventional ones explain the model's mechanism (Janzing et al. 2020).
 *
 * @param predict - The model.
 * @param x - The explained row `[d]`.
 * @param background - Rows `[m, d]` of the data distribution.
 * @param options - method, nSamples, sigma and key.
 * @returns The coalition value function.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * // Two perfectly correlated features: knowing the first fixes the second, so f = x2 has value 1 at x1 = 1.
 * const second = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i + 1]))
 * const v = explain.conditionalValue(second, [1, 1], [[-1, -1], [0, 0], [1, 1]], { method: 'empirical', sigma: 0.1 })
 * Math.round(v([true, false]) * 1e6) / 1e6 // => 1
 * ```
 */
export function conditionalValue(
  predict: PredictFunction,
  x: TensorLike,
  background: TensorLike,
  options: ConditionalValueOptions = {},
): CoalitionValue {
  const xv = rowValues(x)
  const bg = matrixView(background, 'conditionalValue', { nFeatures: xv.length })
  const d = bg.d
  const { mu, sigma } = meanAndCovariance(bg)
  const method = options.method ?? 'gaussian'
  const key = options.key ?? keyFor({}, 'conditionalValue')
  if (method === 'empirical') {
    const h = options.sigma ?? 0.4
    return (s) => {
      const inS = Array.from({ length: d }, (_, j) => j).filter((j) => s[j] === true)
      const data = Float64Array.from(bg.data)
      for (let r = 0; r < bg.n; r++) for (const j of inS) data[r * d + j] = xv[j] ?? 0
      const f = evalRows(predict, data, bg.n, d)
      if (inS.length === 0) return f.reduce((a, b) => a + b, 0) / bg.n
      const inv = toArray(pinv(inS.map((a) => inS.map((b) => sigma[a]?.[b] ?? 0)))) as number[][]
      let sw = 0
      let sf = 0
      for (let r = 0; r < bg.n; r++) {
        const dev = inS.map((j) => (xv[j] ?? 0) - (bg.data[r * d + j] ?? 0))
        let q = 0
        for (let a = 0; a < dev.length; a++)
          for (let b = 0; b < dev.length; b++) q += (dev[a] ?? 0) * (inv[a]?.[b] ?? 0) * (dev[b] ?? 0)
        const w = Math.exp(-q / inS.length / (2 * h * h))
        sw += w
        sf += w * (f[r] ?? 0)
      }
      return sw === 0 ? f.reduce((a, b) => a + b, 0) / bg.n : sf / sw
    }
  }
  const ns = options.nSamples ?? 100
  return (s) => {
    const inS = Array.from({ length: d }, (_, j) => j).filter((j) => s[j] === true)
    const out = Array.from({ length: d }, (_, j) => j).filter((j) => s[j] !== true)
    if (out.length === 0) return evalRows(predict, Float64Array.from(xv), 1, d)[0] ?? 0
    const mean = gaussianConditionalMean(xv, s, mu, sigma)
    // The conditional covariance of the missing features: S_oo - S_os S_ss^-1 S_so.
    const soo = out.map((a) => out.map((b) => sigma[a]?.[b] ?? 0))
    let cov = soo
    if (inS.length > 0) {
      const inv = toArray(pinv(inS.map((a) => inS.map((b) => sigma[a]?.[b] ?? 0)))) as number[][]
      cov = out.map((a, p) =>
        out.map((b, q) => {
          let t = 0
          for (let u = 0; u < inS.length; u++)
            for (let w = 0; w < inS.length; w++)
              t += (sigma[a]?.[inS[u] ?? 0] ?? 0) * (inv[u]?.[w] ?? 0) * (sigma[inS[w] ?? 0]?.[b] ?? 0)
          return (soo[p]?.[q] ?? 0) - t
        }),
      )
    }
    const l = toArray(choleskyJitter(cov, 1e-3).l) as number[][]
    const bits = s.reduce((b, v, j) => (v ? b | (1 << j) : b), 0)
    const z = toFlat(normal(foldIn(key, bits), [ns, out.length]))
    const data = new Float64Array(ns * d)
    for (let t = 0; t < ns; t++) {
      for (let j = 0; j < d; j++) data[t * d + j] = mean[j] ?? 0
      out.forEach((j, p) => {
        let e = 0
        for (let q = 0; q <= p; q++) e += (l[p]?.[q] ?? 0) * (z[t * out.length + q] ?? 0)
        data[t * d + j] = (mean[j] ?? 0) + e
      })
    }
    return evalRows(predict, data, ns, d).reduce((a, b) => a + b, 0) / ns
  }
}

/** Options of {@link sage}. */
export interface SageOptions extends FitOptions {
  /** The loss (default `mse`; `cross-entropy` reads the output as P(y = 1), or class probabilities `[m, k]`). */
  readonly loss?: 'mse' | 'cross-entropy'
  /** Permutations sampled (default 256); ignored when `exact`. */
  readonly nPermutations?: number
  /** Enumerate every coalition (at most 12 features). */
  readonly exact?: boolean
}

/** SAGE values: each feature's share of the reduction in loss the features achieve together. */
export interface SageResult {
  readonly values: readonly number[]
  /** Standard errors of the sampled estimates (0 when exact). */
  readonly standardErrors: readonly number[]
  /** The loss with no features (the background's mean prediction) and with all of them. */
  readonly lossEmpty: number
  readonly lossFull: number
}

/**
 * SAGE (Covert, Lundberg and Lee 2020): global feature importance as Shapley values of the game v(S) = -E[loss(y,
 * f_S(x))], where f_S averages the model over background values of the features outside S (marginal imputation).
 * The values sum to the loss without features minus the loss with all of them, and a feature the model ignores gets
 * 0. Estimated by permutation sampling (each step a data row and a feature order), or exactly for few features.
 *
 * @param predict - The model.
 * @param data - Evaluation rows with targets.
 * @param background - Background rows `[m, d]` for imputation.
 * @param options - loss, nPermutations, exact, key and tracing.
 * @returns The values, their errors and the two end losses.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const first = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i]))
 * const x = [[0, 5], [2, -5]]
 * // Only the first feature matters: it removes all of the loss (1 at the mean prediction).
 * explain.sage(first, { x, y: [0, 2] }, x, { exact: true }).values // => [1, 0]
 * ```
 */
export function sage(
  predict: PredictFunction,
  data: LearnData,
  background: TensorLike,
  options: SageOptions = {},
): SageResult {
  const m = matrixView(data.x, 'sage')
  requireTargets(data, 'sage')
  const y = toFlat(asTensor(data.y as TensorLike))
  const bg = matrixView(background, 'sage', { nFeatures: m.d })
  const d = m.d
  const ce = options.loss === 'cross-entropy'
  // The imputed prediction for row i with features in S at its values: the mean over background rows.
  const imputed = (i: number, s: readonly boolean[]): Float64Array => {
    const rows = new Float64Array(bg.n * d)
    for (let r = 0; r < bg.n; r++)
      for (let j = 0; j < d; j++) rows[r * d + j] = s[j] === true ? (m.data[i * d + j] ?? 0) : (bg.data[r * d + j] ?? 0)
    const out = predict(fromStorage(rows, [bg.n, d], 'float64'))
    const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
    const v = toFlat(out)
    const mean = new Float64Array(k)
    for (let r = 0; r < bg.n; r++) for (let c = 0; c < k; c++) mean[c] = (mean[c] ?? 0) + (v[r * k + c] ?? 0) / bg.n
    return mean
  }
  const lossOf = (i: number, p: Float64Array): number => {
    const yi = y[i] ?? 0
    if (!ce) return ((p[0] ?? 0) - yi) ** 2
    const clip = (q: number): number => Math.min(1 - 1e-12, Math.max(1e-12, q))
    if (p.length === 1) return -(yi * Math.log(clip(p[0] ?? 0)) + (1 - yi) * Math.log(clip(1 - (p[0] ?? 0))))
    return -Math.log(clip(p[Math.round(yi)] ?? 0))
  }
  const all = new Array<boolean>(d).fill(true)
  const none = new Array<boolean>(d).fill(false)
  let lossEmpty = 0
  let lossFull = 0
  for (let i = 0; i < m.n; i++) {
    lossEmpty += lossOf(i, imputed(i, none)) / m.n
    lossFull += lossOf(i, imputed(i, all)) / m.n
  }
  if (options.exact === true) {
    if (d > 12) throw new WhiteboxError('value', 'sage: exact SAGE takes at most 12 features')
    const game: CoalitionValue = (s) => {
      let l = 0
      for (let i = 0; i < m.n; i++) l += lossOf(i, imputed(i, s)) / m.n
      return -l
    }
    return { values: shapleyValues(game, d), standardErrors: new Array<number>(d).fill(0), lossEmpty, lossFull }
  }
  const nPerm = options.nPermutations ?? 256
  const key = keyFor(options, 'sage')
  type State = { readonly sum: Float64Array; readonly sumSq: Float64Array; readonly count: number }
  const algorithm: Algorithm<State> = {
    name: 'sage',
    config: { features: d, nPermutations: nPerm },
    schema: channels({ metrics: { count: { description: 'permutations so far' } } }),
    init: () => ({ sum: new Float64Array(d), sumSq: new Float64Array(d), count: 0 }),
    step: (st) => {
      const k = foldIn(key, st.count)
      const i = toFlat(randint(foldIn(k, 0), [1], 0, m.n))[0] ?? 0
      const order = toFlat(permutation(foldIn(k, 1), d))
      const s = new Array<boolean>(d).fill(false)
      let prev = lossOf(i, imputed(i, s))
      const sum = Float64Array.from(st.sum)
      const sumSq = Float64Array.from(st.sumSq)
      for (const j of order) {
        s[j] = true
        const cur = lossOf(i, imputed(i, s))
        const c = prev - cur
        sum[j] = (sum[j] ?? 0) + c
        sumSq[j] = (sumSq[j] ?? 0) + c * c
        prev = cur
      }
      return { sum, sumSq, count: st.count + 1 }
    },
    done: (st) => (st.count >= nPerm ? 'completed' : undefined),
    observe: (st) => ({ metrics: { count: st.count } }),
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'sage'), maxSteps: nPerm })
  const values = Array.from(state.sum, (v) => v / state.count)
  const standardErrors = Array.from(state.sumSq, (v, j) => {
    const mean = values[j] ?? 0
    return state.count < 2 ? Number.NaN : Math.sqrt(Math.max(0, v / state.count - mean * mean) / (state.count - 1))
  })
  return { values, standardErrors, lossEmpty, lossFull }
}
