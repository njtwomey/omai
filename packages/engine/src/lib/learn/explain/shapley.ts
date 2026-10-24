import { popcount, WhiteboxError } from '../../base/index.ts'
import { type CoalitionalGame, coalitionalGame, shapleyInteractionIndex, shapleyValue } from '../../games/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, permutation, randint } from '../../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { keyFor, runOptionsFor } from '../_shared.ts'
import type { FitOptions } from '../estimator.ts'
import { type MatrixView, matrixView } from '../validation.ts'

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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
