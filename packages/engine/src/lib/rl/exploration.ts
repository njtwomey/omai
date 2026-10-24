import { WhiteboxError } from '../base/index.ts'
import { type Key, normal } from '../random/index.ts'
import { asTensor, mean, square, sub, sum, type TensorLike, toFlat, type Value } from '../tensor/index.ts'

/** Visit counts keyed by a state code (persistent: updates return a new table). */
export interface CountTable {
  readonly counts: Readonly<Record<string, number>>
  readonly total: number
}

/**
 * An empty count table.
 *
 * @returns The table.
 * @concept exploration
 * @example
 * ```ts
 * import { countTable } from 'whitebox-engine/rl'
 * countTable().total // => 0
 * ```
 */
export function countTable(): CountTable {
  return { counts: {}, total: 0 }
}

/**
 * Count a visit.
 *
 * @param table - The table.
 * @param code - The state's code (a tabular index, a hash, a discretisation).
 * @returns The new table.
 * @concept exploration
 * @example
 * ```ts
 * import { countTable, countVisit, visits } from 'whitebox-engine/rl'
 * visits(countVisit(countVisit(countTable(), 's'), 's'), 's') // => 2
 * ```
 */
export function countVisit(table: CountTable, code: string | number): CountTable {
  const k = String(code)
  return { counts: { ...table.counts, [k]: (table.counts[k] ?? 0) + 1 }, total: table.total + 1 }
}

/**
 * The visit count of a code.
 *
 * @param table - The table.
 * @param code - The code.
 * @returns Its count (0 if never visited).
 * @concept exploration
 * @example
 * ```ts
 * import { countTable, visits } from 'whitebox-engine/rl'
 * visits(countTable(), 3) // => 0
 * ```
 */
export function visits(table: CountTable, code: string | number): number {
  return table.counts[String(code)] ?? 0
}

/**
 * The count-based exploration bonus `beta / sqrt(n)` (MBIE-EB, Strehl and Littman 2008; #Exploration, Tang et al.
 * 2017), with unvisited states counted once.
 *
 * @param count - The visit count.
 * @param beta - The bonus scale (default 1).
 * @returns The bonus.
 * @concept exploration
 * @example
 * ```ts
 * import { countBonus } from 'whitebox-engine/rl'
 * countBonus(4, 0.5) // => 0.25
 * ```
 */
export function countBonus(count: number, beta = 1): number {
  return beta / Math.sqrt(Math.max(count, 1))
}

/**
 * A pseudo-count from a density model (Bellemare et al. 2016): with `rho` the model's probability of a state before an
 * update on it and `rhoPrime` after, `N = rho (1 - rhoPrime) / (rhoPrime - rho)`.
 *
 * @param rho - The probability before the update.
 * @param rhoPrime - The recoding probability after it.
 * @returns The pseudo-count (`Infinity` when the update does not raise the probability).
 * @concept exploration
 * @example
 * ```ts
 * import { pseudoCount } from 'whitebox-engine/rl'
 * pseudoCount(0.25, 0.5) // => 0.5
 * ```
 */
export function pseudoCount(rho: number, rhoPrime: number): number {
  if (!(rhoPrime > rho)) return Infinity
  return (rho * (1 - rhoPrime)) / (rhoPrime - rho)
}

/**
 * SimHash codes for continuous observations (Tang et al. 2017): the signs of a fixed Gaussian projection `A x`, with
 * `bits` hyperplanes, so nearby observations share a code and can be counted.
 *
 * @param key - The key for the projection.
 * @param dimension - The observation size.
 * @param bits - The number of hyperplanes (default 16).
 * @returns The hash function.
 * @concept exploration
 * @example
 * ```ts
 * import { simHash } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * const h = simHash(key(0), 2, 8)
 * h([1, 2]) === h([1.000001, 2]) // => true
 * ```
 */
export function simHash(key: Key, dimension: number, bits = 16): (observation: readonly number[]) => string {
  if (!(bits >= 1)) throw new WhiteboxError('value', 'simHash: need at least one bit')
  const a = normal(key, [bits, dimension]).data
  return (x) => {
    let code = ''
    for (let b = 0; b < bits; b++) {
      let s = 0
      for (let d = 0; d < dimension; d++) s += (a[b * dimension + d] ?? 0) * (x[d] ?? 0)
      code += s >= 0 ? '1' : '0'
    }
    return code
  }
}

/**
 * Random network distillation (Burda et al. 2019): a fixed, randomly initialised target network and a predictor
 * trained to match it; the prediction error is large on novel observations.
 *
 * @concept exploration
 */
export interface RandomNetworkDistillation<P> {
  /** The fixed target features `[B, k]`. */
  target(observations: Value): Value
  /** The predicted features `[B, k]` under the predictor's parameters. */
  predictor(params: P, observations: Value): Value
}

/**
 * The RND intrinsic reward per observation: the squared error between target and predicted features, summed over
 * features.
 *
 * @param rnd - The networks.
 * @param params - The predictor's parameters.
 * @param observations - A batch of observations `[B, d]`.
 * @returns One bonus per row.
 * @concept exploration
 * @example
 * ```ts
 * import { rndBonus } from 'whitebox-engine/rl'
 * import { mul } from 'whitebox-engine/tensor'
 * const rnd = { target: (x: never) => mul(x, 2), predictor: (w: number, x: never) => mul(x, w) }
 * rndBonus(rnd, 1, [[1, 2]]) // => [5]
 * ```
 */
export function rndBonus<P>(rnd: RandomNetworkDistillation<P>, params: P, observations: TensorLike): number[] {
  const x = asTensor(observations)
  return toFlat(sum(square(sub(rnd.target(x), rnd.predictor(params, x))), -1))
}

/**
 * The RND predictor's training loss: the mean squared feature error (differentiable in the predictor's parameters;
 * the target is fixed).
 *
 * @param rnd - The networks.
 * @param params - The predictor's parameters.
 * @param observations - A batch of observations.
 * @returns The scalar loss.
 * @concept exploration
 * @example
 * ```ts
 * import { rndLoss } from 'whitebox-engine/rl'
 * import { item, mul } from 'whitebox-engine/tensor'
 * const rnd = { target: (x: never) => mul(x, 2), predictor: (w: number, x: never) => mul(x, w) }
 * item(rndLoss(rnd, 1, [[1, 2]])) // => 2.5
 * ```
 */
export function rndLoss<P>(rnd: RandomNetworkDistillation<P>, params: P, observations: TensorLike): Value {
  const x = asTensor(observations)
  return mean(square(sub(rnd.target(x), rnd.predictor(params, x))))
}

/**
 * The intrinsic curiosity module (Pathak et al. 2017): an encoder `phi`, a forward model predicting `phi(s')` from
 * `phi(s)` and the action, and an inverse model predicting the action from `phi(s)` and `phi(s')`, which shapes the
 * features to what the agent controls.
 *
 * @concept exploration
 */
export interface CuriosityModule<P> {
  encode(params: P, observations: Value): Value
  /** The predicted next features `[B, k]`. */
  forward(params: P, features: Value, actions: Value): Value
  /** Action logits (discrete) or predicted actions (continuous) `[B, nA]`. */
  inverse(params: P, features: Value, nextFeatures: Value): Value
}

/**
 * The curiosity reward per transition: `eta / 2 ||forward(phi(s), a) - phi(s')||^2`.
 *
 * @param icm - The module.
 * @param params - Its parameters.
 * @param observations - Observations `[B, d]`.
 * @param actions - Actions (one-hot or continuous) `[B, a]`.
 * @param nextObservations - Next observations `[B, d]`.
 * @param eta - The scale (default 1).
 * @returns One bonus per transition.
 * @concept exploration
 * @example
 * ```ts
 * import { curiosityBonus } from 'whitebox-engine/rl'
 * import { add } from 'whitebox-engine/tensor'
 * const icm = { encode: (_: null, x: never) => x, forward: (_: null, f: never, a: never) => add(f, a), inverse: (_: null, f: never) => f }
 * curiosityBonus(icm, null, [[0]], [[1]], [[3]]) // => [2]
 * ```
 */
export function curiosityBonus<P>(
  icm: CuriosityModule<P>,
  params: P,
  observations: TensorLike,
  actions: TensorLike,
  nextObservations: TensorLike,
  eta = 1,
): number[] {
  const phi = icm.encode(params, asTensor(observations))
  const phiNext = icm.encode(params, asTensor(nextObservations))
  const err = toFlat(sum(square(sub(icm.forward(params, phi, asTensor(actions)), phiNext)), -1))
  return err.map((e) => (eta / 2) * e)
}
