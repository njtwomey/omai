/**
 * Click models: what a log of ranked lists and clicks says about which results are relevant when users look at the
 * top of the list more than the bottom. The position-based model (a click needs the rank to be examined and the result
 * attractive, fitted by EM), the cascade model (the user scans down and stops at the first click; closed form) and the
 * dynamic Bayesian network model of Chapelle and Zhang (attractiveness, satisfaction and a continuation probability,
 * fitted by EM with a forward-backward pass over the examination chain), each with click probabilities and the
 * log-likelihood of a log. The EM fits are `Algorithm`s, one iteration per step, with serialisable states.
 */

import { type Json, MdmlError, maxOf } from '../base/index.ts'
import type { InteractionData } from '../data/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'

/** One search session: the items shown from the top and whether each was clicked. */
export interface ClickSession {
  readonly items: readonly number[]
  readonly clicks: readonly number[]
}

/**
 * The sessions of an interaction log: rows grouped by their session and ordered by their display position (the
 * position column; row order when absent), the click being the row's value.
 *
 * @param data - An interaction dataset with sessions (or users as sessions) and clicks as values.
 * @returns The sessions in order of session id.
 * @throws MdmlError with code `value` for a log without sessions or positions.
 * @concept click-models
 * @example
 * ```ts
 * import { clickSessions } from 'mdml-engine/learn'
 * import { interactions } from 'mdml-engine/data'
 * clickSessions(interactions({ users: [0, 0, 1, 1], items: [3, 1, 2, 3], values: [0, 1, 1, 0], positions: [0, 1, 0, 1], sessions: [0, 0, 1, 1] })) // => [{ items: [3, 1], clicks: [0, 1] }, { items: [2, 3], clicks: [1, 0] }]
 * ```
 */
export function clickSessions(data: InteractionData): ClickSession[] {
  const sessions = toFlat(data.sessions ?? data.users)
  const items = toFlat(data.items)
  const values = toFlat(data.values)
  if (data.positions === undefined && data.sessions === undefined)
    throw new MdmlError('value', 'clickSessions: needs sessions and display positions')
  const positions = data.positions === undefined ? undefined : toFlat(data.positions)
  const bySession = new Map<number, { rank: number; item: number; click: number }[]>()
  for (let i = 0; i < items.length; i++) {
    const list = bySession.get(sessions[i] as number) ?? []
    list.push({
      rank: positions?.[i] ?? list.length,
      item: items[i] as number,
      click: (values[i] as number) > 0 ? 1 : 0,
    })
    bySession.set(sessions[i] as number, list)
  }
  return [...bySession]
    .sort((a, b) => a[0] - b[0])
    .map(([, rows]) => {
      rows.sort((a, b) => a.rank - b.rank)
      return { items: rows.map((r) => r.item), clicks: rows.map((r) => r.click) }
    })
}

const EPS = 1e-9
const clamp = (x: number): number => Math.min(1 - EPS, Math.max(EPS, x))

// ---------------------------------------------------------------------------------------------------------------
// Position-based model.

/** The position-based model: a click needs the rank examined (probability `examination[rank]`) and the item attractive. */
export interface PositionBasedModel {
  /** The probability each rank is examined, normalised to 1 at the top (the product with attractiveness is what is identified). */
  readonly examination: readonly number[]
  /** The probability an examined item is clicked, per item. */
  readonly attractiveness: readonly number[]
  /** The log likelihood of the training sessions. */
  readonly logLikelihood: number
}

/** The state of {@link positionBasedAlgorithm}: the raw parameters (examination at the top not fixed) and the likelihood. */
export interface PositionBasedState {
  readonly gamma: readonly number[]
  readonly alpha: readonly number[]
  readonly logLikelihood: number
  readonly change: number
  readonly iteration: number
}

function pbmLogLikelihood(
  sessions: readonly ClickSession[],
  gamma: readonly number[],
  alpha: readonly number[],
): number {
  let ll = 0
  for (const s of sessions)
    for (const [r, item] of s.items.entries()) {
      const p = clamp((gamma[r] ?? 0) * (alpha[item] ?? 0))
      ll += s.clicks[r] === 1 ? Math.log(p) : Math.log(1 - p)
    }
  return ll
}

/**
 * EM for the position-based model as an {@link Algorithm}, one iteration per step. The E-step gives, for each shown
 * item, the probability its rank was examined and the item attractive given the click (both certain after a click) or
 * the missing click; the M-step sets each rank's examination and each item's attractiveness to the mean of those
 * probabilities (Chuklin, Markov and de Rijke 2015, section 4.1).
 *
 * @param sessions - The click sessions.
 * @param nItems - The number of items.
 * @param options - The tolerance on parameter change (default 1e-8).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for no sessions.
 * @concept click-models
 * @example
 * ```ts
 * import { positionBasedAlgorithm } from 'mdml-engine/learn'
 * positionBasedAlgorithm([{ items: [0, 1], clicks: [1, 0] }], 2).name // => 'positionBasedEm'
 * ```
 */
export function positionBasedAlgorithm(
  sessions: readonly ClickSession[],
  nItems: number,
  options: { readonly tol?: number } = {},
): Algorithm<PositionBasedState> {
  if (sessions.length === 0) throw new MdmlError('value', 'positionBasedModel: needs sessions')
  const depth = maxOf(sessions.map((s) => s.items.length))
  const tol = options.tol ?? 1e-8
  return {
    name: 'positionBasedEm',
    config: { sessions: sessions.length, items: nItems, ranks: depth } as Json,
    init: () => {
      const gamma = new Array<number>(depth).fill(0.5)
      const alpha = new Array<number>(nItems).fill(0.5)
      return {
        gamma,
        alpha,
        logLikelihood: pbmLogLikelihood(sessions, gamma, alpha),
        change: Number.POSITIVE_INFINITY,
        iteration: 0,
      }
    },
    step: (s) => {
      const eNum = new Float64Array(depth)
      const eDen = new Float64Array(depth)
      const aNum = new Float64Array(nItems)
      const aDen = new Float64Array(nItems)
      for (const sess of sessions)
        for (const [r, item] of sess.items.entries()) {
          const g = s.gamma[r] ?? 0
          const a = s.alpha[item] ?? 0
          let pe: number
          let pa: number
          if (sess.clicks[r] === 1) {
            pe = 1
            pa = 1
          } else {
            const miss = 1 - g * a
            pe = (g * (1 - a)) / miss
            pa = (a * (1 - g)) / miss
          }
          eNum[r] = (eNum[r] ?? 0) + pe
          eDen[r] = (eDen[r] ?? 0) + 1
          aNum[item] = (aNum[item] ?? 0) + pa
          aDen[item] = (aDen[item] ?? 0) + 1
        }
      const gamma = Array.from(eNum, (v, r) => clamp(v / (eDen[r] || 1)))
      const alpha = Array.from(aNum, (v, i) =>
        (aDen[i] ?? 0) === 0 ? (s.alpha[i] ?? 0.5) : clamp(v / (aDen[i] as number)),
      )
      let change = 0
      for (let r = 0; r < depth; r++) change = Math.max(change, Math.abs((gamma[r] ?? 0) - (s.gamma[r] ?? 0)))
      for (let i = 0; i < nItems; i++) change = Math.max(change, Math.abs((alpha[i] ?? 0) - (s.alpha[i] ?? 0)))
      return {
        gamma,
        alpha,
        logLikelihood: pbmLogLikelihood(sessions, gamma, alpha),
        change,
        iteration: s.iteration + 1,
      }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { logLikelihood: s.logLikelihood, change: Math.min(s.change, 1e300) } }),
    serialise: (s) => ({ ...s, change: Math.min(s.change, 1e300) }) as unknown as Json,
    deserialise: (j) => j as unknown as PositionBasedState,
  }
}

/**
 * The model that a state of the position-based EM stands for, with the examination at the top rank fixed to 1 (the
 * attractiveness absorbs the scale, since only their product is identified).
 *
 * @param s - A state of {@link positionBasedAlgorithm}.
 * @returns The model.
 * @concept click-models
 * @example
 * ```ts
 * import { positionBasedFromState } from 'mdml-engine/learn'
 * positionBasedFromState({ gamma: [0.8, 0.4], alpha: [0.5], logLikelihood: 0, change: 0, iteration: 1 }).examination // => [1, 0.5]
 * ```
 */
export function positionBasedFromState(s: PositionBasedState): PositionBasedModel {
  const top = s.gamma[0] ?? 1
  return {
    examination: s.gamma.map((g) => g / top),
    attractiveness: s.alpha.map((a) => a * top),
    logLikelihood: s.logLikelihood,
  }
}

/**
 * Fit the position-based click model by EM.
 *
 * @param sessions - The click sessions.
 * @param nItems - The number of items.
 * @param options - The tolerance and the iteration cap (default 200).
 * @returns The fitted model (examination normalised to 1 at the top rank) and the iterations used.
 * @concept click-models
 * @example
 * ```ts
 * import { positionBasedModel } from 'mdml-engine/learn'
 * positionBasedModel([{ items: [0, 1], clicks: [1, 0] }, { items: [1, 0], clicks: [0, 1] }], 2).model.examination.length // => 2
 * ```
 */
export function positionBasedModel(
  sessions: readonly ClickSession[],
  nItems: number,
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): { readonly model: PositionBasedModel; readonly iterations: number } {
  const r = runAlgorithm(positionBasedAlgorithm(sessions, nItems, options), {
    maxSteps: options.maxIter ?? 200,
    seed: 0,
  })
  return { model: positionBasedFromState(r.state), iterations: r.steps }
}

/**
 * The click probabilities the position-based model gives each rank of a list.
 *
 * @param model - The fitted model.
 * @param items - The items shown from the top.
 * @returns `P(click)` at every rank.
 * @concept click-models
 * @example
 * ```ts
 * import { positionBasedClickProbabilities } from 'mdml-engine/learn'
 * positionBasedClickProbabilities({ examination: [1, 0.5], attractiveness: [0.4, 0.8], logLikelihood: 0 }, [0, 1]) // => [0.4, 0.4]
 * ```
 */
export function positionBasedClickProbabilities(model: PositionBasedModel, items: readonly number[]): number[] {
  return items.map((i, r) => (model.examination[r] ?? 0) * (model.attractiveness[i] ?? 0))
}

// ---------------------------------------------------------------------------------------------------------------
// Cascade model.

/** The cascade model: the user scans down, clicks the first attractive item and stops. */
export interface CascadeModel {
  /** The probability an examined item is clicked, per item. */
  readonly attractiveness: readonly number[]
  /** The log likelihood of the training sessions. */
  readonly logLikelihood: number
}

/**
 * The cascade model fitted in closed form (Craswell et al. 2008): in a session the items down to the first click were
 * examined (all of them without a click), so an item's attractiveness is its clicks over its examinations. Sessions with
 * clicks after the first are outside the model; only their first click is used.
 *
 * @param sessions - The click sessions.
 * @param nItems - The number of items.
 * @param smoothing - Pseudo-counts added to clicks and examinations (default 1 and 2, a Beta(1, 1) prior).
 * @returns The model.
 * @concept click-models
 * @example
 * ```ts
 * import { cascadeModel } from 'mdml-engine/learn'
 * cascadeModel([{ items: [0, 1], clicks: [0, 1] }, { items: [0, 1], clicks: [1, 0] }], 2, [0, 0]).attractiveness.map((a) => Math.round(a * 100) / 100) // => [0.5, 1]
 * ```
 */
export function cascadeModel(
  sessions: readonly ClickSession[],
  nItems: number,
  smoothing: readonly [number, number] = [1, 2],
): CascadeModel {
  const clicks = new Float64Array(nItems)
  const looks = new Float64Array(nItems)
  for (const s of sessions) {
    const first = s.clicks.indexOf(1)
    const last = first < 0 ? s.items.length - 1 : first
    for (let r = 0; r <= last; r++) {
      const item = s.items[r] as number
      looks[item] = (looks[item] ?? 0) + 1
      if (r === first) clicks[item] = (clicks[item] ?? 0) + 1
    }
  }
  const attractiveness = Array.from(clicks, (c, i) =>
    (looks[i] ?? 0) + smoothing[1] === 0 ? 0 : clamp((c + smoothing[0]) / ((looks[i] ?? 0) + smoothing[1])),
  )
  return { attractiveness, logLikelihood: cascadeLogLikelihood(sessions, attractiveness) }
}

/** The log likelihood of sessions under a cascade model: each shown item down to the first click is a Bernoulli draw. */
function cascadeLogLikelihood(sessions: readonly ClickSession[], attractiveness: readonly number[]): number {
  let ll = 0
  for (const s of sessions) {
    const first = s.clicks.indexOf(1)
    const last = first < 0 ? s.items.length - 1 : first
    for (let r = 0; r <= last; r++) {
      const a = clamp(attractiveness[s.items[r] as number] ?? 0)
      ll += r === first ? Math.log(a) : Math.log(1 - a)
    }
  }
  return ll
}

/**
 * The click probabilities the cascade model gives each rank of a list: the item must be attractive and every item above
 * it unattractive.
 *
 * @param model - The fitted model.
 * @param items - The items shown from the top.
 * @returns `P(click)` at every rank.
 * @concept click-models
 * @example
 * ```ts
 * import { cascadeClickProbabilities } from 'mdml-engine/learn'
 * cascadeClickProbabilities({ attractiveness: [0.5, 0.5], logLikelihood: 0 }, [0, 1]) // => [0.5, 0.25]
 * ```
 */
export function cascadeClickProbabilities(model: CascadeModel, items: readonly number[]): number[] {
  let reach = 1
  return items.map((i) => {
    const a = model.attractiveness[i] ?? 0
    const p = reach * a
    reach *= 1 - a
    return p
  })
}

// ---------------------------------------------------------------------------------------------------------------
// Dynamic Bayesian network model.

/** The dynamic Bayesian network click model of Chapelle and Zhang (2009). */
export interface DbnModel {
  /** The probability an examined item is clicked, per item. */
  readonly attractiveness: readonly number[]
  /** The probability a clicked item satisfies the user, per item. */
  readonly satisfaction: readonly number[]
  /** The probability an unsatisfied user goes on to the next result. */
  readonly continuation: number
  /** The log likelihood of the training sessions. */
  readonly logLikelihood: number
}

/** The state of {@link dbnAlgorithm}. */
export interface DbnState extends DbnModel {
  readonly change: number
  readonly iteration: number
}

/** The forward-backward pass of a session: the posterior of examination at each rank, of satisfaction at clicks, and the likelihood. */
function dbnPosteriors(
  s: ClickSession,
  a: readonly number[],
  sat: readonly number[],
  gamma: number,
): { examined: number[]; satisfied: number[]; likelihood: number } {
  const K = s.items.length
  const emit = (e: number, r: number): number => {
    const aa = a[s.items[r] as number] ?? 0
    const click = s.clicks[r] === 1
    return e === 0 ? (click ? 0 : 1) : click ? aa : 1 - aa
  }
  // T1[r]: probability the user is still examining at r + 1 given examining at r.
  const t1 = (r: number): number => (s.clicks[r] === 1 ? (1 - (sat[s.items[r] as number] ?? 0)) * gamma : gamma)
  const alpha = Array.from({ length: K + 1 }, () => [0, 0] as [number, number])
  alpha[0] = [0, 1]
  for (let r = 0; r < K; r++) {
    const f0 = (alpha[r] as [number, number])[0] * emit(0, r)
    const f1 = (alpha[r] as [number, number])[1] * emit(1, r)
    alpha[r + 1] = [f0 + f1 * (1 - t1(r)), f1 * t1(r)]
  }
  const beta = Array.from({ length: K + 1 }, () => [1, 1] as [number, number])
  for (let r = K - 1; r >= 0; r--) {
    const next = beta[r + 1] as [number, number]
    beta[r] = [emit(0, r) * next[0], emit(1, r) * (t1(r) * next[1] + (1 - t1(r)) * next[0])]
  }
  const likelihood = (alpha[K] as [number, number])[0] + (alpha[K] as [number, number])[1]
  const examined: number[] = []
  const satisfied: number[] = []
  for (let r = 0; r < K; r++) {
    const al = alpha[r] as [number, number]
    const be = beta[r] as [number, number]
    const z = al[0] * be[0] + al[1] * be[1]
    examined.push(z === 0 ? 0 : (al[1] * be[1]) / z)
    if (s.clicks[r] === 1) {
      const sr = sat[s.items[r] as number] ?? 0
      const next = beta[r + 1] as [number, number]
      const pS = sr * next[0]
      const pCont = (1 - sr) * gamma * next[1]
      const pStop = (1 - sr) * (1 - gamma) * next[0]
      satisfied.push(pS + pCont + pStop === 0 ? 0 : pS / (pS + pCont + pStop))
    } else satisfied.push(0)
  }
  return { examined, satisfied, likelihood }
}

function dbnLogLikelihood(
  sessions: readonly ClickSession[],
  a: readonly number[],
  sat: readonly number[],
  gamma: number,
): number {
  return sessions.reduce((acc, s) => acc + Math.log(Math.max(dbnPosteriors(s, a, sat, gamma).likelihood, 1e-300)), 0)
}

/**
 * EM for the dynamic Bayesian network click model as an {@link Algorithm}, one iteration per step. The E-step runs a
 * forward-backward pass over each session's chain of examination states (an examined item is clicked with its
 * attractiveness; after a click the user is satisfied with the item's satisfaction and stops; otherwise they continue
 * with the continuation probability), and the M-step sets an item's attractiveness to its clicks over its expected
 * examinations, its satisfaction to the expected share of its clicks that satisfied, and the continuation to the
 * expected share of unsatisfied examinations that went on.
 *
 * @param sessions - The click sessions.
 * @param nItems - The number of items.
 * @param options - The tolerance on parameter change (default 1e-8) and the pseudo-counts for attractiveness and satisfaction (default Beta(1, 1)).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for no sessions.
 * @concept click-models
 * @example
 * ```ts
 * import { dbnAlgorithm } from 'mdml-engine/learn'
 * dbnAlgorithm([{ items: [0, 1], clicks: [1, 0] }], 2).name // => 'dbnEm'
 * ```
 */
export function dbnAlgorithm(
  sessions: readonly ClickSession[],
  nItems: number,
  options: { readonly tol?: number; readonly prior?: number } = {},
): Algorithm<DbnState> {
  if (sessions.length === 0) throw new MdmlError('value', 'dbn: needs sessions')
  const tol = options.tol ?? 1e-8
  const prior = options.prior ?? 1
  return {
    name: 'dbnEm',
    config: { sessions: sessions.length, items: nItems } as Json,
    init: () => {
      const attractiveness = new Array<number>(nItems).fill(0.5)
      const satisfaction = new Array<number>(nItems).fill(0.5)
      return {
        attractiveness,
        satisfaction,
        continuation: 0.9,
        logLikelihood: dbnLogLikelihood(sessions, attractiveness, satisfaction, 0.9),
        change: Number.POSITIVE_INFINITY,
        iteration: 0,
      }
    },
    step: (s) => {
      const aNum = new Float64Array(nItems)
      const aDen = new Float64Array(nItems)
      const sNum = new Float64Array(nItems)
      const sDen = new Float64Array(nItems)
      let gNum = 0
      let gDen = 0
      for (const sess of sessions) {
        const post = dbnPosteriors(sess, s.attractiveness, s.satisfaction, s.continuation)
        const K = sess.items.length
        for (let r = 0; r < K; r++) {
          const item = sess.items[r] as number
          aDen[item] = (aDen[item] ?? 0) + (post.examined[r] ?? 0)
          if (sess.clicks[r] === 1) {
            aNum[item] = (aNum[item] ?? 0) + 1
            sNum[item] = (sNum[item] ?? 0) + (post.satisfied[r] ?? 0)
            sDen[item] = (sDen[item] ?? 0) + 1
          }
          if (r < K - 1) {
            gNum += post.examined[r + 1] ?? 0
            gDen += (post.examined[r] ?? 0) - (post.satisfied[r] ?? 0)
          }
        }
      }
      const attractiveness = Array.from(aNum, (v, i) => clamp((v + prior) / ((aDen[i] ?? 0) + 2 * prior)))
      const satisfaction = Array.from(sNum, (v, i) => clamp((v + prior) / ((sDen[i] ?? 0) + 2 * prior)))
      const continuation = clamp(gDen === 0 ? s.continuation : gNum / gDen)
      let change = Math.abs(continuation - s.continuation)
      for (let i = 0; i < nItems; i++)
        change = Math.max(
          change,
          Math.abs((attractiveness[i] ?? 0) - (s.attractiveness[i] ?? 0)),
          Math.abs((satisfaction[i] ?? 0) - (s.satisfaction[i] ?? 0)),
        )
      return {
        attractiveness,
        satisfaction,
        continuation,
        logLikelihood: dbnLogLikelihood(sessions, attractiveness, satisfaction, continuation),
        change,
        iteration: s.iteration + 1,
      }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { logLikelihood: s.logLikelihood, continuation: s.continuation, change: Math.min(s.change, 1e300) },
    }),
    serialise: (s) => ({ ...s, change: Math.min(s.change, 1e300) }) as unknown as Json,
    deserialise: (j) => j as unknown as DbnState,
  }
}

/**
 * Fit the dynamic Bayesian network click model by EM.
 *
 * @param sessions - The click sessions.
 * @param nItems - The number of items.
 * @param options - The tolerance, the iteration cap (default 200) and the pseudo-count.
 * @returns The fitted model and the iterations used.
 * @concept click-models
 * @example
 * ```ts
 * import { dbnModel } from 'mdml-engine/learn'
 * dbnModel([{ items: [0, 1], clicks: [1, 0] }, { items: [1, 0], clicks: [0, 0] }], 2, { maxIter: 5 }).model.attractiveness.length // => 2
 * ```
 */
export function dbnModel(
  sessions: readonly ClickSession[],
  nItems: number,
  options: { readonly tol?: number; readonly maxIter?: number; readonly prior?: number } = {},
): { readonly model: DbnModel; readonly iterations: number } {
  const r = runAlgorithm(dbnAlgorithm(sessions, nItems, options), { maxSteps: options.maxIter ?? 200, seed: 0 })
  const { attractiveness, satisfaction, continuation, logLikelihood } = r.state
  return { model: { attractiveness, satisfaction, continuation, logLikelihood }, iterations: r.steps }
}

/**
 * The click probabilities the dynamic Bayesian network gives each rank of a list, given no clicks above it is not
 * assumed: the marginal over the examination chain (the unconditional click probability at each rank).
 *
 * @param model - The fitted model.
 * @param items - The items shown from the top.
 * @returns `P(click)` at every rank.
 * @concept click-models
 * @example
 * ```ts
 * import { dbnClickProbabilities } from 'mdml-engine/learn'
 * dbnClickProbabilities({ attractiveness: [0.5, 0.5], satisfaction: [1, 1], continuation: 1, logLikelihood: 0 }, [0, 1]) // => [0.5, 0.25]
 * ```
 */
export function dbnClickProbabilities(model: DbnModel, items: readonly number[]): number[] {
  const out: number[] = []
  let examine = 1
  for (const i of items) {
    const a = model.attractiveness[i] ?? 0
    const s = model.satisfaction[i] ?? 0
    out.push(examine * a)
    // continue if (examined and not clicked) or (clicked and not satisfied), then with probability gamma
    examine = examine * (1 - a + a * (1 - s)) * model.continuation
  }
  return out
}
