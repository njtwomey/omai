/**
 * Decision transformers (Chen et al. 2021): offline RL as conditional sequence modelling. A small causal transformer
 * reads the interleaved return-to-go, state and action tokens of the last K steps and predicts each action from its
 * state token; acting conditions on a target return that falls by the rewards received. The offline learner, an
 * in-context agent that carries the episode's context forward, and helpers that train one on a dataset and evaluate it
 * over whole episodes.
 */

import { argmaxOf, type Json } from '../base/index.ts'
import { crossEntropy } from '../losses/index.ts'
import { gelu } from '../nn/index.ts'
import type { GradientTransformation, ParamTree } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, randint } from '../random/index.ts'
import { type Agent, categoricalPolicy, type Env, type Transition, trainAgent } from '../rl/index.ts'
import {
  add,
  div,
  fromStorage,
  matmul,
  mean,
  mul,
  oneHot,
  reshape,
  sliceAxis,
  softmax,
  split,
  sqrt,
  square,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import type { RunOptions } from '../trace/index.ts'
import { gradientStepWithTerms, optimiserOf, type Spaces, vectorTensor } from './_common.ts'
import { type Net, network, type OfflineSpaces, offlineSpaces, squashToBox } from './_offline.ts'
import {
  datasetEpisodes,
  type OfflineLearner,
  type OfflineOptions,
  type TransitionDataset,
  trainOffline,
} from './offline.ts'

/** Options of {@link decisionTransformer}. */
export interface DecisionTransformerOptions {
  /** Steps of context `K` (default 8); the sequence has `3K` tokens. */
  readonly context?: number
  /** The embedding width (default 32). */
  readonly embedding?: number
  /** Attention heads (default 2; must divide the embedding width). */
  readonly heads?: number
  /** Transformer blocks (default 1). */
  readonly layers?: number
  readonly learningRate?: number
  /** An optimiser replacing Adam (for example AdamW, as in the paper). */
  readonly optimizer?: GradientTransformation
  readonly maxGradNorm?: number
  /** The return to condition on when acting (default the best episode return in the dataset). */
  readonly targetReturn?: number
  /** Returns-to-go are divided by this (default the largest absolute episode return in the dataset, or 1). */
  readonly returnScale?: number
  /** The size of the timestep embedding table (default the longest episode in the dataset). */
  readonly maxTimestep?: number
  readonly key?: Key
}

/** The dataset's episodes as the model reads them. */
export interface DecisionTransformerData {
  readonly states: readonly (readonly (readonly number[])[])[]
  /** Encoded actions (one-hot or continuous) per step. */
  readonly actions: readonly (readonly (readonly number[])[])[]
  /** Class indices of discrete actions per step (empty for continuous ones). */
  readonly labels: readonly (readonly number[])[]
  /** Returns-to-go per step, unscaled. */
  readonly returnsToGo: readonly (readonly number[])[]
  /** The cumulative step count before each episode. */
  readonly offsets: readonly number[]
  readonly total: number
}

/** The state of a decision-transformer learner. */
export interface DecisionTransformerState {
  readonly params: ParamTree
  readonly optState: unknown
  readonly updates: number
  readonly loss: number
  readonly data: DecisionTransformerData
  readonly targetReturn: number
  readonly returnScale: number
  readonly maxTimestep: number
}

/** A context window as tensors. */
interface Window {
  readonly rtg: Tensor
  readonly states: Tensor
  readonly actions: Tensor
  readonly timesteps: Tensor
  /** 1 for real steps, 0 for left padding, `[B, K]`. */
  readonly valid: Tensor
}

// ---------------------------------------------------------------------------------------------- the network

interface Model {
  readonly params: ParamTree
  apply(p: ParamTree, w: Window): Value
}

type Layer = { readonly g: Value; readonly b: Value }
const layerNorm = (x: Value, p: Layer): Value => {
  const mu = mean(x, -1, true)
  const centred = sub(x, mu)
  const v = mean(square(centred), -1, true)
  return add(mul(div(centred, sqrt(add(v, 1e-5))), p.g), p.b)
}
const lnInit = (h: number): Layer => ({
  g: vectorTensor(new Array<number>(h).fill(1)),
  b: vectorTensor(new Array<number>(h).fill(0)),
})

function causalMask(L: number, valid: Tensor, K: number): Tensor {
  const B = valid.shape[0] ?? 1
  const v = valid.data
  const out = new Float64Array(B * L * L)
  for (let b = 0; b < B; b++)
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        const ok = j <= i && (v[b * K + Math.floor(j / 3)] ?? 0) > 0
        out[(b * L + i) * L + j] = ok ? 0 : -1e9
      }
  return fromStorage(out, [B, 1, L, L], 'float64')
}

function buildModel(
  sp: OfflineSpaces,
  K: number,
  h: number,
  heads: number,
  layers: number,
  maxT: number,
  key: Key,
): Model {
  const nets: Record<string, Net> = {
    returns: network([1, h], foldIn(key, 0)),
    states: network([sp.obsSize, h], foldIn(key, 1)),
    actions: network([sp.actionSize, h], foldIn(key, 2)),
    head: network([h, sp.actionSize], foldIn(key, 3)),
  }
  for (let l = 0; l < layers; l++) {
    nets[`qkv${l}`] = network([h, 3 * h], foldIn(key, 10 + 4 * l))
    nets[`proj${l}`] = network([h, h], foldIn(key, 11 + 4 * l))
    nets[`up${l}`] = network([h, 4 * h], foldIn(key, 12 + 4 * l))
    nets[`down${l}`] = network([4 * h, h], foldIn(key, 13 + 4 * l))
  }
  const params: Record<string, ParamTree> = {}
  for (const [name, n] of Object.entries(nets)) params[name] = n.params
  params['time'] = mul(normal(foldIn(key, 4), [maxT, h]), 0.02)
  params['lnIn'] = lnInit(h)
  params['lnOut'] = lnInit(h)
  for (let l = 0; l < layers; l++) {
    params[`ln1_${l}`] = lnInit(h)
    params[`ln2_${l}`] = lnInit(h)
  }
  const dh = h / heads
  const L = 3 * K
  const apply = (p: ParamTree, w: Window): Value => {
    const P = p as Readonly<Record<string, ParamTree>>
    const net = (name: string, x: Value): Value => (nets[name] as Net).apply(P[name] as ParamTree, x)
    const B = w.states.shape[0] ?? 1
    const time = matmul(oneHot(w.timesteps, maxT), P['time'] as Value)
    const r = add(net('returns', reshape(w.rtg, [B, K, 1])), time)
    const s = add(net('states', w.states), time)
    const a = add(net('actions', w.actions), time)
    let x = layerNorm(reshape(stack([r, s, a], 2), [B, L, h]), P['lnIn'] as Layer)
    const mask = causalMask(L, w.valid, K)
    const toHeads = (t: Value): Value => transpose(reshape(t, [B, L, heads, dh]), [0, 2, 1, 3])
    for (let l = 0; l < layers; l++) {
      const [q, k, v] = split(net(`qkv${l}`, layerNorm(x, P[`ln1_${l}`] as Layer)), 3, -1) as [Value, Value, Value]
      const scores = add(mul(matmul(toHeads(q), swapaxes(toHeads(k), -1, -2)), 1 / Math.sqrt(dh)), mask)
      const attended = matmul(softmax(scores, -1), toHeads(v))
      x = add(x, net(`proj${l}`, reshape(transpose(attended, [0, 2, 1, 3]), [B, L, h])))
      x = add(x, net(`down${l}`, gelu(net(`up${l}`, layerNorm(x, P[`ln2_${l}`] as Layer)))))
    }
    x = layerNorm(x, P['lnOut'] as Layer)
    // The prediction of a_t is read from the output at the state token s_t.
    const stateTokens = reshape(sliceAxis(reshape(x, [B, K, 3, h]), 2, 1, 2), [B, K, h])
    const out = net('head', stateTokens)
    return sp.discrete ? out : squashToBox(sp, out)
  }
  return { params, apply }
}

// ------------------------------------------------------------------------------------------- data and windows

function trajectories<O, A>(sp: OfflineSpaces, dataset: TransitionDataset<O, A>): DecisionTransformerData {
  const episodes = datasetEpisodes(dataset)
  const encode = (a: A): number[] => {
    if (!sp.discrete) return [...(a as readonly number[])]
    const x = new Array<number>(sp.actionSize).fill(0)
    x[a as number] = 1
    return x
  }
  const offsets: number[] = []
  let total = 0
  for (const e of episodes) {
    offsets.push(total)
    total += e.length
  }
  return {
    states: episodes.map((e) => e.map((t) => sp.features(t.observation))),
    actions: episodes.map((e) => e.map((t) => encode(t.action))),
    labels: episodes.map((e) => e.map((t) => (sp.discrete ? (t.action as number) : 0))),
    returnsToGo: episodes.map((e) => {
      const out = new Array<number>(e.length).fill(0)
      let g = 0
      for (let i = e.length - 1; i >= 0; i--) {
        g += (e[i] as Transition<O, A>).reward
        out[i] = g
      }
      return out
    }),
    offsets,
    total,
  }
}

/** One step of a decision transformer's context: features, encoded action, return-to-go and timestep. */
export interface ContextStep {
  readonly state: readonly number[]
  readonly action: readonly number[]
  readonly rtg: number
  readonly t: number
}

// Windows of the last K steps (left-padded) as tensors.
function windows(
  steps: readonly (readonly ContextStep[])[],
  sp: OfflineSpaces,
  K: number,
  scale: number,
  maxT: number,
): Window {
  const B = steps.length
  const rtg = new Float64Array(B * K)
  const states = new Float64Array(B * K * sp.obsSize)
  const actions = new Float64Array(B * K * sp.actionSize)
  const timesteps = new Float64Array(B * K)
  const valid = new Float64Array(B * K)
  for (const [b, seq] of steps.entries()) {
    const pad = K - seq.length
    for (const [i, st] of seq.entries()) {
      const j = b * K + pad + i
      rtg[j] = st.rtg / scale
      timesteps[j] = Math.min(maxT - 1, st.t)
      valid[j] = 1
      states.set(st.state.slice(0, sp.obsSize), j * sp.obsSize)
      actions.set(st.action.slice(0, sp.actionSize), j * sp.actionSize)
    }
  }
  return {
    rtg: fromStorage(rtg, [B, K], 'float64'),
    states: fromStorage(states, [B, K, sp.obsSize], 'float64'),
    actions: fromStorage(actions, [B, K, sp.actionSize], 'float64'),
    timesteps: fromStorage(timesteps, [B, K], 'float64'),
    valid: fromStorage(valid, [B, K], 'float64'),
  }
}

function locate(data: DecisionTransformerData, g: number): readonly [number, number] {
  let lo = 0
  let hi = data.offsets.length - 1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if ((data.offsets[mid] ?? 0) <= g) lo = mid
    else hi = mid - 1
  }
  return [lo, g - (data.offsets[lo] ?? 0)]
}

// ------------------------------------------------------------------------------------------------- the learner

/**
 * The decision transformer (Chen et al. 2021) as an offline learner: a causal transformer over the interleaved
 * tokens `(R_t, s_t, a_t)` of the last `K` steps (returns-to-go `R_t = sum_{t' >= t} r_t'`, linear token embeddings
 * plus a learned timestep embedding, pre-norm blocks of multi-head causal self-attention and a GELU MLP), trained to
 * predict each `a_t` from the output at `s_t` (cross-entropy for discrete actions, squared error for continuous
 * ones) on windows sampled uniformly over the dataset's timesteps (left-padded at episode starts, padding masked).
 * Its `greedy` acts from a one-step context; {@link decisionTransformerAgent} keeps the episode's context.
 * Dropout and learning-rate warm-up of the paper are left out.
 *
 * @param env - The environment's spaces.
 * @param options - Context length, model sizes, optimiser and return conditioning.
 * @returns The learner.
 * @concept offline-rl
 * @example
 * ```ts
 * import { decisionTransformerLearner } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * decisionTransformerLearner(cartPole.create(), { context: 4 }).config // => { context: 4, embedding: 32, heads: 2, layers: 1 }
 * ```
 */
export function decisionTransformerLearner<O, A>(
  env: Spaces,
  options: DecisionTransformerOptions = {},
): OfflineLearner<DecisionTransformerState, O, A> {
  const sp = offlineSpaces(env)
  const K = options.context ?? 8
  const h = options.embedding ?? 32
  const heads = options.heads ?? 2
  const layers = options.layers ?? 1
  const opt = optimiserOf(options, 1e-3)
  const modelFor = (maxT: number): Model => buildModel(sp, K, h, heads, layers, maxT, options.key ?? makeKey(0))
  let cached: { readonly maxT: number; readonly model: Model } | undefined
  const model = (maxT: number): Model => {
    if (cached === undefined || cached.maxT !== maxT) cached = { maxT, model: modelFor(maxT) }
    return cached.model
  }
  return {
    name: 'decisionTransformer',
    config: { context: K, embedding: h, heads, layers },
    metrics: { loss: { description: 'action prediction loss of the last minibatch', direction: 'lower' } },
    init: (_k, dataset) => {
      const data = trajectories(sp, dataset)
      const returns = data.returnsToGo.map((r) => r[0] ?? 0)
      let best = -Infinity
      let largest = 0
      let longest = 1
      for (const [i, g] of returns.entries()) {
        best = Math.max(best, g)
        largest = Math.max(largest, Math.abs(g))
        longest = Math.max(longest, data.states[i]?.length ?? 0)
      }
      const maxT = options.maxTimestep ?? longest
      const params = model(maxT).params
      return {
        params,
        optState: opt.init(params),
        updates: 0,
        loss: 0,
        data,
        targetReturn: options.targetReturn ?? (Number.isFinite(best) ? best : 0),
        returnScale: options.returnScale ?? (largest > 0 ? largest : 1),
        maxTimestep: maxT,
      }
    },
    update: (s, batch, k) => {
      const B = batch.length
      const d = s.data
      const seqs: ContextStep[][] = []
      const labels: number[] = []
      const ends = randint(foldIn(k, 0), [B], 0, d.total).data
      for (const g of ends) {
        const [e, t] = locate(d, g)
        const seq: ContextStep[] = []
        for (let i = Math.max(0, t - K + 1); i <= t; i++)
          seq.push({
            state: d.states[e]?.[i] ?? [],
            action: d.actions[e]?.[i] ?? [],
            rtg: d.returnsToGo[e]?.[i] ?? 0,
            t: i,
          })
        seqs.push(seq)
        const pad = K - seq.length
        for (let j = 0; j < K; j++) labels.push(j < pad ? 0 : (d.labels[e]?.[t - K + 1 + j] ?? 0))
      }
      const w = windows(seqs, sp, K, s.returnScale, s.maxTimestep)
      const m = model(s.maxTimestep)
      const valid = reshape(w.valid, [B * K])
      const count = Math.max(
        1,
        toFlat(valid).reduce((a, b) => a + b, 0),
      )
      const loss = (p: ParamTree): readonly [Value, Value] => {
        const pred = m.apply(p, w)
        const perStep = sp.discrete
          ? crossEntropy(reshape(pred, [B * K, sp.actionSize]), vectorTensor(labels), { reduction: 'none' })
          : sum(reshape(square(sub(pred, w.actions)), [B * K, sp.actionSize]), -1)
        const l = div(sum(mul(perStep, valid)), count)
        return [l, reshape(l, [1])]
      }
      const step = gradientStepWithTerms(loss, s.params, opt, s.optState)
      return { ...s, params: step.params, optState: step.optState, updates: s.updates + 1, loss: step.loss }
    },
    stats: (s) => ({ loss: s.loss }),
    greedy: (s, o) => predict(model(s.maxTimestep), sp, s, [], o, s.targetReturn, 0, K),
  }
}

function predict<A>(
  model: Model,
  sp: OfflineSpaces,
  s: DecisionTransformerState,
  history: readonly ContextStep[],
  observation: unknown,
  rtg: number,
  t: number,
  K: number,
  key?: Key,
): A {
  const current: ContextStep = {
    state: sp.features(observation),
    action: new Array<number>(sp.actionSize).fill(0),
    rtg,
    t,
  }
  const seq = [...history.slice(-(K - 1)), current]
  const out = model.apply(s.params, windows([seq], sp, K, s.returnScale, s.maxTimestep))
  const last = toFlat(reshape(sliceAxis(out, 1, K - 1, K), [sp.actionSize]))
  if (!sp.discrete) return last.map((v, i) => Math.min(sp.high[i] ?? v, Math.max(sp.low[i] ?? v, v))) as A
  if (key === undefined) return argmaxOf(last) as A
  return toFlat(categoricalPolicy(vectorTensor(last)).sample(key))[0] as A
}

// -------------------------------------------------------------------------------------------- acting in context

/** The state of {@link decisionTransformerAgent}: the trained model and the current episode's context. */
export interface DecisionTransformerAgentState {
  readonly model: DecisionTransformerState
  readonly history: readonly ContextStep[]
  readonly returnToGo: number
  readonly t: number
}

/**
 * A trained decision transformer as an {@link Agent} that keeps the episode's context: `act` and `greedy` predict the
 * next action from the last `K - 1` steps and the current observation, conditioned on the remaining target return;
 * `learn` appends the step and subtracts its reward from the return-to-go (no parameters change), restarting the
 * context at episode ends. Run it with `trainAgent` (or {@link evaluateInContext}) to evaluate it.
 *
 * @param env - The environment's spaces.
 * @param trained - The trained learner state.
 * @param options - The options it was trained with, a target return override, and whether `act` samples discrete
 *   actions (default false: the most likely action).
 * @returns The agent.
 * @concept offline-rl
 * @example
 * ```ts
 * import { decisionTransformerAgent, decisionTransformerLearner } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = cartPole.create()
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: true, truncated: false }
 * const s = decisionTransformerLearner(env, { context: 2, embedding: 4 }).init(key(0), [t])
 * decisionTransformerAgent(env, s, { context: 2, embedding: 4 }).init(key(0)).returnToGo // => 1
 * ```
 */
export function decisionTransformerAgent<O, A>(
  env: Spaces,
  trained: DecisionTransformerState,
  options: DecisionTransformerOptions & { readonly sample?: boolean } = {},
): Agent<DecisionTransformerAgentState, O, A> {
  const sp = offlineSpaces(env)
  const K = options.context ?? 8
  const model = buildModel(
    sp,
    K,
    options.embedding ?? 32,
    options.heads ?? 2,
    options.layers ?? 1,
    trained.maxTimestep,
    options.key ?? makeKey(0),
  )
  const target = options.targetReturn ?? trained.targetReturn
  const fresh = (): DecisionTransformerAgentState => ({ model: trained, history: [], returnToGo: target, t: 0 })
  const run = (s: DecisionTransformerAgentState, o: O, k?: Key): A =>
    predict<A>(model, sp, s.model, s.history, o, s.returnToGo, s.t, K, k)
  return {
    name: 'decisionTransformer',
    config: { context: K, targetReturn: target } as Json,
    init: fresh,
    act: (s, o, k) => run(s, o, options.sample === true && sp.discrete ? k : undefined),
    greedy: (s, o) => run(s, o),
    learn: (s, t) => {
      if (t.terminated || t.truncated) return fresh()
      const encoded = sp.discrete
        ? Array.from({ length: sp.actionSize }, (_, i) => (i === (t.action as number) ? 1 : 0))
        : [...(t.action as readonly number[])]
      const step: ContextStep = { state: sp.features(t.observation), action: encoded, rtg: s.returnToGo, t: s.t }
      return { ...s, history: [...s.history, step].slice(-(K - 1)), returnToGo: s.returnToGo - t.reward, t: s.t + 1 }
    },
  }
}

/** The outcome of {@link decisionTransformer}. */
export interface DecisionTransformerResult<O, A> {
  readonly state: DecisionTransformerState
  /** The trained model acting in context. */
  readonly agent: Agent<DecisionTransformerAgentState, O, A>
  readonly steps: number
}

/**
 * Train a decision transformer on a dataset ({@link decisionTransformerLearner} run by `trainOffline`) and wrap it
 * as an in-context agent ({@link decisionTransformerAgent}).
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset (whole episodes, in order).
 * @param options - The model's options, steps, batch size and run options.
 * @returns The trained state and agent.
 * @concept offline-rl
 * @example
 * ```ts
 * import { decisionTransformer } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: true, truncated: false }
 * decisionTransformer(cartPole.create(), [t], { context: 2, embedding: 4, steps: 2 }).steps // => 2
 * ```
 */
export function decisionTransformer<O, A>(
  env: Spaces,
  dataset: TransitionDataset<O, A>,
  options: DecisionTransformerOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): DecisionTransformerResult<O, A> {
  const r = trainOffline(decisionTransformerLearner<O, A>(env, options), dataset, options)
  return { state: r.state, agent: decisionTransformerAgent<O, A>(env, r.state, options), steps: r.steps }
}

/**
 * Evaluate an in-context agent (such as a decision transformer) over whole episodes with `trainAgent`, whose
 * `learn` calls carry the context forward.
 *
 * @param env - The environment.
 * @param agent - The agent.
 * @param options - Episodes (default 10) and the seed.
 * @returns The episode returns and their mean.
 * @concept offline-rl
 * @example
 * ```ts
 * import { evaluateInContext } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * import { make, policyAgent } from 'mdml-engine/rl'
 * evaluateInContext(make(cartPole), policyAgent(() => 0), { episodes: 2 }).returns.length // => 2
 * ```
 */
export function evaluateInContext<G, S, O, A>(
  env: Env<S, O, A>,
  agent: Agent<G, O, A>,
  options: { readonly episodes?: number; readonly seed?: number } = {},
): { readonly returns: readonly number[]; readonly meanReturn: number } {
  const episodes = options.episodes ?? 10
  const r = trainAgent(env, agent, { episodes, seed: options.seed ?? 0, checkpoints: false })
  const returns = r.state.returns
  let total = 0
  for (const g of returns) total += g
  return { returns, meanReturn: returns.length === 0 ? 0 : total / returns.length }
}
