/**
 * Distributional value-based agents, which learn the whole distribution of each action's return rather than its mean:
 * C51 (a categorical distribution on fixed atoms, with its projection of the Bellman target), Rainbow (C51 with double
 * Q-learning, duelling heads, prioritised replay, n-step returns and noisy networks), QR-DQN (fixed quantiles trained
 * by the quantile Huber loss) and IQN (implicit quantile networks over sampled fractions, with risk distortions for
 * risk-sensitive policies), plus readers of the learnt distributions for figures.
 */

import { type Json, linspaceNumbers, OmaiError } from '../base/index.ts'
import { ndtr, ndtri } from '../numerics/special/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../random/index.ts'
import type { Agent } from '../rl/index.ts'
import {
  add,
  argmax,
  broadcastTo,
  cos,
  exp,
  fromStorage,
  logSoftmax,
  matmul,
  mean,
  mul,
  neg,
  oneHot,
  relu,
  repeat,
  reshape,
  softmax,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { actionCount, observationSize, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import {
  asFloat,
  distributionalAgent,
  type Head,
  quantileHuberKernel,
  rowsOf,
  scalarTensor,
  selectAtoms,
  selectLast,
} from './_distributional.ts'
import {
  type DqnOptions,
  type DqnState,
  type QNetwork,
  type QParams,
  qLayer,
  qLayerInit,
  qNetwork,
} from './value-based.ts'

// ------------------------------------------------------------------------------------- projection and losses

/**
 * The fixed support of a categorical return distribution: `atoms` evenly spaced values from `vMin` to `vMax` (as
 * `numpy.linspace`).
 *
 * @param vMin - The smallest return.
 * @param vMax - The largest return.
 * @param atoms - The number of atoms (at least 2).
 * @returns The support.
 * @throws OmaiError with code `value` for fewer than two atoms or an empty range.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { categoricalSupport } from 'omai-engine/agents'
 * categoricalSupport(-1, 1, 5) // => [-1, -0.5, 0, 0.5, 1]
 * ```
 */
export function categoricalSupport(vMin: number, vMax: number, atoms: number): number[] {
  if (!(Number.isInteger(atoms) && atoms >= 2)) throw new OmaiError('value', 'categoricalSupport: need two atoms')
  if (!(vMax > vMin)) throw new OmaiError('value', 'categoricalSupport: need vMin < vMax')
  return linspaceNumbers(vMin, vMax, atoms)
}

/**
 * The categorical projection of C51 (Bellemare, Dabney and Munos 2017, Algorithm 1): each atom `z_j` of the next-state
 * distribution moves to `r + d z_j` (clipped to `[vMin, vMax]`) and its probability is split between the two nearest
 * atoms of the fixed support in proportion to closeness, so the projected distribution lives on the same support.
 * The mean moves to `r + d E[Z]` whenever no atom is clipped.
 *
 * @param probabilities - Next-state distributions, one row per sample, on the support of `vMin`, `vMax`.
 * @param rewards - The rewards (n-step sums for n-step targets).
 * @param discounts - The bootstrap discounts (`gamma^n`, or 0 at terminal states).
 * @param vMin - The smallest return of the support.
 * @param vMax - The largest return of the support.
 * @returns The projected distributions.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { categoricalProjection } from 'omai-engine/agents'
 * categoricalProjection([[0, 1, 0]], [0.5], [0.5], -1, 1) // => [[0, 0.5, 0.5]]
 * ```
 */
export function categoricalProjection(
  probabilities: readonly (readonly number[])[],
  rewards: readonly number[],
  discounts: readonly number[],
  vMin: number,
  vMax: number,
): number[][] {
  const n = probabilities[0]?.length ?? 0
  const z = categoricalSupport(vMin, vMax, n)
  const dz = (vMax - vMin) / (n - 1)
  return probabilities.map((p, b) => {
    const m = new Array<number>(n).fill(0)
    const r = rewards[b] ?? 0
    const d = discounts[b] ?? 0
    for (let j = 0; j < n; j++) {
      const tz = Math.min(vMax, Math.max(vMin, r + d * (z[j] ?? 0)))
      const pos = Math.min(n - 1, Math.max(0, (tz - vMin) / dz))
      const l = Math.floor(pos)
      const u = Math.ceil(pos)
      const pj = p[j] ?? 0
      if (l === u) m[l] = (m[l] ?? 0) + pj
      else {
        m[l] = (m[l] ?? 0) + pj * (u - pos)
        m[u] = (m[u] ?? 0) + pj * (pos - l)
      }
    }
    return m
  })
}

/** Options of {@link quantileHuberLoss}. */
export interface QuantileHuberOptions {
  /** The Huber threshold `kappa` (default 1; 0 gives the plain quantile regression loss). */
  readonly kappa?: number
  /** `'mean'` over samples (default), `'sum'`, or `'none'` for one loss per sample. */
  readonly reduction?: 'mean' | 'sum' | 'none'
  /** Per-sample weights `[B]` (importance weights of prioritised replay). */
  readonly weight?: TensorLike
}

/**
 * The quantile Huber loss of QR-DQN and IQN (Dabney et al. 2018, eq. 10; 2018b, eq. 3): with pairwise errors
 * `u_ij = T_j - theta_i` between target samples and predicted quantiles at fractions `tau_i`,
 * `rho(u) = |tau_i - 1{u < 0}| L_kappa(u) / kappa`, summed over `i` and averaged over `j` for each sample.
 *
 * @param quantiles - Predicted quantiles `[B, N]`.
 * @param targets - Target samples `[B, M]` (constants).
 * @param taus - The fractions of the predicted quantiles, `[N]` or `[B, N]`.
 * @param options - `kappa`, the reduction and weights.
 * @returns The loss (or per-sample losses `[B]`), differentiable in the quantiles.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { quantileHuberLoss } from 'omai-engine/agents'
 * import { item } from 'omai-engine/tensor'
 * item(quantileHuberLoss([[0]], [[2]], [0.25], { kappa: 1 })) // => 0.375
 * ```
 */
export function quantileHuberLoss(
  quantiles: TensorLike,
  targets: TensorLike,
  taus: TensorLike,
  options: QuantileHuberOptions = {},
): Value {
  const theta = asFloat(quantiles)
  const t = asFloat(targets)
  const fractions = broadcastTo(asFloat(taus), theta.shape)
  let perSample = quantileHuberKernel(theta, t, fractions, scalarTensor(options.kappa ?? 1))
  if (options.weight !== undefined) perSample = mul(perSample, options.weight)
  const reduction = options.reduction ?? 'mean'
  return reduction === 'none' ? perSample : reduction === 'sum' ? sum(perSample) : mean(perSample)
}

/** A risk distortion of quantile fractions for IQN's risk-sensitive policies (Dabney et al. 2018b, section 3.1). */
export type RiskDistortion = 'neutral' | { readonly kind: 'cvar' | 'wang' | 'cpw' | 'pow'; readonly eta: number }

/**
 * A distortion `beta: [0, 1] -> [0, 1]` of quantile fractions: acting on `E_{tau ~ U}[Z_{beta(tau)}]` gives a
 * distortion risk measure. CVaR (`eta tau`, risk-averse for `eta < 1`), Wang (`Phi(Phi^-1(tau) + eta)`, averse for
 * `eta < 0`), CPW (`tau^eta / (tau^eta + (1 - tau)^eta)^(1/eta)`) and the power family (averse for `eta < 0`).
 *
 * @param distortion - The distortion.
 * @returns The map over arrays of fractions.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { riskDistortion } from 'omai-engine/agents'
 * riskDistortion({ kind: 'cvar', eta: 0.25 })([0.5, 1]) // => [0.125, 0.25]
 * ```
 */
export function riskDistortion(distortion: RiskDistortion): (taus: readonly number[]) => number[] {
  if (distortion === 'neutral') return (taus) => [...taus]
  const eta = distortion.eta
  switch (distortion.kind) {
    case 'cvar':
      return (taus) => taus.map((t) => eta * t)
    case 'wang':
      return (taus) => toFlat(ndtr(add(ndtri(vectorTensor(taus)), eta)))
    case 'cpw':
      return (taus) => taus.map((t) => t ** eta / (t ** eta + (1 - t) ** eta) ** (1 / eta))
    default: {
      const e = 1 / (1 + Math.abs(eta))
      return (taus) => taus.map((t) => (eta >= 0 ? t ** e : 1 - (1 - t) ** e))
    }
  }
}

// --------------------------------------------------------------------------------------------- IQN network

/** Options of {@link implicitQuantileNetwork}. */
export interface ImplicitQuantileNetworkOptions {
  /** Hidden widths of the state embedding `psi` (default `[64]`); the last is the embedding width. */
  readonly hidden?: readonly number[]
  /** Cosine basis size `n` of the fraction embedding `phi` (default 64, as in the paper). */
  readonly embedding?: number
}

/**
 * An implicit quantile network as a pure function of its parameters: `apply(params, x [B, d], taus [B, N])` returns the
 * quantiles `Z_tau(x, a)` as `[B, N, nActions]`.
 *
 * @concept distributional-rl
 */
export interface ImplicitQuantileNetwork {
  readonly params: QParams
  readonly nActions: number
  apply(params: QParams, x: Value, taus: TensorLike): Value
}

/**
 * Build the network of IQN (Dabney et al. 2018b): a state embedding `psi(x)` (an MLP with ReLUs), a fraction embedding
 * `phi_j(tau) = ReLU(sum_{i<n} cos(pi i tau) w_ij + b_j)`, their elementwise product, then a hidden layer and one
 * output per action.
 *
 * @param inputSize - The observation size.
 * @param nActions - The number of actions.
 * @param options - Hidden widths and the cosine basis size.
 * @param key - The initialisation key.
 * @returns The network.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { implicitQuantileNetwork } from 'omai-engine/agents'
 * import { key } from 'omai-engine/random'
 * import { tensor } from 'omai-engine/tensor'
 * const net = implicitQuantileNetwork(3, 2, { hidden: [8], embedding: 4 }, key(0))
 * net.apply(net.params, tensor([[1, 0, 0]]), tensor([[0.1, 0.5, 0.9]])).shape // => [1, 3, 2]
 * ```
 */
export function implicitQuantileNetwork(
  inputSize: number,
  nActions: number,
  options: ImplicitQuantileNetworkOptions,
  key: Key,
): ImplicitQuantileNetwork {
  const hidden = options.hidden ?? [64]
  const n = options.embedding ?? 64
  const params: Record<string, Record<string, Tensor>> = {}
  let width = inputSize
  for (const [i, h] of hidden.entries()) {
    params[`h${i}`] = qLayerInit(foldIn(key, i), width, h, false, 0)
    width = h
  }
  params['embedding'] = qLayerInit(foldIn(key, 100), n, width, false, 0)
  params['merge'] = qLayerInit(foldIn(key, 101), width, width, false, 0)
  params['out'] = qLayerInit(foldIn(key, 102), width, nActions, false, 0)
  const basis = rowsTensor([Array.from({ length: n }, (_, i) => Math.PI * i)], n) // [1, n]
  // Rows are flattened to [B N, .] so that every layer is a plain matrix product.
  const apply = (p: QParams, x: Value, taus: TensorLike): Value => {
    const t = asFloat(taus)
    const [B = 1, N = 1] = t.shape
    let psi = x
    for (let i = 0; i < hidden.length; i++) psi = relu(qLayer(p[`h${i}`] ?? {}, psi))
    const features = cos(matmul(reshape(t, [B * N, 1]), basis)) // [B N, n]
    const phi = relu(qLayer(p['embedding'] ?? {}, features)) // [B N, h]
    const merged = mul(repeat(psi, N, 0), phi)
    const z = qLayer(p['out'] ?? {}, relu(qLayer(p['merge'] ?? {}, merged)))
    return reshape(z, [B, N, nActions])
  }
  return { params, nActions, apply }
}

/** The state of a distributional agent (DQN's state: networks, optimiser state, replay, counters, statistics). */
export type DistributionalState = DqnState

// --------------------------------------------------------------------------------------------------------- C51

/** Options of {@link c51} (and {@link rainbow}). */
export interface C51Options extends DqnOptions {
  /** Atoms of the support (default 51). */
  readonly atoms?: number
  /** The support's range (default -10 to 10). */
  readonly vMin?: number
  readonly vMax?: number
}

function categoricalHead(env: Spaces, options: C51Options): { head: Head; net: QNetwork } {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'c51')
  const atoms = options.atoms ?? 51
  const vMin = options.vMin ?? -10
  const vMax = options.vMax ?? 10
  const support = categoricalSupport(vMin, vMax, atoms)
  const z = vectorTensor(support)
  const net = qNetwork(d, nA, { ...options, atoms }, options.key ?? makeKey(0))
  const noisy = options.noisy ?? false
  const probs = (p: QParams, x: Value, k?: Key): Value => softmax(net.apply(p, x, k), -1)
  const q = (p: QParams, x: Value, k?: Key): Value => matmul(probs(p, x, k), z)
  const head: Head = {
    params: net.params,
    stochastic: noisy,
    q,
    prepare: (online, target, batch, k) => {
      const nk = (j: number): Key | undefined => (noisy ? foldIn(k, j) : undefined)
      const pNext = probs(target, batch.next, nk(0))
      const qNext = options.double === true ? q(online, batch.next, nk(1)) : matmul(pNext, z)
      const best = oneHot(argmax(qNext, -1), nA)
      const m = categoricalProjection(
        rowsOf(selectAtoms(pNext, best), atoms),
        batch.rewards,
        batch.discounts,
        vMin,
        vMax,
      )
      const targetT = rowsTensor(m, atoms)
      const entropy = m.map((row) => {
        let h = 0
        for (const pi of row) if (pi > 0) h -= pi * Math.log(pi)
        return h
      })
      const targetMean = m.map((row) => {
        let s = 0
        for (const [j, pi] of row.entries()) s += pi * (support[j] ?? 0)
        return s
      })
      const noise = nk(2)
      return {
        terms: (p) => {
          const logp = logSoftmax(selectAtoms(net.apply(p, batch.obs, noise), batch.mask), -1)
          return { loss: neg(sum(mul(targetT, logp), -1)), q: matmul(exp(logp), z) }
        },
        targetMean,
        priorityOffset: entropy,
      }
    },
  }
  return { head, net }
}

/**
 * C51 (Bellemare, Dabney and Munos 2017): the network outputs a categorical distribution over `atoms` fixed returns
 * for each action; the target is the {@link categoricalProjection} of `r + gamma^n Z(s', a*)` with
 * `a* = argmax_a E[Z(s', a)]` under the target network (the online one with `double`), and the loss is the
 * cross-entropy to it. Every DQN variant applies (double, duelling, prioritised replay with KL priorities, n-step,
 * noisy networks).
 *
 * @param env - The environment's spaces (vector observations, discrete actions).
 * @param options - DQN's options plus the support.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { c51 } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * c51(cartPole.create(), { atoms: 11, vMin: 0, vMax: 100 }).name // => 'c51'
 * ```
 */
export function c51(env: Spaces, options: C51Options = {}): Agent<DistributionalState, readonly number[], number> {
  const { head } = categoricalHead(env, options)
  return distributionalAgent('c51', env, options, head, {
    atoms: options.atoms ?? 51,
    vMin: options.vMin ?? -10,
    vMax: options.vMax ?? 10,
  })
}

/** Options of {@link rainbow}: C51's, with Rainbow's components on by default. */
export type RainbowOptions = C51Options

/**
 * Rainbow (Hessel et al. 2018): C51 with double Q-learning, duelling heads, prioritised replay (priorities from the KL
 * loss, exponent 0.5, importance exponent annealed from 0.4 to 1), 3-step returns and noisy networks
 * (`sigma0 = 0.5`) instead of epsilon-greedy exploration. Each component can be switched off for ablations.
 *
 * @param env - The environment's spaces.
 * @param options - C51's options; the Rainbow components default on.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { rainbow } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * rainbow(cartPole.create()).config['nStep'] // => 3
 * ```
 */
export function rainbow(
  env: Spaces,
  options: RainbowOptions = {},
): Agent<DistributionalState, readonly number[], number> {
  const full: C51Options = {
    double: true,
    dueling: true,
    noisy: true,
    sigma0: 0.5,
    nStep: 3,
    prioritized: { alpha: 0.5, beta: 0.4 },
    ...options,
  }
  const { head } = categoricalHead(env, full)
  return distributionalAgent('rainbow', env, full, head, {
    atoms: full.atoms ?? 51,
    vMin: full.vMin ?? -10,
    vMax: full.vMax ?? 10,
  })
}

/**
 * The return distributions of a C51 or Rainbow state at an observation (noise-free), for inspection and figures.
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param options - The options the agent was built with.
 * @returns The support and one distribution per action.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { c51, categoricalDistribution } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const env = cartPole.create()
 * const s = c51(env, { hidden: [8], atoms: 5 }).init(key(0))
 * categoricalDistribution(env, s, [0, 0, 0, 0], { hidden: [8], atoms: 5 }).probabilities.length // => 2
 * ```
 */
export function categoricalDistribution(
  env: Spaces,
  state: DistributionalState,
  observation: readonly number[],
  options: C51Options = {},
): { readonly support: number[]; readonly probabilities: number[][] } {
  const atoms = options.atoms ?? 51
  const { net } = categoricalHead(env, options)
  const p = softmax(net.apply(state.params, rowsTensor([observation], observationSize(env.observationSpace))), -1)
  return {
    support: categoricalSupport(options.vMin ?? -10, options.vMax ?? 10, atoms),
    probabilities: rowsOf(p, atoms),
  }
}

// ------------------------------------------------------------------------------------------------------ QR-DQN

/** Options of {@link qrDqn}. */
export interface QrDqnOptions extends DqnOptions {
  /** The number of quantiles (default 200, as in the paper). */
  readonly quantiles?: number
  /** The Huber threshold of the quantile loss (default 1). */
  readonly kappa?: number
}

function quantileNet(env: Spaces, options: QrDqnOptions): QNetwork {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'qrDqn')
  return qNetwork(d, nA, { ...options, atoms: options.quantiles ?? 200 }, options.key ?? makeKey(0))
}

/**
 * QR-DQN (Dabney et al. 2018): the network outputs `N` quantiles of the return per action at the fixed fractions
 * `tau_i = (2i + 1) / 2N`; targets are the `N` samples `r + gamma^n theta_j(s', a*)` with `a*` maximising the mean,
 * and the loss is the {@link quantileHuberLoss}. Every DQN variant applies.
 *
 * @param env - The environment's spaces.
 * @param options - DQN's options plus the number of quantiles and `kappa`.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { qrDqn } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * qrDqn(cartPole.create(), { quantiles: 8 }).config['quantiles'] // => 8
 * ```
 */
export function qrDqn(env: Spaces, options: QrDqnOptions = {}): Agent<DistributionalState, readonly number[], number> {
  const nA = actionCount(env.actionSpace, 'qrDqn')
  const N = options.quantiles ?? 200
  const kappa = options.kappa ?? 1
  const net = quantileNet(env, options)
  const taus = vectorTensor(Array.from({ length: N }, (_, i) => (2 * i + 1) / (2 * N)))
  const noisy = options.noisy ?? false
  const q = (p: QParams, x: Value, k?: Key): Value => mean(net.apply(p, x, k), -1)
  const head: Head = {
    params: net.params,
    stochastic: noisy,
    q,
    prepare: (online, target, batch, k) => {
      const nk = (j: number): Key | undefined => (noisy ? foldIn(k, j) : undefined)
      const zNext = net.apply(target, batch.next, nk(0))
      const qNext = options.double === true ? q(online, batch.next, nk(1)) : mean(zNext, -1)
      const next = rowsOf(selectAtoms(zNext, oneHot(argmax(qNext, -1), nA)), N)
      const targets = next.map((row, b) => row.map((v) => (batch.rewards[b] ?? 0) + (batch.discounts[b] ?? 0) * v))
      const targetT = rowsTensor(targets, N)
      const noise = nk(2)
      return {
        terms: (p) => {
          const chosen = selectAtoms(net.apply(p, batch.obs, noise), batch.mask)
          return { loss: quantileHuberLoss(chosen, targetT, taus, { kappa, reduction: 'none' }), q: mean(chosen, -1) }
        },
        targetMean: targets.map((row) => row.reduce((a, b) => a + b, 0) / N),
      }
    },
  }
  return distributionalAgent('qrDqn', env, options, head, { quantiles: N, kappa })
}

/**
 * The quantiles of a QR-DQN state at an observation, one row of `N` per action.
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param options - The options the agent was built with.
 * @returns The quantiles per action.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { qrDqn, quantileDistribution } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const env = cartPole.create()
 * const s = qrDqn(env, { hidden: [8], quantiles: 4 }).init(key(0))
 * quantileDistribution(env, s, [0, 0, 0, 0], { hidden: [8], quantiles: 4 })[0]?.length // => 4
 * ```
 */
export function quantileDistribution(
  env: Spaces,
  state: DistributionalState,
  observation: readonly number[],
  options: QrDqnOptions = {},
): number[][] {
  const net = quantileNet(env, options)
  return rowsOf(
    net.apply(state.params, rowsTensor([observation], observationSize(env.observationSpace))),
    options.quantiles ?? 200,
  )
}

// --------------------------------------------------------------------------------------------------------- IQN

/** Options of {@link iqn}. */
export interface IqnOptions extends Omit<DqnOptions, 'dueling' | 'noisy' | 'sigma0' | 'atoms'> {
  /** Cosine basis size (default 64). */
  readonly embedding?: number
  /** Fractions sampled for the online quantiles, `N` (default 64). */
  readonly quantiles?: number
  /** Fractions sampled for the target quantiles, `N'` (default 64). */
  readonly targetQuantiles?: number
  /** Fractions sampled to estimate action values when acting, `K` (default 32). */
  readonly policySamples?: number
  /** The Huber threshold (default 1). */
  readonly kappa?: number
  /** The risk distortion of the policy (default `'neutral'`). */
  readonly distortion?: RiskDistortion
}

/**
 * IQN, implicit quantile networks (Dabney et al. 2018b): the network maps a state and a fraction `tau ~ U[0, 1]`
 * (through a cosine embedding) to the `tau`-quantile of each action's return. Training samples `N` online and `N'`
 * target fractions and minimises the {@link quantileHuberLoss} over all pairs; actions maximise
 * `mean_k Z_{beta(tau_k)}(s, a)` over `K` sampled fractions under a {@link riskDistortion} `beta` (risk-neutral by
 * default; CVaR and others give risk-sensitive policies). The greedy policy uses `K` evenly spaced fractions.
 *
 * @param env - The environment's spaces.
 * @param options - DQN's replay and target options, the sample sizes, the embedding and the distortion.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { iqn } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * iqn(cartPole.create(), { distortion: { kind: 'cvar', eta: 0.25 } }).name // => 'iqn'
 * ```
 */
export function iqn(env: Spaces, options: IqnOptions = {}): Agent<DistributionalState, readonly number[], number> {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'iqn')
  const N = options.quantiles ?? 64
  const N2 = options.targetQuantiles ?? 64
  const K = options.policySamples ?? 32
  const kappa = options.kappa ?? 1
  const distortion = options.distortion ?? 'neutral'
  const beta = riskDistortion(distortion)
  const net = implicitQuantileNetwork(
    d,
    nA,
    {
      hidden: options.hidden ?? [64],
      ...(options.embedding === undefined ? {} : { embedding: options.embedding }),
    },
    options.key ?? makeKey(0),
  )
  const grid = beta(Array.from({ length: K }, (_, k) => (k + 0.5) / K))
  const fractions = (rows: number, k?: Key): Tensor => {
    if (k === undefined)
      return rowsTensor(
        Array.from({ length: rows }, () => grid),
        K,
      )
    const u = Array.from(uniform(k, [rows * K]).data)
    return fromStorage(Float64Array.from(beta(u)), [rows, K], 'float64')
  }
  const q = (p: QParams, x: Value, k?: Key): Value => mean(net.apply(p, x, fractions(x.shape[0] ?? 1, k)), 1)
  const head: Head = {
    params: net.params,
    stochastic: true,
    q,
    prepare: (online, target, batch, k) => {
      const B = batch.size
      const best = oneHot(argmax(q(options.double === true ? online : target, batch.next, foldIn(k, 0)), -1), nA)
      const tauNext = uniform(foldIn(k, 1), [B, N2])
      const next = rowsOf(selectLast(net.apply(target, batch.next, tauNext), best), N2)
      const targets = next.map((row, b) => row.map((v) => (batch.rewards[b] ?? 0) + (batch.discounts[b] ?? 0) * v))
      const targetT = rowsTensor(targets, N2)
      const tau = uniform(foldIn(k, 2), [B, N])
      return {
        terms: (p) => {
          const chosen = selectLast(net.apply(p, batch.obs, tau), batch.mask)
          return { loss: quantileHuberLoss(chosen, targetT, tau, { kappa, reduction: 'none' }), q: mean(chosen, -1) }
        },
        targetMean: targets.map((row) => row.reduce((a, b) => a + b, 0) / N2),
      }
    },
  }
  return distributionalAgent('iqn', env, { ...options, hidden: options.hidden ?? [64] }, head, {
    quantiles: N,
    targetQuantiles: N2,
    policySamples: K,
    kappa,
    distortion: distortion as Json,
  })
}

/**
 * The quantiles of an IQN state at an observation for given fractions, one row per action.
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param taus - The fractions.
 * @param options - The options the agent was built with.
 * @returns `Z_tau(s, a)` for each action (rows) and fraction (columns).
 * @concept distributional-rl
 * @example
 * ```ts
 * import { implicitQuantiles, iqn } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const env = cartPole.create()
 * const s = iqn(env, { hidden: [8], embedding: 4 }).init(key(0))
 * implicitQuantiles(env, s, [0, 0, 0, 0], [0.1, 0.9], { hidden: [8], embedding: 4 })[1]?.length // => 2
 * ```
 */
export function implicitQuantiles(
  env: Spaces,
  state: DistributionalState,
  observation: readonly number[],
  taus: readonly number[],
  options: IqnOptions = {},
): number[][] {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'implicitQuantiles')
  const net = implicitQuantileNetwork(
    d,
    nA,
    { hidden: options.hidden ?? [64], ...(options.embedding === undefined ? {} : { embedding: options.embedding }) },
    makeKey(0),
  )
  const z = net.apply(state.params, rowsTensor([observation], d), rowsTensor([taus], taus.length)) // [1, T, nA]
  const flat = toFlat(z)
  return Array.from({ length: nA }, (_, a) => taus.map((_, t) => flat[t * nA + a] ?? 0))
}
