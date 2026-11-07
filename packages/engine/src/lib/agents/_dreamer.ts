// A small Dreamer-style latent world model (Hafner et al. 2020; KL balancing from DreamerV2): networks, the filtering
// pass and its ELBO, and imagined rollouts with lambda-returns for the actor and critic. Private to
// methods/rl/agents; the agent is `dreamer` in model-based.ts.
import { stopGradient } from '../autodiff/index.ts'
import { binaryCrossEntropyWithLogits } from '../losses/index.ts'
import { mlp, type Sequential, SiLU } from '../nn/index.ts'
import type { ParamTree } from '../optim/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { categoricalPolicy, gaussianPolicy, type PolicyHead } from '../rl/index.ts'
import {
  add,
  concat,
  div,
  log,
  maximum,
  mean,
  mul,
  oneHot,
  sigmoid,
  sliceAxis,
  softplus,
  square,
  squeeze,
  sub,
  sum,
  tanh,
  type Value,
  zeros,
} from '../tensor/index.ts'

/** Sizes of the latent world model. */
export interface DreamerSizes {
  /** Observation embedding (default 16). */
  readonly embed: number
  /** Deterministic state `h` (default 16). */
  readonly deter: number
  /** Stochastic state `z` (default 4). */
  readonly stoch: number
  /** Hidden width of every MLP (default 32). */
  readonly hidden: number
}

/** The networks (structure only; parameters live in the agent state). */
export interface DreamerNets {
  readonly sizes: DreamerSizes
  readonly observationSize: number
  readonly actionSize: number
  readonly discrete: boolean
  readonly low: readonly number[]
  readonly high: readonly number[]
  readonly world: Readonly<Record<WorldPart, Sequential>>
  readonly actor: Sequential
  readonly critic: Sequential
}

type WorldPart = 'encoder' | 'transition' | 'prior' | 'posterior' | 'decoder' | 'reward' | 'continuation'

/** The parameters of the world model, actor and critic. */
export interface DreamerParams {
  readonly world: Readonly<Record<WorldPart, ParamTree>>
  readonly actor: ParamTree
  readonly critic: ParamTree
}

const net = (widths: readonly number[], key: Key): Sequential => mlp(widths, { activation: () => new SiLU(), key })

/** Build the networks of a world model for an observation size and an action encoding. */
export function dreamerNets(
  observationSize: number,
  action: { readonly size: number; readonly discrete: boolean; readonly low: number[]; readonly high: number[] },
  sizes: DreamerSizes,
  key: Key,
): DreamerNets {
  const { embed, deter, stoch, hidden } = sizes
  const feat = deter + stoch
  return {
    sizes,
    observationSize,
    actionSize: action.size,
    discrete: action.discrete,
    low: action.low,
    high: action.high,
    world: {
      encoder: net([observationSize, hidden, embed], foldIn(key, 0)),
      transition: net([deter + stoch + action.size, hidden, deter], foldIn(key, 1)),
      prior: net([deter, hidden, 2 * stoch], foldIn(key, 2)),
      posterior: net([deter + embed, hidden, 2 * stoch], foldIn(key, 3)),
      decoder: net([feat, hidden, observationSize], foldIn(key, 4)),
      reward: net([feat, hidden, 1], foldIn(key, 5)),
      continuation: net([feat, hidden, 1], foldIn(key, 6)),
    },
    actor: net([feat, hidden, action.discrete ? action.size : 2 * action.size], foldIn(key, 7)),
    critic: net([feat, hidden, 1], foldIn(key, 8)),
  }
}

/** The initial parameters of the networks. */
export function dreamerParams(nets: DreamerNets): DreamerParams {
  const world = Object.fromEntries(Object.entries(nets.world).map(([k, m]) => [k, m.params])) as Record<
    WorldPart,
    ParamTree
  >
  return { world, actor: nets.actor.params, critic: nets.critic.params }
}

const apply = (m: Sequential, p: ParamTree, x: Value): Value => m.apply(p as never, x)
const scalarHead = (m: Sequential, p: ParamTree, x: Value): Value => squeeze(apply(m, p, x), -1)

/** A diagonal Gaussian from a network output `[mean, raw std]`, with the std bounded below by 0.1. */
function gaussian(raw: Value, n: number): { readonly mean: Value; readonly std: Value } {
  return { mean: sliceAxis(raw, -1, 0, n), std: add(softplus(sliceAxis(raw, -1, n, 2 * n)), 0.1) }
}

/** `KL(q || p)` of diagonal Gaussians, summed over the last axis. */
function klGaussian(q: { mean: Value; std: Value }, p: { mean: Value; std: Value }): Value {
  const ratio = add(div(add(square(q.std), square(sub(q.mean, p.mean))), mul(square(p.std), 2)), -0.5)
  return sum(add(sub(log(p.std), log(q.std)), ratio), -1)
}

const detach = (g: { mean: Value; std: Value }): { mean: Value; std: Value } => ({
  mean: stopGradient(g.mean),
  std: stopGradient(g.std),
})

/** One step of the deterministic state: `h' = tanh(MLP([h, z, a]))` (an MLP in place of Dreamer's GRU). */
export function transition(nets: DreamerNets, world: DreamerParams['world'], h: Value, z: Value, a: Value): Value {
  return tanh(apply(nets.world.transition, world.transition, concat([h, z, a], -1)))
}

/** The prior `p(z | h)` and the posterior `q(z | h, o)`. */
export function prior(nets: DreamerNets, world: DreamerParams['world'], h: Value): { mean: Value; std: Value } {
  return gaussian(apply(nets.world.prior, world.prior, h), nets.sizes.stoch)
}
export function posterior(
  nets: DreamerNets,
  world: DreamerParams['world'],
  h: Value,
  obs: Value,
): { mean: Value; std: Value } {
  const e = apply(nets.world.encoder, world.encoder, obs)
  return gaussian(apply(nets.world.posterior, world.posterior, concat([h, e], -1)), nets.sizes.stoch)
}

/** A batch of sequences, time-major: observations `[B, d]`, encoded actions `[B, A]`, rewards and continuations `[B]`. */
export interface SequenceBatch {
  readonly observations: readonly Value[]
  readonly actions: readonly Value[]
  readonly rewards: readonly Value[]
  readonly continuations: readonly Value[]
}

/** The terms of the world model's loss and the filtered states. */
export interface WorldPass {
  readonly loss: Value
  readonly reconstruction: Value
  readonly reward: Value
  readonly continuation: Value
  readonly kl: Value
  readonly h: readonly Value[]
  readonly z: readonly Value[]
}

/**
 * Filter a batch of sequences through the world model and score it: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` from zeros,
 * `z_t ~ q(z | h_t, o_t)` by reparameterisation, then reconstruction, reward and continuation losses (the reward and
 * continuation of the transition into step t are predicted at step t) and the KL-balanced regulariser
 * `alpha KL(sg(q) || p) + (1 - alpha) KL(q || sg(p))` with free nats.
 */
export function worldPass(
  nets: DreamerNets,
  world: DreamerParams['world'],
  batch: SequenceBatch,
  key: Key,
  options: { readonly klBalance: number; readonly freeNats: number; readonly klScale: number },
): WorldPass {
  const B = batch.observations[0]?.shape[0] ?? 1
  const { deter, stoch } = nets.sizes
  let h: Value = zeros([B, deter])
  let z: Value = zeros([B, stoch])
  let a: Value = zeros([B, nets.actionSize])
  const hs: Value[] = []
  const zs: Value[] = []
  let recon: Value = zeros([])
  let rew: Value = zeros([])
  let cont: Value = zeros([])
  let kl: Value = zeros([])
  const L = batch.observations.length
  for (let t = 0; t < L; t++) {
    const obs = batch.observations[t] as Value
    h = transition(nets, world, h, z, a)
    const p = prior(nets, world, h)
    const q = posterior(nets, world, h, obs)
    z = add(q.mean, mul(q.std, normal(foldIn(key, t), [B, stoch])))
    const feat = concat([h, z], -1)
    recon = add(recon, mean(mul(sum(square(sub(apply(nets.world.decoder, world.decoder, feat), obs)), -1), 0.5)))
    if (t > 0) {
      rew = add(
        rew,
        mean(mul(square(sub(scalarHead(nets.world.reward, world.reward, feat), batch.rewards[t - 1] as Value)), 0.5)),
      )
      cont = add(
        cont,
        binaryCrossEntropyWithLogits(
          scalarHead(nets.world.continuation, world.continuation, feat),
          batch.continuations[t - 1] as Value,
        ),
      )
    }
    const balanced = add(
      mul(mean(klGaussian(detach(q), p)), options.klBalance),
      mul(mean(klGaussian(q, detach(p))), 1 - options.klBalance),
    )
    kl = add(kl, maximum(balanced, options.freeNats))
    hs.push(h)
    zs.push(z)
    a = batch.actions[t] as Value
  }
  const steps = Math.max(1, L - 1)
  const reconstruction = div(recon, L)
  const reward = div(rew, steps)
  const continuation = div(cont, steps)
  const klMean = div(kl, L)
  const loss = add(add(add(reconstruction, reward), continuation), mul(klMean, options.klScale))
  return { loss, reconstruction, reward, continuation, kl: klMean, h: hs, z: zs }
}

/** The actor's policy head at latent features. */
export function actorHead(nets: DreamerNets, actor: ParamTree, feat: Value): PolicyHead {
  const out = apply(nets.actor, actor, feat)
  if (nets.discrete) return categoricalPolicy(out)
  const n = nets.actionSize
  const logStd = log(add(softplus(sliceAxis(out, -1, n, 2 * n)), 0.05))
  return gaussianPolicy(sliceAxis(out, -1, 0, n), logStd, { squash: true, low: [...nets.low], high: [...nets.high] })
}

/** An imagined rollout from given latent states. */
export interface Imagination {
  /** Features `[N, D + Z]` at steps 0..H (H + 1 of them). */
  readonly features: readonly Value[]
  /** Actions taken at steps 0..H-1 (encoded for the transition). */
  readonly actions: readonly Value[]
  readonly rewards: readonly Value[]
  /** Discounts `gamma c_k` predicted for each step. */
  readonly discounts: readonly Value[]
  readonly values: readonly Value[]
  /** Lambda-returns for steps 0..H-1. */
  readonly returns: readonly Value[]
  readonly entropy: Value
}

/**
 * Roll the world model forward from latent states with the actor (reparameterised continuous actions, sampled
 * discrete ones), predict rewards and continuations, value the states with the critic, and form lambda-returns
 * `R_k = r_k + gamma c_k ((1 - lambda) v_{k+1} + lambda R_{k+1})`, `R_H = v_H`.
 */
export function imagine(
  nets: DreamerNets,
  params: DreamerParams,
  h0: Value,
  z0: Value,
  horizon: number,
  key: Key,
  options: { readonly gamma: number; readonly lambda: number; readonly actions?: readonly Value[] },
): Imagination {
  const N = h0.shape[0] ?? 1
  let h = h0
  let z = z0
  const features: Value[] = [concat([h, z], -1)]
  const actions: Value[] = []
  const rewards: Value[] = []
  const discounts: Value[] = []
  let entropy: Value = zeros([])
  for (let k = 0; k < horizon; k++) {
    const feat = features[k] as Value
    const head = actorHead(nets, params.actor, feat)
    entropy = add(entropy, mean(head.entropy()))
    let a: Value
    const given = options.actions?.[k]
    if (given !== undefined) a = given
    else if (nets.discrete) {
      const idx = head.sample(foldIn(key, 2 * k))
      a = oneHot(idx, nets.actionSize)
    } else a = head.rsample(foldIn(key, 2 * k))
    actions.push(a)
    h = transition(nets, params.world, h, z, a)
    const p = prior(nets, params.world, h)
    z = add(p.mean, mul(p.std, normal(foldIn(key, 2 * k + 1), [N, nets.sizes.stoch])))
    const next = concat([h, z], -1)
    features.push(next)
    rewards.push(scalarHead(nets.world.reward, params.world.reward, next))
    discounts.push(mul(sigmoid(scalarHead(nets.world.continuation, params.world.continuation, next)), options.gamma))
  }
  const values = features.map((f) => scalarHead(nets.critic, params.critic, f))
  const returns: Value[] = new Array(horizon)
  let next = values[horizon] as Value
  for (let k = horizon - 1; k >= 0; k--) {
    const blend = add(mul(values[k + 1] as Value, 1 - options.lambda), mul(next, options.lambda))
    next = add(rewards[k] as Value, mul(discounts[k] as Value, blend))
    returns[k] = next
  }
  return { features, actions, rewards, discounts, values, returns, entropy: div(entropy, Math.max(1, horizon)) }
}
