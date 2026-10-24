// MuZero-style learned models (Schrittwieser et al. 2020): representation, dynamics and prediction networks, the
// learned model as an environment for lib/rl `mcts`, n-step targets and the unrolled loss. Private to
// methods/rl/agents; the agent is `muZero` in model-based.ts.
import { stopGradient } from '../../../lib/autodiff/index.ts'
import { crossEntropy } from '../../../lib/losses/index.ts'
import { mlp, ReLU, type Sequential } from '../../../lib/nn/index.ts'
import type { Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key } from '../../../lib/random/index.ts'
import { type Env, mcts } from '../../../lib/rl/index.ts'
import { scene } from '../../../lib/scene/index.ts'
import { box, discrete } from '../../../lib/space/index.ts'
import {
  add,
  concat,
  div,
  max,
  mean,
  min,
  mul,
  oneHot,
  sliceAxis,
  softmax,
  square,
  squeeze,
  sub,
  sum,
  toFlat,
  type Value,
} from '../../../lib/tensor/index.ts'
import { rowsTensor, vectorTensor } from './_common.ts'

/** The three networks of a MuZero agent (structure only). */
export interface MuZeroNets {
  readonly representation: Sequential
  readonly dynamics: Sequential
  readonly prediction: Sequential
  readonly latent: number
  readonly nActions: number
  readonly observationSize: number
}

/** The parameters of the three networks. */
export interface MuZeroParams {
  readonly representation: Tree
  readonly dynamics: Tree
  readonly prediction: Tree
}

const net = (widths: readonly number[], key: Key): Sequential => mlp(widths, { activation: () => new ReLU(), key })

/** Build the networks: `h(o) -> s`, `g(s, a) -> (r, s')`, `f(s) -> (policy logits, v)`. */
export function muZeroNets(
  observationSize: number,
  nActions: number,
  latent: number,
  hidden: number,
  key: Key,
): MuZeroNets {
  return {
    representation: net([observationSize, hidden, latent], foldIn(key, 0)),
    dynamics: net([latent + nActions, hidden, latent + 1], foldIn(key, 1)),
    prediction: net([latent, hidden, nActions + 1], foldIn(key, 2)),
    latent,
    nActions,
    observationSize,
  }
}

/** The initial parameters. */
export function muZeroParams(nets: MuZeroNets): MuZeroParams {
  return {
    representation: nets.representation.params,
    dynamics: nets.dynamics.params,
    prediction: nets.prediction.params,
  }
}

// MuZero scales hidden states to [0, 1] (min-max per state) after the representation and every dynamics step.
const scale = (s: Value): Value => {
  const lo = min(s, -1, true)
  const hi = max(s, -1, true)
  return div(sub(s, lo), add(sub(hi, lo), 1e-6))
}

const apply = (m: Sequential, p: Tree, x: Value): Value => m.apply(p as never, x)

/** The hidden state of observations `[B, d]`. */
export function represent(nets: MuZeroNets, p: MuZeroParams, obs: Value): Value {
  return scale(apply(nets.representation, p.representation, obs))
}

/** One step of the learned dynamics from hidden states `[B, S]` and one-hot actions: next states and rewards `[B]`. */
export function dynamicsStep(nets: MuZeroNets, p: MuZeroParams, s: Value, a: Value): { state: Value; reward: Value } {
  const out = apply(nets.dynamics, p.dynamics, concat([s, a], -1))
  return {
    state: scale(sliceAxis(out, -1, 0, nets.latent)),
    reward: squeeze(sliceAxis(out, -1, nets.latent, nets.latent + 1), -1),
  }
}

/** Policy logits `[B, nActions]` and values `[B]` of hidden states. */
export function predict(nets: MuZeroNets, p: MuZeroParams, s: Value): { logits: Value; value: Value } {
  const out = apply(nets.prediction, p.prediction, s)
  return {
    logits: sliceAxis(out, -1, 0, nets.nActions),
    value: squeeze(sliceAxis(out, -1, nets.nActions, nets.nActions + 1), -1),
  }
}

/** The learned model as an environment over hidden states (deterministic, never terminating), for `mcts`. */
export function latentEnv(nets: MuZeroNets, p: MuZeroParams): Env<readonly number[], readonly number[], number> {
  return {
    spec: { id: 'MuZeroLatent-v0', description: 'a learned latent model', deterministic: true },
    observationSpace: box(0, 1, [nets.latent]),
    actionSpace: discrete(nets.nActions),
    reset: () => {
      const s = new Array<number>(nets.latent).fill(0)
      return { state: s, observation: s, info: {} }
    },
    step: (s, a) => {
      const r = dynamicsStep(nets, p, rowsTensor([s], nets.latent), oneHot(vectorTensor([a]), nets.nActions))
      const next = toFlat(r.state)
      return {
        state: next,
        observation: next,
        reward: toFlat(r.reward)[0] ?? 0,
        terminated: false,
        truncated: false,
        info: {},
      }
    },
    render: () => scene(200, 40, [], { title: 'MuZero latent state' }),
    observe: (s) => s,
  }
}

/** The result of a search from an observation: root visit counts, mean values and the root value. */
export interface SearchResult {
  readonly visits: readonly number[]
  readonly values: readonly number[]
  readonly rootValue: number
}

/**
 * Search the learned model with PUCT (lib/rl `mcts` with the prediction network's prior and value at the leaves)
 * from the hidden state of an observation.
 */
export function searchFrom(
  nets: MuZeroNets,
  p: MuZeroParams,
  observation: readonly number[],
  options: {
    readonly simulations: number
    readonly c: number
    readonly gamma: number
    readonly depth: number
    readonly key: Key
  },
): SearchResult {
  const root = toFlat(represent(nets, p, rowsTensor([observation], nets.observationSize)))
  const evaluate = (s: readonly number[]): { prior: number[]; value: number } => {
    const out = predict(nets, p, rowsTensor([s], nets.latent))
    return { prior: toFlat(softmax(out.logits, -1)), value: toFlat(out.value)[0] ?? 0 }
  }
  const cache = new Map<string, { prior: number[]; value: number }>()
  const cached = (s: readonly number[]): { prior: number[]; value: number } => {
    const k = s.join(',')
    let e = cache.get(k)
    if (e === undefined) {
      e = evaluate(s)
      cache.set(k, e)
    }
    return e
  }
  const r = mcts(latentEnv(nets, p), root, {
    simulations: options.simulations,
    c: options.c,
    gamma: options.gamma,
    depth: options.depth,
    prior: (s) => cached(s).prior,
    value: (s) => cached(s).value,
    stateKey: (s) => s.join(','),
    key: options.key,
  })
  let total = 0
  let weighted = 0
  for (const [i, n] of r.visits.entries()) {
    total += n
    weighted += n * (r.values[i] ?? 0)
  }
  return { visits: r.visits, values: r.values, rootValue: total > 0 ? weighted / total : cached(root).value }
}

/** One stored step of an episode: what was seen and done, and the search's policy and value. */
export interface MuZeroStep {
  readonly observation: readonly number[]
  readonly action: number
  readonly reward: number
  /** Normalised root visit counts. */
  readonly policy: readonly number[]
  readonly rootValue: number
}

/** A finished episode with the value to bootstrap from after its end (0 when it terminated). */
export interface MuZeroEpisode {
  readonly steps: readonly MuZeroStep[]
  readonly final: number
  readonly terminated: boolean
}

/** Training targets for `unroll` steps from position `t`: values (n-step), rewards and policies, with masks. */
export interface UnrollTargets {
  readonly actions: readonly number[]
  readonly values: readonly number[]
  readonly valueMask: readonly number[]
  readonly rewards: readonly number[]
  readonly policies: readonly (readonly number[])[]
  readonly policyMask: readonly number[]
}

/**
 * MuZero's targets from position `t`: for k = 0..K the n-step value `sum_{j<m} gamma^j r_{i+j} + gamma^m v_{i+m}`
 * (`i = t + k`, `m = min(n, T - i)`, bootstrapping from the stored root values, or the episode's final value at its
 * end), the observed reward of the action at `i - 1`, and the search policy at `i`; past the end of an episode the
 * policy is masked, rewards are 0, and values are 0 for terminated episodes (masked for truncated ones).
 */
export function unrollTargets(
  episode: MuZeroEpisode,
  t: number,
  unroll: number,
  nStep: number,
  gamma: number,
  nActions: number,
): UnrollTargets {
  const T = episode.steps.length
  const at = (i: number): MuZeroStep | undefined => episode.steps[i]
  const values: number[] = []
  const valueMask: number[] = []
  const rewards: number[] = []
  const policies: number[][] = []
  const policyMask: number[] = []
  const actions: number[] = []
  for (let k = 0; k <= unroll; k++) {
    const i = t + k
    if (i < T) {
      const m = Math.min(nStep, T - i)
      let z = 0
      for (let j = 0; j < m; j++) z += gamma ** j * (at(i + j)?.reward ?? 0)
      z += gamma ** m * (i + m < T ? (at(i + m)?.rootValue ?? 0) : episode.final)
      values.push(z)
      valueMask.push(1)
      policies.push([...(at(i)?.policy ?? [])])
      policyMask.push(1)
    } else {
      values.push(0)
      valueMask.push(episode.terminated ? 1 : 0)
      policies.push(new Array<number>(nActions).fill(1 / nActions))
      policyMask.push(0)
    }
    if (k > 0) rewards.push(i - 1 < T ? (at(i - 1)?.reward ?? 0) : 0)
    if (k < unroll) actions.push(i < T ? (at(i)?.action ?? 0) : 0)
  }
  return { actions, values, valueMask, rewards, policies, policyMask }
}

/**
 * The unrolled loss of a batch: from `h(o_t)`, K dynamics steps along the stored actions; at each step the value
 * error and the policy cross-entropy against the targets (masked), and for k >= 1 the reward error; the gradient into
 * the dynamics is halved at every step (MuZero's gradient scaling) and the K + 1 terms are averaged.
 */
export function unrollLoss(
  nets: MuZeroNets,
  p: MuZeroParams,
  observations: readonly (readonly number[])[],
  targets: readonly UnrollTargets[],
  unroll: number,
): Value {
  let s = represent(nets, p, rowsTensor(observations, nets.observationSize))
  let total: Value = vectorTensor([0])
  const col = (f: (u: UnrollTargets) => number): Value => vectorTensor(targets.map(f))
  for (let k = 0; k <= unroll; k++) {
    const out = predict(nets, p, s)
    const vMask = col((u) => u.valueMask[k] ?? 0)
    const pMask = col((u) => u.policyMask[k] ?? 0)
    const valueLoss = mean(
      mul(
        square(
          sub(
            out.value,
            col((u) => u.values[k] ?? 0),
          ),
        ),
        vMask,
      ),
    )
    const pol = rowsTensor(
      targets.map((u) => u.policies[k] ?? []),
      nets.nActions,
    )
    const ce = crossEntropy(out.logits, pol, { reduction: 'none' })
    let term = add(valueLoss, mean(mul(ce, pMask)))
    if (k < unroll) {
      const a = oneHot(
        col((u) => u.actions[k] ?? 0),
        nets.nActions,
      )
      const step = dynamicsStep(nets, p, s, a)
      const rewardLoss = mean(
        square(
          sub(
            step.reward,
            col((u) => u.rewards[k] ?? 0),
          ),
        ),
      )
      term = add(term, rewardLoss)
      s = add(mul(step.state, 0.5), mul(stopGradient(step.state), 0.5))
    }
    total = add(total, term)
  }
  return div(sum(total), unroll + 1)
}
