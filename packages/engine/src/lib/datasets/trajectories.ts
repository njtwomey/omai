import { softmaxOf } from '../base/index.ts'
import { type DatasetGenerator, type TrajectoryData, trajectories } from '../data/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { categorical, dirichlet, foldIn, type Key, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { matrix, stamp } from './_util.ts'

const logistic = (z: number): number => 1 / (1 + Math.exp(-z))

/**
 * Logged contextual-bandit data for off-policy evaluation and learning: contexts `x ~ N(0, I)`, `K` actions with
 * Bernoulli rewards `P(r = 1 | x, a) = logistic(theta_a · x + b_a)`, and a known stochastic logging policy (a softmax
 * of the true scores at temperature `temperature`, mixed with uniform exploration `epsilon`). Every row records the
 * logging propensity of its action. The truth has the reward model, the logging policy and the value of any target
 * policy (`truth.extra.policyValue`, the expected reward over 20000 fresh contexts using the exact reward model).
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { loggedBandit } from 'whitebox-engine/datasets'
 * generate(loggedBandit, { knobs: { rounds: 50, actions: 3 } }).data.nActions // => 3
 * ```
 */
export const loggedBandit: DatasetGenerator<TrajectoryData> = {
  name: 'loggedBandit',
  kind: 'trajectories',
  description: 'Contextual bandit logs from a known softmax logging policy with Bernoulli rewards.',
  knobs: dict({
    rounds: int(1, 10_000_000, { default: 2000 }),
    actions: int(2, 100, { default: 5 }),
    features: int(1, 100, { default: 5 }),
    temperature: real(0.01, 100, { default: 1 }),
    epsilon: real(0, 1, { default: 0.1, label: 'uniform exploration' }),
  }),
  generate: (k, knobs) => {
    const n = knobs['rounds'] as number
    const ka = knobs['actions'] as number
    const d = knobs['features'] as number
    const temp = knobs['temperature'] as number
    const eps = knobs['epsilon'] as number
    const theta = Array.from({ length: ka }, (_, a) => Array.from(normal(foldIn(foldIn(k, 0), a), [d]).data))
    const bias = Array.from(normal(foldIn(k, 1), [ka]).data, (v) => 0.5 * v - 0.5)
    const score = (x: ArrayLike<number>, a: number) =>
      (theta[a] ?? []).reduce((s, t, j) => s + t * (x[j] ?? 0), bias[a] ?? 0)
    const logging = (x: ArrayLike<number>): number[] => {
      const p = softmaxOf(Array.from({ length: ka }, (_, a) => score(x, a) / temp))
      return p.map((v) => (1 - eps) * v + eps / ka)
    }
    const xs = normal(foldIn(k, 2), [n, d]).data
    const u = uniform(foldIn(k, 3), [n, 2]).data
    const actions = new Int32Array(n)
    const rewards = new Float64Array(n)
    const props = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const x = xs.subarray(i * d, (i + 1) * d)
      const p = logging(x)
      let acc = 0
      let a = ka - 1
      for (const [j, pj] of p.entries()) {
        acc += pj
        if ((u[2 * i] ?? 1) < acc) {
          a = j
          break
        }
      }
      actions[i] = a
      props[i] = p[a] ?? 0
      rewards[i] = (u[2 * i + 1] ?? 1) < logistic(score(x, a)) ? 1 : 0
    }
    const fresh = normal(foldIn(k, 4), [20000, d]).data
    const policyValue = (policy: (x: ArrayLike<number>) => readonly number[]): number => {
      let total = 0
      for (let i = 0; i < 20000; i++) {
        const x = fresh.subarray(i * d, (i + 1) * d)
        const p = policy(x)
        for (let a = 0; a < ka; a++) total += (p[a] ?? 0) * logistic(score(x, a))
      }
      return total / 20000
    }
    const optimal = (x: ArrayLike<number>) => {
      let best = 0
      for (let a = 1; a < ka; a++) if (score(x, a) > score(x, best)) best = a
      return Array.from({ length: ka }, (_, a) => (a === best ? 1 : 0))
    }
    return {
      data: trajectories(
        {
          observations: matrix(xs, n, d),
          actions,
          rewards,
          terminations: new Array<boolean>(n).fill(true),
          episodes: Array.from({ length: n }, (_, i) => i),
          propensities: props,
          nActions: ka,
        },
        stamp('loggedBandit', loggedBandit.description, knobs),
      ),
      truth: {
        extra: {
          rewardProbability: (x: ArrayLike<number>, a: number) => logistic(score(x, a)),
          loggingPolicy: logging,
          optimalPolicy: optimal,
          policyValue,
          loggingValue: policyValue(logging),
          optimalValue: policyValue(optimal),
          theta,
          bias,
        },
      },
    }
  },
}

/** A finite MDP: transitions `P[s][a][s']`, expected rewards `R[s][a]`, start distribution and discount. */
export interface TabularMdp {
  readonly states: number
  readonly actions: number
  readonly transitions: readonly (readonly (readonly number[])[])[]
  readonly rewards: readonly (readonly number[])[]
  readonly start: readonly number[]
  readonly gamma: number
}

/**
 * The exact value of a stochastic policy in a tabular MDP: solve `(I - gamma P_pi) V = R_pi`.
 *
 * @param mdp - The MDP.
 * @param policy - `policy[s][a]`, the probability of action `a` in state `s`.
 * @returns The state values `V[s]` and the start-distribution value.
 * @example
 * ```ts
 * import { policyEvaluation } from 'whitebox-engine/datasets'
 * const mdp = { states: 1, actions: 1, transitions: [[[1]]], rewards: [[1]], start: [1], gamma: 0.5 }
 * policyEvaluation(mdp, [[1]]).value // => 2
 * ```
 */
export function policyEvaluation(
  mdp: TabularMdp,
  policy: readonly (readonly number[])[],
): { readonly values: readonly number[]; readonly value: number } {
  const s = mdp.states
  const a = Array.from({ length: s }, (_, i) =>
    Array.from({ length: s }, (_, j) => {
      let p = 0
      for (let act = 0; act < mdp.actions; act++) p += (policy[i]?.[act] ?? 0) * (mdp.transitions[i]?.[act]?.[j] ?? 0)
      return (i === j ? 1 : 0) - mdp.gamma * p
    }),
  )
  const r = Array.from({ length: s }, (_, i) => {
    let v = 0
    for (let act = 0; act < mdp.actions; act++) v += (policy[i]?.[act] ?? 0) * (mdp.rewards[i]?.[act] ?? 0)
    return v
  })
  const values = toFlat(solve(tensor(a), tensor(r)))
  return { values, value: values.reduce((t, v, i) => t + v * (mdp.start[i] ?? 0), 0) }
}

/** Optimal action values by value iteration to a tolerance of 1e-12. */
function optimalQ(mdp: TabularMdp): number[][] {
  let v = new Array<number>(mdp.states).fill(0)
  let q: number[][] = []
  for (let it = 0; it < 100_000; it++) {
    q = Array.from({ length: mdp.states }, (_, s) =>
      Array.from({ length: mdp.actions }, (_, a) => {
        let x = mdp.rewards[s]?.[a] ?? 0
        for (let t = 0; t < mdp.states; t++) x += mdp.gamma * (mdp.transitions[s]?.[a]?.[t] ?? 0) * (v[t] ?? 0)
        return x
      }),
    )
    const next = q.map((row) => row.reduce((m, x) => Math.max(m, x), Number.NEGATIVE_INFINITY))
    const delta = next.reduce((m, x, i) => Math.max(m, Math.abs(x - (v[i] ?? 0))), 0)
    v = next
    if (delta < 1e-12) break
  }
  return q
}

/** The MDP families of {@link offlineMdp}. */
export type MdpLayout = 'random' | 'chain'

function makeMdp(k: Key, layout: MdpLayout, states: number, actions: number, gamma: number): TabularMdp {
  if (layout === 'chain') {
    // Strens' chain: action 0 moves forward (reward 10 at the end), action 1 returns to the start (reward 2); the
    // chosen action slips to the other with probability 0.2.
    const transitions = Array.from({ length: states }, (_, s) =>
      Array.from({ length: 2 }, (_, a) =>
        Array.from({ length: states }, (_, t) => {
          const forward = t === Math.min(states - 1, s + 1) ? 1 : 0
          const back = t === 0 ? 1 : 0
          return a === 0 ? 0.8 * forward + 0.2 * back : 0.8 * back + 0.2 * forward
        }),
      ),
    )
    const rewards = Array.from({ length: states }, (_, s) => [
      s === states - 1 ? 0.8 * 10 + 0.2 * 2 : 0.2 * 2,
      0.8 * 2 + (s === states - 1 ? 0.2 * 10 : 0),
    ])
    return {
      states,
      actions: 2,
      transitions,
      rewards,
      start: Array.from({ length: states }, (_, s) => (s === 0 ? 1 : 0)),
      gamma,
    }
  }
  const transitions = Array.from({ length: states }, (_, s) =>
    Array.from({ length: actions }, (_, a) =>
      Array.from(dirichlet(foldIn(foldIn(k, 0), s * actions + a), new Array(states).fill(0.5), 1).data),
    ),
  )
  const raw = normal(foldIn(k, 1), [states, actions]).data
  const rewards = Array.from({ length: states }, (_, s) =>
    Array.from({ length: actions }, (_, a) => raw[s * actions + a] ?? 0),
  )
  return { states, actions, transitions, rewards, start: new Array<number>(states).fill(1 / states), gamma }
}

/**
 * Offline reinforcement-learning data from a known MDP and a known behaviour policy: a random tabular MDP (Dirichlet
 * transitions, Gaussian rewards) or Strens' chain, and episodes of fixed length from an epsilon-greedy policy around
 * the optimal one. Observations are state indices. The truth is the MDP itself, the optimal action values, the
 * behaviour policy and its exact value, and `truth.extra.policyValue` for any tabular policy.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { offlineMdp } from 'whitebox-engine/datasets'
 * generate(offlineMdp, { knobs: { episodes: 5, horizon: 10 } }).data.rewards.shape // => [50]
 * ```
 */
export const offlineMdp: DatasetGenerator<TrajectoryData> = {
  name: 'offlineMdp',
  kind: 'trajectories',
  description: 'Episodes from a known tabular MDP under a known epsilon-greedy behaviour policy.',
  knobs: dict({
    layout: choice(['random', 'chain'], { default: 'random' }),
    states: int(2, 500, { default: 10 }),
    actions: int(2, 20, { default: 3 }),
    gamma: real(0, 0.999, { default: 0.9 }),
    episodes: int(1, 100_000, { default: 100 }),
    horizon: int(1, 10_000, { default: 20 }),
    epsilon: real(0, 1, { default: 0.3, label: 'behaviour exploration' }),
  }),
  generate: (k, knobs) => {
    const mdp = makeMdp(
      foldIn(k, 0),
      knobs['layout'] as MdpLayout,
      knobs['states'] as number,
      knobs['actions'] as number,
      knobs['gamma'] as number,
    )
    const q = optimalQ(mdp)
    const eps = knobs['epsilon'] as number
    const behaviour = q.map((row) => {
      let best = 0
      for (let a = 1; a < row.length; a++) if ((row[a] ?? 0) > (row[best] ?? 0)) best = a
      return row.map((_, a) => (1 - eps) * (a === best ? 1 : 0) + eps / mdp.actions)
    })
    const episodes = knobs['episodes'] as number
    const horizon = knobs['horizon'] as number
    const total = episodes * horizon
    const obs = new Float64Array(total)
    const next = new Float64Array(total)
    const acts = new Int32Array(total)
    const rew = new Float64Array(total)
    const trunc = new Array<boolean>(total).fill(false)
    const ep = new Int32Array(total)
    const props = new Float64Array(total)
    const starts = categorical(foldIn(k, 1), mdp.start as number[], [episodes]).data
    const noise = normal(foldIn(k, 2), [total]).data
    for (let e = 0; e < episodes; e++) {
      let s = starts[e] ?? 0
      const ke = foldIn(foldIn(k, 3), e)
      const ua = uniform(foldIn(ke, 0), [horizon]).data
      const us = uniform(foldIn(ke, 1), [horizon]).data
      for (let t = 0; t < horizon; t++) {
        const i = e * horizon + t
        const pi = behaviour[s] ?? []
        let a = pi.length - 1
        let acc = 0
        for (const [j, p] of pi.entries()) {
          acc += p
          if ((ua[t] ?? 1) < acc) {
            a = j
            break
          }
        }
        const row = mdp.transitions[s]?.[a] ?? []
        let s2 = row.length - 1
        acc = 0
        for (const [j, p] of row.entries()) {
          acc += p
          if ((us[t] ?? 1) < acc) {
            s2 = j
            break
          }
        }
        obs[i] = s
        acts[i] = a
        rew[i] = (mdp.rewards[s]?.[a] ?? 0) + (knobs['layout'] === 'chain' ? 0 : 0.1 * (noise[i] ?? 0))
        next[i] = s2
        ep[i] = e
        props[i] = pi[a] ?? 0
        trunc[i] = t === horizon - 1
        s = s2
      }
    }
    const optimal = q.map((row) => {
      let best = 0
      for (let a = 1; a < row.length; a++) if ((row[a] ?? 0) > (row[best] ?? 0)) best = a
      return row.map((_, a) => (a === best ? 1 : 0))
    })
    return {
      data: trajectories(
        {
          observations: matrix(obs, total, 1),
          actions: acts,
          rewards: rew,
          nextObservations: matrix(next, total, 1),
          truncations: trunc,
          episodes: ep,
          propensities: props,
          nActions: mdp.actions,
        },
        stamp('offlineMdp', offlineMdp.description, knobs),
      ),
      truth: {
        extra: {
          mdp,
          optimalQ: q,
          optimalPolicy: optimal,
          optimalValue: policyEvaluation(mdp, optimal).value,
          behaviourPolicy: behaviour,
          behaviourValue: policyEvaluation(mdp, behaviour).value,
          policyValue: (policy: readonly (readonly number[])[]) => policyEvaluation(mdp, policy).value,
        },
      },
    }
  },
}
