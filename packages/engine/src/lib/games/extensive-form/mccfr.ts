/**
 * Monte Carlo counterfactual regret minimisation: CFR that samples parts of the game tree each iteration instead of
 * walking all of it, so it scales to larger games. External and outcome sampling as a traced algorithm, following
 * OpenSpiel's solvers.
 */

import { OmaiError } from '../../base/index.ts'
import { Generator, type Key } from '../../random/index.ts'
import { type Algorithm, runAlgorithm } from '../../trace/index.ts'
import type { GameRunOptions } from '../normal-form/index.ts'
import { averageOf, CFR_SCHEMA, cfrObservation, regretMatch } from './_regrets.ts'
import { CHANCE, TERMINAL, tablePolicy } from './_tree.ts'
import { exploitability } from './best-response.ts'
import type { CfrResult } from './cfr.ts'
import { type ExtensiveGame, type GameTree, treeOf } from './tree.ts'

/** Options for {@link mccfr}. */
export interface MccfrOptions extends GameRunOptions {
  /** Iterations (default 10000); each samples once per player. */
  readonly iterations?: number
  /** External sampling (Lanctot et al. 2009; the default) or outcome sampling. */
  readonly sampling?: 'external' | 'outcome'
  /** Outcome sampling's exploration: the weight of the uniform policy at the updating player's nodes (default 0.6). */
  readonly exploration?: number
}

/** The state of Monte Carlo CFR: cumulative regrets and average-policy sums per information set. */
export interface MccfrState {
  readonly t: number
  readonly regrets: readonly Float64Array[]
  readonly policySums: readonly Float64Array[]
}

/** A stream of uniform draws in [0, 1) from a key. */
function stream(key: Key): () => number {
  const rng = new Generator(key)
  let block = new Float64Array(0)
  let i = 0
  return () => {
    if (i >= block.length) {
      block = Float64Array.from(rng.uniform([256]).data)
      i = 0
    }
    const u = block[i] ?? 0
    i += 1
    return u
  }
}

function sample(probs: ArrayLike<number>, u: number): number {
  let acc = 0
  for (let a = 0; a < probs.length; a++) {
    acc += probs[a] ?? 0
    if (u < acc) return a
  }
  return probs.length - 1
}

/**
 * Monte Carlo counterfactual regret minimisation as a traced algorithm, following OpenSpiel's
 * `ExternalSamplingSolver` (simple averaging) and `OutcomeSamplingSolver`: every iteration samples, for each player in
 * turn, chance outcomes and (external) the other players' actions, or (outcome) one whole trajectory with an
 * exploratory policy and importance weights; regrets and average-policy sums start at 1e-6 as in OpenSpiel. Draws come
 * from the step keys, so a run is reproducible.
 *
 * @param game - A game or its tree.
 * @param options - The sampling scheme and exploration.
 * @returns The algorithm; each step is one iteration.
 * @throws OmaiError with code `value` for an exploration outside (0, 1].
 * @concept counterfactual-regret
 * @reference Lanctot, Waugh, Zinkevich and Bowling (2009), Monte Carlo sampling for regret minimization in extensive games.
 * @example
 * ```ts
 * import { kuhnPoker, mccfrAlgorithm } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(mccfrAlgorithm(kuhnPoker()), { maxSteps: 10, seed: 0 }).state.t // => 10
 * ```
 */
export function mccfrAlgorithm<S>(
  game: ExtensiveGame<S> | GameTree,
  options: Omit<MccfrOptions, 'iterations'> = {},
): Algorithm<MccfrState> {
  const tree = treeOf(game)
  const sampling = options.sampling ?? 'external'
  const eps = options.exploration ?? 0.6
  if (!(eps > 0 && eps <= 1)) throw new OmaiError('value', 'mccfr: exploration must be in (0, 1]')
  const n = tree.numPlayers
  const external = (
    s: { regrets: Float64Array[]; sums: Float64Array[] },
    id: number,
    player: number,
    u: () => number,
  ): number => {
    const node = tree.nodes[id]
    if (node === undefined) return 0
    if (node.player === TERMINAL) return node.returns[player] ?? 0
    if (node.player === CHANCE) return external(s, node.children[sample(node.chance, u())] ?? 0, player, u)
    const k = node.infoset
    const policy = regretMatch(s.regrets[k] ?? new Float64Array(0))
    let value = 0
    if (node.player !== player) value = external(s, node.children[sample(policy, u())] ?? 0, player, u)
    else {
      const child = node.children.map((c) => external(s, c, player, u))
      for (const [a, v] of child.entries()) value += (policy[a] ?? 0) * v
      const r = s.regrets[k] ?? new Float64Array(0)
      for (const [a, v] of child.entries()) r[a] = (r[a] ?? 0) + v - value
    }
    // Simple averaging: the next player's policy is accumulated where it is sampled.
    if (node.player === (player + 1) % n) {
      const sums = s.sums[k] ?? new Float64Array(0)
      for (let a = 0; a < sums.length; a++) sums[a] = (sums[a] ?? 0) + (policy[a] ?? 0)
    }
    return value
  }
  const outcome = (
    s: { regrets: Float64Array[]; sums: Float64Array[] },
    id: number,
    player: number,
    mine: number,
    others: number,
    sampled: number,
    u: () => number,
  ): number => {
    const node = tree.nodes[id]
    if (node === undefined) return 0
    if (node.player === TERMINAL) return node.returns[player] ?? 0
    if (node.player === CHANCE) {
      const a = sample(node.chance, u())
      const p = node.chance[a] ?? 0
      return outcome(s, node.children[a] ?? 0, player, mine, p * others, p * sampled, u)
    }
    const k = node.infoset
    const policy = regretMatch(s.regrets[k] ?? new Float64Array(0))
    const m = policy.length
    const behaviour = node.player === player ? policy.map((p) => eps / m + (1 - eps) * p) : policy
    const a = sample(behaviour, u())
    const pa = policy[a] ?? 0
    const child = outcome(
      s,
      node.children[a] ?? 0,
      player,
      node.player === player ? mine * pa : mine,
      node.player === player ? others : others * pa,
      sampled * (behaviour[a] ?? 1),
      u,
    )
    // Importance-corrected action values: the sampled one is child / q(a), the others 0 (no baseline).
    const values = policy.map((_, b) => (b === a ? child / (behaviour[a] ?? 1) : 0))
    let estimate = 0
    for (let b = 0; b < m; b++) estimate += (policy[b] ?? 0) * (values[b] ?? 0)
    if (node.player === player) {
      const r = s.regrets[k] ?? new Float64Array(0)
      const sums = s.sums[k] ?? new Float64Array(0)
      for (let b = 0; b < m; b++) {
        r[b] = (r[b] ?? 0) + ((values[b] ?? 0) - estimate) * (others / sampled)
        sums[b] = (sums[b] ?? 0) + (mine * (policy[b] ?? 0)) / sampled
      }
    }
    return estimate
  }
  return {
    name: `mccfr-${sampling}`,
    config: { sampling, ...(sampling === 'outcome' ? { exploration: eps } : {}) },
    schema: CFR_SCHEMA,
    init: () => ({
      t: 0,
      regrets: tree.infosets.map((i) => new Float64Array(i.actions.length).fill(1e-6)),
      policySums: tree.infosets.map((i) => new Float64Array(i.actions.length).fill(1e-6)),
    }),
    step: (st, ctx) => {
      const u = stream(ctx.key)
      const s = {
        regrets: st.regrets.map((r) => Float64Array.from(r)),
        sums: st.policySums.map((r) => Float64Array.from(r)),
      }
      for (let p = 0; p < n; p++) {
        if (sampling === 'external') external(s, 0, p, u)
        else outcome(s, 0, p, 1, 1, 1, u)
      }
      return { t: st.t + 1, regrets: s.regrets, policySums: s.sums }
    },
    observe: (s) => cfrObservation(tree, s.t, s.regrets, s.policySums),
  }
}

/**
 * Run Monte Carlo CFR (see {@link mccfrAlgorithm}).
 *
 * @param game - A game or its tree.
 * @param options - Iterations, sampling scheme, exploration, seed or key, and tracing.
 * @returns The average policy, the current (regret-matching) policy and the average's exploitability.
 * @concept counterfactual-regret
 * @example
 * ```ts
 * import { kuhnPoker, mccfr } from 'omai-engine/games'
 * mccfr(kuhnPoker(), { iterations: 3000, seed: 1 }).exploitability < 0.05 // => true
 * ```
 */
export function mccfr<S>(game: ExtensiveGame<S> | GameTree, options: MccfrOptions = {}): CfrResult {
  const tree = treeOf(game)
  const { iterations = 10_000, ...rest } = options
  const { state } = runAlgorithm(mccfrAlgorithm(tree, rest), { ...rest, maxSteps: iterations })
  const averagePolicy = tablePolicy(tree, averageOf(state.policySums))
  return {
    iterations: state.t,
    averagePolicy,
    currentPolicy: tablePolicy(tree, state.regrets.map(regretMatch)),
    exploitability: exploitability(tree, averagePolicy),
  }
}
