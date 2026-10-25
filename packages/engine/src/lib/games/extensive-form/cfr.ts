/**
 * Counterfactual regret minimisation (CFR), the standard method for solving imperfect-information games such as poker:
 * regrets are accumulated at every information set and the average policy converges to a Nash equilibrium. A traced
 * algorithm following OpenSpiel's CFR, CFR+, linear CFR and discounted CFR solvers step for step, with exploitability
 * tracked along the way.
 */

import { WhiteboxError } from '../../base/index.ts'
import { type Algorithm, runAlgorithm } from '../../trace/index.ts'
import type { GameRunOptions } from '../normal-form/index.ts'
import { averageOf, CFR_SCHEMA, cfrObservation, regretMatch } from './_regrets.ts'
import { CHANCE, TERMINAL, tablePolicy } from './_tree.ts'
import { exploitability } from './best-response.ts'
import { type ExtensiveGame, type GameTree, type TabularPolicy, treeOf } from './tree.ts'

/** Options for {@link cfr}. */
export interface CfrOptions extends GameRunOptions {
  /** Iterations (default 1000). */
  readonly iterations?: number
  /**
   * The variant (OpenSpiel's solvers): `'cfr'` (Zinkevich et al. 2007), `'cfr+'` (Tammelin 2014: regret matching+ and
   * linear averaging), `'linear'` (LCFR) or `'discounted'` (DCFR with alpha, beta, gamma; Brown and Sandholm 2019).
   */
  readonly variant?: 'cfr' | 'cfr+' | 'linear' | 'discounted'
  /** Update the players in turn, each against the others' updated policies (default true, as OpenSpiel). */
  readonly alternating?: boolean
  /** DCFR's discount of positive regrets, t^alpha / (t^alpha + 1) (default 1.5). */
  readonly alpha?: number
  /** DCFR's discount of negative regrets, t^beta / (t^beta + 1) (default 0). */
  readonly beta?: number
  /** DCFR's weight t^gamma of iteration t in the average policy (default 2). */
  readonly gamma?: number
}

/** The state of a CFR run: cumulative regrets and policy sums per information set, and the current policy. */
export interface CfrState {
  readonly t: number
  readonly regrets: readonly Float64Array[]
  readonly policySums: readonly Float64Array[]
  readonly current: readonly Float64Array[]
}

/** The outcome of {@link cfr}. */
export interface CfrResult {
  readonly iterations: number
  /** The average policy, which converges to a Nash equilibrium in two-player zero-sum games. */
  readonly averagePolicy: TabularPolicy
  /** The last policy (for CFR+ it converges too, in practice). */
  readonly currentPolicy: TabularPolicy
  /** The exploitability of the average policy. */
  readonly exploitability: number
}

/**
 * Counterfactual regret minimisation as a traced algorithm, following OpenSpiel's `CFRSolver`, `CFRPlusSolver`,
 * `LCFRSolver` and `DCFRSolver` step for step: each iteration walks the whole tree once per player (alternating
 * updates), accumulates counterfactual regrets and reach-weighted policy sums at that player's information sets,
 * then sets the current policy by regret matching. Each step emits the exploitability of the average policy when
 * traced.
 *
 * @param game - A game or its tree.
 * @param options - The variant, alternation and DCFR's parameters.
 * @returns The algorithm; each step is one iteration.
 * @throws WhiteboxError with code `value` for an unknown variant.
 * @concept counterfactual-regret
 * @reference Zinkevich, Johanson, Bowling and Piccione (2007), Regret minimization in games with incomplete information.
 * @example
 * ```ts
 * import { cfrAlgorithm, kuhnPoker } from 'whitebox-engine/games'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * runAlgorithm(cfrAlgorithm(kuhnPoker()), { maxSteps: 10 }).state.t // => 10
 * ```
 */
export function cfrAlgorithm<S>(
  game: ExtensiveGame<S> | GameTree,
  options: Omit<CfrOptions, 'iterations'> = {},
): Algorithm<CfrState> {
  const tree = treeOf(game)
  const variant = options.variant ?? 'cfr'
  if (!['cfr', 'cfr+', 'linear', 'discounted'].includes(variant))
    throw new WhiteboxError('value', `cfr: unknown variant '${variant}'`)
  const alternating = options.alternating ?? true
  const plus = variant === 'cfr+'
  const discounted = variant === 'linear' || variant === 'discounted'
  const alpha = variant === 'linear' ? 1 : (options.alpha ?? 1.5)
  const beta = variant === 'linear' ? 1 : (options.beta ?? 0)
  const gamma = variant === 'linear' ? 1 : variant === 'cfr+' ? 1 : (options.gamma ?? 2)
  const weighted = variant !== 'cfr'
  const n = tree.numPlayers
  const sweep = (
    state: { regrets: Float64Array[]; sums: Float64Array[]; current: readonly Float64Array[] },
    t: number,
    player: number,
  ): void => {
    const weight = weighted ? t ** gamma : 1
    const visit = (id: number, reach: Float64Array): Float64Array => {
      const node = tree.nodes[id]
      if (node === undefined) return new Float64Array(n)
      if (node.player === TERMINAL) return Float64Array.from(node.returns)
      if (node.player === CHANCE) {
        const v = new Float64Array(n)
        for (const [a, c] of node.children.entries()) {
          const p = node.chance[a] ?? 0
          const next = Float64Array.from(reach)
          next[n] = (next[n] ?? 0) * p
          const cv = visit(c, next)
          for (let q = 0; q < n; q++) v[q] = (v[q] ?? 0) + p * (cv[q] ?? 0)
        }
        return v
      }
      let alive = false
      for (let q = 0; q < n; q++) if ((reach[q] ?? 0) !== 0) alive = true
      if (!alive) return new Float64Array(n)
      const k = node.infoset
      const policy = state.current[k] ?? new Float64Array(0)
      const v = new Float64Array(n)
      const children: Float64Array[] = []
      for (const [a, c] of node.children.entries()) {
        const p = policy[a] ?? 0
        const next = Float64Array.from(reach)
        next[node.player] = (next[node.player] ?? 0) * p
        const cv = visit(c, next)
        children.push(cv)
        for (let q = 0; q < n; q++) v[q] = (v[q] ?? 0) + p * (cv[q] ?? 0)
      }
      if (player >= 0 && node.player !== player) return v
      const me = node.player
      let cf = 1
      for (let q = 0; q <= n; q++) if (q !== me) cf *= reach[q] ?? 0
      const regrets = state.regrets[k] ?? new Float64Array(0)
      const sums = state.sums[k] ?? new Float64Array(0)
      for (let a = 0; a < policy.length; a++) {
        regrets[a] = (regrets[a] ?? 0) + cf * ((children[a]?.[me] ?? 0) - (v[me] ?? 0))
        sums[a] = (sums[a] ?? 0) + weight * (reach[me] ?? 0) * (policy[a] ?? 0)
      }
      return v
    }
    visit(0, new Float64Array(n + 1).fill(1))
  }
  const owners = tree.infosets.map((i) => i.player)
  return {
    name: `cfr-${variant}`,
    config: { variant, alternating, ...(variant === 'discounted' ? { alpha, beta, gamma } : {}) },
    schema: CFR_SCHEMA,
    init: () => ({
      t: 0,
      regrets: tree.infosets.map((i) => new Float64Array(i.actions.length)),
      policySums: tree.infosets.map((i) => new Float64Array(i.actions.length)),
      current: tree.infosets.map((i) => new Float64Array(i.actions.length).fill(1 / i.actions.length)),
    }),
    step: (s) => {
      const t = s.t + 1
      const state = {
        regrets: s.regrets.map((r) => Float64Array.from(r)),
        sums: s.policySums.map((r) => Float64Array.from(r)),
        current: s.current as readonly Float64Array[],
      }
      const finish = (who: number): void => {
        if (plus) for (const r of state.regrets) for (let a = 0; a < r.length; a++) if ((r[a] ?? 0) < 0) r[a] = 0
        if (discounted) {
          const pos = t ** alpha / (t ** alpha + 1)
          const neg = t ** beta / (t ** beta + 1)
          for (const [k, r] of state.regrets.entries()) {
            if (who >= 0 && owners[k] !== who) continue
            for (let a = 0; a < r.length; a++) r[a] = (r[a] ?? 0) * ((r[a] ?? 0) >= 0 ? pos : neg)
          }
        }
        state.current = state.regrets.map(regretMatch)
      }
      if (alternating)
        for (let p = 0; p < n; p++) {
          sweep(state, t, p)
          finish(p)
        }
      else {
        sweep(state, t, -1)
        finish(-1)
      }
      return { t, regrets: state.regrets, policySums: state.sums, current: state.current }
    },
    observe: (s) => cfrObservation(tree, s.t, s.regrets, s.policySums),
  }
}

/**
 * Run counterfactual regret minimisation (see {@link cfrAlgorithm}) on a game.
 *
 * @param game - A game or its tree.
 * @param options - Iterations, the variant, alternation, DCFR's parameters and tracing.
 * @returns The average and current policies and the average's exploitability.
 * @concept counterfactual-regret
 * @example
 * ```ts
 * import { cfr, kuhnPoker } from 'whitebox-engine/games'
 * cfr(kuhnPoker(), { iterations: 200 }).exploitability < 0.01 // => true
 * ```
 */
export function cfr<S>(game: ExtensiveGame<S> | GameTree, options: CfrOptions = {}): CfrResult {
  const tree = treeOf(game)
  const { iterations = 1000, ...rest } = options
  const { state } = runAlgorithm(cfrAlgorithm(tree, rest), { ...rest, maxSteps: iterations })
  const averagePolicy = tablePolicy(tree, averageOf(state.policySums))
  return {
    iterations: state.t,
    averagePolicy,
    currentPolicy: tablePolicy(tree, state.current),
    exploitability: exploitability(tree, averagePolicy),
  }
}

/**
 * CFR+ (Tammelin 2014): {@link cfr} with regret matching+ and linearly weighted averages, as OpenSpiel's
 * `CFRPlusSolver`; it converges much faster in practice.
 *
 * @param game - A game or its tree.
 * @param options - Iterations, alternation and tracing.
 * @returns The average and current policies and the average's exploitability.
 * @concept counterfactual-regret
 * @example
 * ```ts
 * import { cfrPlus, kuhnPoker } from 'whitebox-engine/games'
 * cfrPlus(kuhnPoker(), { iterations: 200 }).exploitability < 0.001 // => true
 * ```
 */
export function cfrPlus<S>(game: ExtensiveGame<S> | GameTree, options: Omit<CfrOptions, 'variant'> = {}): CfrResult {
  return cfr(game, { ...options, variant: 'cfr+' })
}
