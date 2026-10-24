import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../../../lib/random/index.ts'
import {
  type AecEnv,
  evaluatePolicy,
  greedyPolicyOf,
  playAec,
  stateObservations,
  timeLimit,
  trainAgent,
} from '../../../lib/rl/index.ts'
import { cartPole, cliffWalking, frozenLake, nim, ticTacToe } from '../../../methods/rl/index.ts'
import {
  alphaBeta,
  alphaZero,
  alphaZeroAgent,
  gameMctsAgent,
  mctsAgent,
  minimaxAgent,
  policyIterationAgent,
  randomLegalPolicy,
  sweepingPlannerAgent,
  valueIterationAgent,
} from './planning.ts'

// Search and learning tests stay small; the generous timeout covers a loaded machine.
const SLOW = { timeout: 30_000 }

/** Win, draw and loss counts of `policy` (for `seat`) against uniformly random legal play. */
function versusRandom<S>(
  env: AecEnv<S, readonly number[], number>,
  policy: (state: S, k: ReturnType<typeof key>) => number,
  seat: 'player_0' | 'player_1',
  games: number,
): { win: number; draw: number; loss: number } {
  const g = stateObservations(env)
  const other = seat === 'player_0' ? 'player_1' : 'player_0'
  const out = { win: 0, draw: 0, loss: 0 }
  for (let i = 0; i < games; i++) {
    const ep = playAec(g, { [seat]: policy, [other]: randomLegalPolicy(g, other) }, key(1000 + i))
    const r = ep.returns[seat] ?? 0
    if (r > 0) out.win += 1
    else if (r < 0) out.loss += 1
    else out.draw += 1
  }
  return out
}

describe('dynamic-programming agents', () => {
  it('plan once on the model and act greedily', () => {
    const env = cliffWalking.create()
    for (const agent of [valueIterationAgent(env, { gamma: 1 }), policyIterationAgent(env, { gamma: 0.9 })]) {
      const s = agent.init(key(0))
      // The optimal path along the cliff: 13 steps, return -13.
      const ev = evaluatePolicy(env, greedyPolicyOf(agent, s), { episodes: 2, key: key(1) })
      expect(ev.meanReturn).toBe(-13)
      expect(
        agent.learn(
          s,
          { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false },
          key(0),
        ),
      ).toBe(s)
    }
    expect(valueIterationAgent(env, { gamma: 1 }).stats?.(valueIterationAgent(env, { gamma: 1 }).init(key(0)))).toEqual(
      {
        startValue: -13,
      },
    )
    expect(() => valueIterationAgent(cartPole.create())).toThrow(/no model/)
    const lake = frozenLake.create({ isSlippery: false })
    const r = trainAgent(lake, valueIterationAgent(lake, { gamma: 0.9 }), { episodes: 1, checkpoints: false })
    expect(r.returns).toEqual([1])
  })

  it('plan a sweep at a time, improving episode by episode to the optimum', () => {
    // The planner reads the model of the bare environment; it trains under a time limit.
    const dry = frozenLake.create({ isSlippery: false })
    const lake = timeLimit(dry, 50)
    const vi = sweepingPlannerAgent(dry, { gamma: 0.9 })
    // Rewards reach the start after six sweeps; until then the greedy policy never reaches the goal.
    const early = trainAgent(lake, vi, { episodes: 12, maxSteps: 2000, checkpoints: false })
    expect(early.returns.slice(0, 3)).toEqual([0, 0, 0])
    expect(early.returns.at(-1)).toBe(1)
    expect(early.agent.iterations).toBe(12)
    const edge = cliffWalking.create()
    const cliff = timeLimit(edge, 100)
    const pi = sweepingPlannerAgent(edge, { planner: 'policyIteration', gamma: 0.9, every: 2, sweeps: 3 })
    const s = trainAgent(cliff, pi, { episodes: 20, maxSteps: 4000, checkpoints: false }).agent
    expect(s.iterations).toBe(30)
    expect(s.residual).toBe(0)
    expect(evaluatePolicy(cliff, greedyPolicyOf(pi, s), { episodes: 1, key: key(1) }).meanReturn).toBe(-13)
    expect(() => sweepingPlannerAgent(cartPole.create())).toThrow(/no model/)
  })
})

describe('MCTS agents', () => {
  it('balance a pole by planning from the true state', SLOW, () => {
    const env = cartPole.create()
    const agent = mctsAgent(env, { simulations: 30, depth: 20, c: 5 })
    let s = env.reset(key(3)).state
    let steps = 0
    for (; steps < 60; steps++) {
      const r = env.step(s, agent.act(null, s, foldIn(key(4), steps)), key(steps))
      if (r.terminated) break
      s = r.state
    }
    expect(steps).toBe(60)
  })

  it('play games with negamax backups', SLOW, () => {
    const env = nim.create({ heaps: [1, 2, 3, 4] })
    const agent = gameMctsAgent(env, { simulations: 200 })
    const counts = versusRandom(env, (s, k) => agent.act(null, s, k), 'player_0', 10)
    expect(counts.win).toBeGreaterThanOrEqual(9)
  })
})

describe('minimax agents', () => {
  it('search tic-tac-toe exactly and never lose', SLOW, () => {
    const env = ticTacToe.create()
    const root = alphaBeta(env, env.reset(key(0)))
    expect(root.value).toBe(0)
    const plain = alphaBeta(env, env.reset(key(0)), { stateKey: null })
    expect(plain.value).toBe(0)
    expect(root.nodes).toBeLessThan(plain.nodes)
    const agent = minimaxAgent(env, { stateKey: (s) => s.board.join('') })
    for (const seat of ['player_0', 'player_1'] as const) {
      expect(versusRandom(env, (s) => agent.greedy(null, s), seat, 15).loss).toBe(0)
    }
    // Self-play between perfect players is a draw.
    const g = stateObservations(env)
    expect(
      playAec(g, { player_0: (s) => agent.greedy(null, s), player_1: (s) => agent.greedy(null, s) }, key(0)).returns,
    ).toEqual({
      player_0: 0,
      player_1: 0,
    })
  })

  it('play Nim perfectly: a winning move whenever the xor of the heaps is non-zero', () => {
    const env = nim.create({ heaps: [1, 3, 5, 7] })
    const agent = minimaxAgent(env, { stateKey: (s) => s.heaps.join(',') })
    for (const heaps of [
      [1, 3, 5, 6],
      [0, 3, 5, 7],
      [1, 2, 4, 6],
      [1, 1, 1, 0],
    ]) {
      const s = { ...env.reset(key(0)), heaps }
      const a = agent.greedy(null, s)
      const after = env.step(s, a, key(0)).state.heaps
      expect(
        after.reduce((x, h) => x ^ h, 0),
        heaps.join(','),
      ).toBe(0)
    }
    expect(alphaBeta(env, env.reset(key(0)), { stateKey: (s) => s.heaps.join(',') }).value).toBe(-1)
  })

  it('cut off at a depth with an evaluation function and order moves', () => {
    const env = ticTacToe.create()
    const centre = (s: { readonly board: readonly number[] }, player: string): number =>
      s.board[4] === (player === 'player_0' ? 1 : 2) ? 0.5 : s.board[4] === 0 ? 0 : -0.5
    const r = alphaBeta(env, env.reset(key(0)), {
      depth: 1,
      evaluate: centre,
      order: (_, actions) => [...actions].reverse(),
    })
    expect(r.action).toBe(4)
    expect(r.value).toBe(0.5)
  })
})

describe('AlphaZero', () => {
  it('learns from self-play to beat a random player', SLOW, () => {
    const env = nim.create({ heaps: [1, 2, 3] })
    const opts = {
      hidden: [32],
      simulations: 16,
      gamesPerIteration: 8,
      trainSteps: 32,
      batchSize: 32,
      iterations: 14,
      learningRate: 3e-3,
    }
    const trained = alphaZero(env, { ...opts, key: key(1), seed: 2 })
    const agent = alphaZeroAgent(env, trained.params, { ...opts, simulations: 0 })
    const untrained = alphaZeroAgent(env, alphaZero(env, { ...opts, iterations: 0, key: key(1) }).params, {
      ...opts,
      simulations: 0,
    })
    const trainedCounts = versusRandom(env, (s) => agent.greedy(trained.params, s), 'player_1', 40)
    const baseline = versusRandom(env, (s) => untrained.greedy(untrained.init(key(0)), s), 'player_1', 40)
    // [1, 2, 3] is a second-player win; the raw trained policy (no search) finds it against most random openings.
    expect(trainedCounts.win).toBeGreaterThanOrEqual(30)
    expect(trainedCounts.win).toBeGreaterThan(baseline.win + 10)
    expect(trained.state.games).toBe(14 * 8)
    expect(trained.state.valueLoss).toBeLessThan(0.6)
    // With search on top of the network, the agent takes the immediate win in a single heap.
    const one = nim.create({ heaps: [3] })
    const searcher = alphaZeroAgent(
      one,
      alphaZero(one, { hidden: [8], iterations: 1, gamesPerIteration: 1, simulations: 4, trainSteps: 1 }).params,
      {
        hidden: [8],
        simulations: 30,
      },
    )
    expect(searcher.greedy(searcher.init(key(0)), one.reset(key(0)))).toBe(2)
  })
})
