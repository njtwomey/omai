import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, trainAgent } from '../../../lib/rl/index.ts'
import { discrete } from '../../../lib/space/index.ts'
import { FOUR_ROOMS, fourRooms, makeFourRooms } from '../environments/index.ts'
import {
  hallwayOptions,
  intraOptionQLearning,
  optionCritic,
  optionTerminations,
  optionValues,
  primitiveOptions,
  smdpQLearning,
  subgoalOption,
} from './hierarchical.ts'
import { qLearning } from './tabular.ts'

const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)
const mean = (xs: readonly number[]): number => sum(xs) / xs.length

// Free cells of the layout in row-major order, as makeFourRooms indexes its states.
const cells: [number, number][] = []
for (const [r, row] of FOUR_ROOMS.entries()) for (let c = 0; c < row.length; c++) if (row[c] !== 'w') cells.push([r, c])
const cellIndex = (r: number, c: number): number => cells.findIndex(([a, b]) => a === r && b === c)
const HALLWAYS = [cellIndex(3, 6), cellIndex(6, 2), cellIndex(7, 9), cellIndex(10, 6)]

describe('options in the four-rooms domain (Sutton, Precup and Singh 1999)', () => {
  it('derives eight hallway options whose policies reach their hallway from anywhere in their room', () => {
    const exact = makeFourRooms({ success: 1 })
    const options = hallwayOptions(exact)
    expect(options).toHaveLength(8)
    const model = exact.model
    if (model === undefined) throw new Error('no model')
    for (const o of options) {
      // The goal (the east hallway by default) is absorbing, so walks start elsewhere.
      const starts = cells.map((_, i) => i).filter((s) => o.initiation(s) && s !== cellIndex(7, 9))
      expect(starts.length).toBeGreaterThanOrEqual(20)
      for (const start of starts) {
        let s = start
        for (let t = 0; t < 30 && o.termination(s) < 1; t++) {
          const a = (o.policy(s) as number[]).indexOf(1)
          s = model.transitions(s, a).find((x) => x.probability === 1)?.state ?? s
        }
        expect(o.termination(s)).toBe(1)
        expect(HALLWAYS).toContain(s)
      }
      // Each hallway is the target of two options (one from each adjacent room).
    }
    const targets = options.map((o) => o.name.replace(/^room \d+ to /, ''))
    for (const t of new Set(targets)) expect(targets.filter((x) => x === t)).toHaveLength(2)
  })

  it('primitive options are one-step options available everywhere', () => {
    const [, right] = primitiveOptions(4)
    expect([right?.initiation(5), right?.policy(5), right?.termination(9)]).toEqual([true, [0, 1, 0, 0], 1])
    const model = makeFourRooms().model
    if (model === undefined) throw new Error('no model')
    const toHallway = subgoalOption(model, [cellIndex(3, 5)], cellIndex(3, 6))
    expect(toHallway.policy(cellIndex(3, 5))).toEqual([0, 1, 0, 0])
    expect(toHallway.initiation(cellIndex(3, 6))).toBe(false)
  })

  it('hallway options speed up learning compared with primitive Q-learning', () => {
    const env = make(fourRooms)
    const options = [...hallwayOptions(makeFourRooms()), ...primitiveOptions(4)]
    const steps = (agent: Parameters<typeof trainAgent>[1]): number =>
      trainAgent(env, agent, { episodes: 60, seed: 0, checkpoints: false }).steps
    const flat = steps(qLearning(env, { alpha: 0.1, gamma: 0.9 }) as never)
    const smdp = steps(smdpQLearning(env, { options, alpha: 0.1, gamma: 0.9 }) as never)
    const intra = steps(intraOptionQLearning(env, { options, alpha: 0.1, gamma: 0.9 }) as never)
    expect(smdp).toBeLessThan(flat / 3)
    expect(intra).toBeLessThan(flat / 3)
  })

  it('intra-option Q-learning learns a good greedy policy that polls the best option at every step', () => {
    const env = make(fourRooms)
    const options = [...hallwayOptions(makeFourRooms()), ...primitiveOptions(4)]
    const agent = intraOptionQLearning(env, { options, alpha: 0.1, gamma: 0.9 })
    const r = trainAgent(env, agent, { episodes: 60, seed: 0, checkpoints: false })
    const greedy = evaluatePolicy(env, greedyPolicyOf(agent, r.agent), { episodes: 20, seed: 3 })
    expect(greedy.meanReturn).toBe(1)
    expect(mean(greedy.lengths)).toBeLessThan(30)
    expect(optionValues(r.agent)).toHaveLength(104)
  })

  it('acts with the running option, which learn records; options run until they terminate', () => {
    const env = makeFourRooms({ success: 1 })
    const options = hallwayOptions(env)
    const agent = smdpQLearning(env, { options })
    let s = agent.init(key(0))
    let obs = env.reset(key(1)).observation
    let state = obs
    let changes = 0
    for (let t = 0; t < 40; t++) {
      const a = agent.act(s, obs, key(100 + t))
      const step = env.step(state, a, key(200 + t))
      const before = s.decisions
      s = agent.learn(
        s,
        {
          observation: obs,
          action: a,
          reward: step.reward,
          nextObservation: step.observation,
          terminated: step.terminated,
          truncated: false,
        },
        key(300 + t),
      )
      const running = s.running
      // The recorded option is the one whose policy produced the action.
      if (running !== undefined) expect(options[running.option]?.policy(obs)[a]).toBe(1)
      if (s.decisions > before) changes += 1
      if (step.terminated) break
      obs = step.observation
      state = step.state
    }
    expect(changes).toBeGreaterThan(0)
    expect(agent.stats?.(s)['meanOptionLength']).toBeGreaterThan(1)
  })
})

describe('the SMDP and intra-option backups', () => {
  // A corridor of four states; "right" moves on and stops at state 3, "stay" is a primitive.
  const spaces = { observationSpace: discrete(4), actionSpace: discrete(2) }
  const right = {
    name: 'right',
    initiation: () => true,
    policy: () => [0, 1],
    termination: (s: number) => (s === 3 ? 1 : 0),
  }
  const [stay] = primitiveOptions(2)
  const step = (s: number, r: number) => ({
    observation: s,
    action: 1,
    reward: r,
    nextObservation: s + 1,
    terminated: false,
    truncated: false,
  })

  it('SMDP Q-learning backs up the whole option once it terminates', () => {
    const agent = smdpQLearning(spaces, { options: [right], alpha: 0.5, gamma: 0.9 })
    let s = agent.init(key(0))
    for (const [i, t] of [step(0, 1), step(1, 2), step(2, 4)].entries()) s = agent.learn(s, t, key(i))
    expect(s.running).toBeUndefined()
    expect(s.Q[0]).toBeCloseTo(0.5 * (1 + 0.9 * 2 + 0.81 * 4), 12)
    expect(s.Q[1]).toBe(0)
  })

  it('intra-option Q-learning updates every consistent option at every step', () => {
    const agent = intraOptionQLearning(spaces, { options: [right, stay as never], alpha: 0.5, gamma: 0.9 })
    const s = agent.learn(agent.init(key(0)), step(0, 1), key(1))
    // "right" took action 1 at state 0 and learns; the "stay" primitive (action 0) does not.
    expect(Array.from(s.Q.subarray(0, 2))).toEqual([0.5, 0])
  })
})

describe('option-critic (Bacon, Harb and Precup 2017)', () => {
  it('improves its returns on four rooms from reward alone', () => {
    const env = make(fourRooms)
    const agent = optionCritic(env)
    const r = trainAgent(env, agent, { episodes: 250, seed: 0, checkpoints: false })
    expect(mean(r.lengths.slice(-30))).toBeLessThan(mean(r.lengths.slice(0, 30)) / 2)
    // Discounted returns rise as episodes shorten.
    const discounted = r.lengths.map((n, i) => (r.returns[i] ?? 0) * 0.99 ** (n - 1))
    expect(mean(discounted.slice(-30))).toBeGreaterThan(mean(discounted.slice(0, 30)))
    const beta = optionTerminations(r.agent)
    expect(beta).toHaveLength(4)
    expect(beta[0]).toHaveLength(104)
  })

  it('a deliberation cost lowers terminations and lengthens options', () => {
    const env = make(fourRooms)
    // One step: the cost pushes the termination of the running option at the next state down.
    const t = { observation: 0, action: 1, reward: 0, nextObservation: 1, terminated: false, truncated: false }
    const varthetaAfter = (xi: number): number[] => {
      const agent = optionCritic(env, { deliberationCost: xi })
      const s0 = agent.init(key(0))
      const o = agent.learn(s0, t, key(1))
      return Array.from(o.vartheta)
    }
    const [plain, costly] = [varthetaAfter(0), varthetaAfter(0.1)]
    const changed = plain.findIndex((x, i) => x !== costly[i])
    expect(changed).toBeGreaterThanOrEqual(0)
    expect(costly[changed] ?? 0).toBeLessThan(plain[changed] ?? 0)
    // Over training, options run longer.
    const lengthOf = (xi: number): number => {
      const agent = optionCritic(env, { deliberationCost: xi })
      const r = trainAgent(env, agent, { episodes: 60, seed: 1, checkpoints: false })
      return agent.stats?.(r.agent)['meanOptionLength'] ?? 0
    }
    expect(lengthOf(0.1)).toBeGreaterThan(lengthOf(0) + 0.1)
  })

  it('rejects an empty option set', () => {
    expect(() => smdpQLearning(makeFourRooms(), { options: [] })).toThrow(/at least one option/)
    expect(() => hallwayOptions({ ...makeFourRooms(), model: undefined } as never)).toThrow(/no model/)
  })
})
