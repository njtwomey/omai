import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  achievedGoal,
  desiredGoal,
  goalReached,
  goalReward,
  makeReachGoal,
  reachGoalOptimalSteps,
} from './goal-reaching.ts'

describe('the reaching task', () => {
  it('observes the position then the goal and pays a sparse reward', () => {
    const env = makeReachGoal()
    const r = env.reset(key(3))
    expect(r.observation).toHaveLength(4)
    expect(achievedGoal(r.observation)).toEqual([...r.state.position])
    expect(desiredGoal(r.observation)).toEqual([...r.state.goal])
    const s = env.step({ position: [0.5, 0.5], goal: [0.9, 0.9], t: 0 }, 1, key(0))
    expect(s.reward).toBe(-1)
    expect(s.state.position[0]).toBeCloseTo(0.6, 12)
    const done = env.step({ position: [0.5, 0.5], goal: [0.62, 0.5], t: 0 }, 1, key(0))
    expect([done.reward, done.terminated]).toEqual([0, true])
  })

  it('stays in the square and recomputes rewards for any goal', () => {
    const env = makeReachGoal()
    expect(env.step({ position: [1, 1], goal: [0, 0], t: 0 }, 1, key(0)).state.position).toEqual([1, 1])
    expect(goalReward([0.2, 0.2], [0.21, 0.2])).toBe(0)
    expect(goalReward([0.2, 0.2], [0.5, 0.5])).toBe(-1)
    expect(goalReached([0, 0], [0.04, 0])).toBe(true)
  })

  it('knows the fewest moves to a goal, which a greedy walk achieves', () => {
    const options = { stepSize: 0.1, tolerance: 0.05 }
    expect(reachGoalOptimalSteps([0.5, 0.5], [0.8, 0.5], options)).toBe(3)
    expect(reachGoalOptimalSteps([0.5, 0.5], [0.5, 0.5], options)).toBe(0)
    const env = makeReachGoal(options)
    const state0 = { position: [0.2, 0.3] as const, goal: [0.7, 0.9] as const, t: 0 }
    let s = state0
    let steps = 0
    while (!goalReached(s.position, s.goal, 0.05) && steps < 50) {
      const dx = s.goal[0] - s.position[0]
      const dy = s.goal[1] - s.position[1]
      const a = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : dy > 0 ? 0 : 2
      s = env.step(s, a, key(steps)).state
      steps += 1
    }
    expect(steps).toBeLessThanOrEqual(reachGoalOptimalSteps(state0.position, state0.goal, options) + 1)
  })
})
