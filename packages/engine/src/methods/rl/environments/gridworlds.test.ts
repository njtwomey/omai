import { describe, expect, it } from 'vitest'
import { maxOf } from '../../../lib/base/index.ts'
import { key } from '../../../lib/random/index.ts'
import { type Env, listEnvironments, mdpFromEnv, valueIteration } from '../../../lib/rl/index.ts'
import { contains } from '../../../lib/space/index.ts'
import * as environments from '../index.ts'
import {
  frozenLake8x8,
  makeFourRooms,
  makeFrozenLake,
  makeGridWorld,
  makeKeyMaze,
  randomFrozenLakeMap,
} from './gridworlds.ts'

function checkModel(env: Env<number, number, number>): void {
  const model = env.model
  if (model === undefined) throw new Error('no model')
  for (const s of model.states) {
    for (const a of model.actions) {
      const total = model.transitions(s, a).reduce((acc, o) => acc + o.probability, 0)
      expect(Math.abs(total - 1)).toBeLessThan(1e-12)
    }
  }
  expect(Math.abs(model.initial.reduce((acc, x) => acc + x.probability, 0) - 1)).toBeLessThan(1e-12)
}

describe('every registered environment', () => {
  it('resets, steps, stays in its spaces and renders', () => {
    for (const def of listEnvironments(environments)) {
      const env = def.create() as Env<unknown, unknown, unknown>
      const r = env.reset(key(1))
      expect(contains(env.observationSpace, r.observation as never), def.id).toBe(true)
      const space = env.actionSpace
      const action =
        space.kind === 'discrete' ? 0 : space.kind === 'box' ? new Array<number>(space.shape[0] ?? 1).fill(0) : 0
      const s = env.step(r.state, action, key(2))
      expect(typeof s.reward, def.id).toBe('number')
      const pic = env.render(s.state)
      expect(pic.kind, def.id).toBe('scene')
      expect(pic.width, def.id).toBeGreaterThan(0)
      if (env.model !== undefined) checkModel(env as Env<number, number, number>)
    }
  })
})

describe('grid worlds', () => {
  it('solve FrozenLake 8x8 close to its known optimum', () => {
    const env = frozenLake8x8.create()
    const v = valueIteration(mdpFromEnv(env, 1), { tol: 1e-10, maxIter: 20000 })
    // gymnasium's registry notes an optimum of 0.91 for the 8 x 8 lake (undiscounted success probability).
    expect(v.V[0]).toBeGreaterThan(0.9)
    expect(v.V[0]).toBeLessThanOrEqual(1)
  })

  it('build random FrozenLake maps that always have a path', () => {
    for (let i = 0; i < 5; i++) {
      const map = randomFrozenLakeMap(key(i), 6, 0.6)
      expect(map[0]?.[0]).toBe('S')
      expect(map[5]?.[5]).toBe('G')
      const env = makeFrozenLake({ desc: map, isSlippery: false })
      expect(valueIteration(mdpFromEnv(env, 1)).V[0]).toBe(1)
    }
    expect(() => makeFrozenLake({ desc: ['FF', 'FG'] })).toThrow(/start/)
  })

  it('solve the AIMA grid world with its well-known utilities', () => {
    const v = valueIteration(mdpFromEnv(makeGridWorld(), 1)).V
    // Russell and Norvig, Figure 17.3: U(1,1) = 0.705 (bottom left) and U(3,3) = 0.918 (left of the +1). AIMA also
    // charges R(s) = -0.04 in the state itself, while environments pay on arrival, so our values are 0.04 higher.
    expect(v[8]).toBeCloseTo(0.7053 + 0.04, 3)
    expect(v[2]).toBeCloseTo(0.9178 + 0.04, 3)
    expect(() => makeGridWorld({ map: ['..G'] })).toThrow(/start/)
  })

  it('needs the key to cross the door of the key maze', () => {
    const env = makeKeyMaze()
    const v = valueIteration(mdpFromEnv(env, 1)).V
    // 16 moves (fetch the key, return, pass the door): 15 step costs then the goal.
    expect(v[10]).toBeCloseTo(1 - 15 * 0.01, 12)
    const noKey = makeKeyMaze({ map: ['#####', '#SDG#', '#####'] })
    expect(valueIteration(mdpFromEnv(noKey, 0.9)).V[6]).toBeLessThan(0)
  })

  it('lays out the four rooms with 104 free cells and four hallways', () => {
    const env = makeFourRooms()
    expect(env.model?.states.length).toBe(104)
    const v = valueIteration(mdpFromEnv(env, 0.9)).V
    expect(maxOf(v.slice(0, 10))).toBeGreaterThan(0)
    expect(() => makeFourRooms({ goal: [0, 0] })).toThrow(/free/)
  })

  it('draws its world with an agent marker', () => {
    const s = makeFrozenLake().render(5)
    expect([s.width, s.height, s.title]).toEqual([256, 256, 'FrozenLake'])
    expect(s.children.some((c) => c.id === 'agent')).toBe(true)
    expect(makeKeyMaze().render(10 + 54).title).toBe('KeyMaze (key held)')
  })
})
