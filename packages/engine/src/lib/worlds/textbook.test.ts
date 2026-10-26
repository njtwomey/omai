// The textbook worlds reproduce the existing reinforcement learning environments exactly (tests may use higher
// layers): every outcome of every action, and the initial distributions. This is what lets those environments be
// rebuilt on worlds.
import { describe, expect, it } from 'vitest'
import {
  makeCliffWalking,
  makeFourRooms,
  makeFrozenLake,
  makeGridWorld,
  makeKeyMaze,
  makeWindyGridWorld,
  taxiDecode,
  taxiEncode,
  taxiTransitions,
} from '../environments/index.ts'
import { key } from '../random/index.ts'
import type { Env } from '../rl/index.ts'
import {
  aimaGridWorld,
  cliffWalkingWorld,
  destination,
  fourRoomsWorld,
  frozenLakeWorld,
  keyMazeWorld,
  lockedDoors,
  type MiniGridLayout,
  miniGridWorld,
  reachable,
  taxiWorld,
  transitionModel,
  type WorldTask,
  windyGridWorld,
} from './index.ts'

function sameDynamics(task: WorldTask, env: Env<number, number, number>): void {
  const m = transitionModel(task.world, task.rules, task.options)
  const model = env.model
  if (model === undefined) throw new Error('the environment has no model')
  expect(m.nStates).toBe(model.states.length)
  expect(m.nActions).toBe(model.actions.length)
  for (let s = 0; s < m.nStates; s++) {
    for (let a = 0; a < m.nActions; a++) {
      const ours = m.outcomes(s, a).map((o) => [o.probability, o.state, o.reward, o.terminated])
      const theirs = model.transitions(s, a).map((o) => [o.probability, o.state, o.reward, o.terminated])
      expect(ours.length, `state ${s} action ${a}`).toBe(theirs.length)
      for (const [i, o] of ours.entries()) {
        const t = theirs[i] ?? []
        expect(o[0], `state ${s} action ${a}`).toBeCloseTo(t[0] as number, 14)
        expect(o.slice(1), `state ${s} action ${a}`).toEqual(t.slice(1))
      }
    }
  }
  const initial = new Array<number>(m.nStates).fill(0)
  for (const { probability, state } of model.initial) initial[state] = probability
  for (let s = 0; s < m.nStates; s++) expect(m.initial[s]).toBeCloseTo(initial[s] ?? 0, 14)
}

describe('textbook worlds against the reinforcement learning environments', () => {
  it('FrozenLake: maps, slipperiness, success rates and reward schedules', () => {
    sameDynamics(frozenLakeWorld(), makeFrozenLake())
    sameDynamics(frozenLakeWorld({ mapName: '8x8' }), makeFrozenLake({ mapName: '8x8' }))
    sameDynamics(frozenLakeWorld({ isSlippery: false }), makeFrozenLake({ isSlippery: false }))
    sameDynamics(frozenLakeWorld({ successRate: 0.6 }), makeFrozenLake({ successRate: 0.6 }))
    sameDynamics(frozenLakeWorld({ rewardSchedule: [10, -5, -1] }), makeFrozenLake({ rewardSchedule: [10, -5, -1] }))
    sameDynamics(frozenLakeWorld({ desc: ['SFH', 'FFF', 'HFG'] }), makeFrozenLake({ desc: ['SFH', 'FFF', 'HFG'] }))
  })

  it('CliffWalking, dry and slippery', () => {
    sameDynamics(cliffWalkingWorld(), makeCliffWalking())
    sameDynamics(cliffWalkingWorld({ isSlippery: true }), makeCliffWalking({ isSlippery: true }))
  })

  it('the AIMA grid, four rooms and the key maze', () => {
    sameDynamics(aimaGridWorld(), makeGridWorld())
    sameDynamics(aimaGridWorld({ slip: 0, stepReward: -1 }), makeGridWorld({ slip: 0, stepReward: -1 }))
    sameDynamics(fourRoomsWorld(), makeFourRooms())
    sameDynamics(fourRoomsWorld({ goal: [3, 3], success: 0.5 }), makeFourRooms({ goal: [3, 3], success: 0.5 }))
    sameDynamics(keyMazeWorld(), makeKeyMaze())
    expect(() => fourRoomsWorld({ goal: [0, 0] })).toThrow()
  })

  it('the windy grid world with every option', () => {
    for (const kingsMoves of [false, true]) {
      for (const stay of [false, true]) {
        for (const stochasticWind of [false, true]) {
          sameDynamics(
            windyGridWorld({ kingsMoves, stay, stochasticWind }),
            makeWindyGridWorld({ kingsMoves, stay, stochasticWind }),
          )
        }
      }
    }
  })

  it("Taxi's map moves the taxi as gymnasium's table does", () => {
    const t = taxiWorld()
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const s = taxiEncode(row, col, 0, 1)
        for (let a = 0; a < 4; a++) {
          const [r2, c2] = taxiDecode(taxiTransitions(s, a)[0]?.state ?? s)
          const act = t.options.actions?.actions[a]
          const dir = act?.kind === 'move' ? act.direction : -1
          const to = destination(t.world, row * 5 + col, dir)
          expect(to < 0 ? row * 5 + col : to, `(${row}, ${col}) action ${a}`).toBe((r2 ?? 0) * 5 + (c2 ?? 0))
        }
      }
    }
  })
})

describe('MiniGrid-style layouts', () => {
  const layouts: readonly MiniGridLayout[] = ['empty', 'doorKey', 'fourRooms', 'lavaGap', 'crossing']
  it('are walled, keyed, and solvable once keys open their doors', () => {
    for (const layout of layouts) {
      for (let seed = 0; seed < 25; seed++) {
        const t = miniGridWorld(
          layout,
          key(seed),
          layout === 'crossing' ? { crossings: 1 + (seed % 3), obstacle: seed % 2 === 0 ? 'lava' : 'wall' } : {},
        )
        const w = t.world
        const size = w.topology.kind === 'square' ? w.topology.cols : 0
        for (let i = 0; i < size; i++) {
          for (const c of [i, (size - 1) * size + i, i * size, i * size + size - 1]) expect(w.terrain[c]).toBe(1)
        }
        const start = w.starts[0] ?? 0
        const goal = w.goals[0] ?? 0
        expect(start).not.toBe(goal)
        // Lava is never stepped on; doors open (keys are reachable on the start's side).
        const lava = (c: number): boolean => w.terrains[w.terrain[c] ?? 0]?.name === 'lava'
        const doors = new Set(w.items.filter((it) => it.kind === 'door').map((it) => it.cell))
        const keySide = reachable(w, start, { blocked: (c) => lava(c) || doors.has(c) })
        for (const it of w.items.filter((x) => x.kind === 'key')) expect(keySide[it.cell]).toBe(1)
        expect(reachable(w, start, { blocked: lava })[goal]).toBe(1)
        if (layout === 'doorKey') expect(keySide[goal]).toBe(0)
        expect(miniGridWorld(layout, key(seed), {}).world.terrain).toEqual(
          miniGridWorld(layout, key(seed), {}).world.terrain,
        )
      }
    }
    expect(() => miniGridWorld('fourRooms', key(0), { size: 5 })).toThrow()
  })

  it('turn, move forward and open their doors with the key', () => {
    const t = miniGridWorld('doorKey', key(3), { size: 6 })
    const m = transitionModel(t.world, t.rules, t.options)
    expect(m.nActions).toBe(3)
    expect(t.rules.some((r) => r.name === lockedDoors().name)).toBe(true)
    const start = m.state(Array.from(m.initial).indexOf(1))
    expect(start.heading).toBe(1)
  })
})
