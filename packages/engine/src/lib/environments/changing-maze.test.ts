import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { changingMazeTruth, makeChangingMaze } from './changing-maze.ts'

describe('the changing mazes', () => {
  it('have the shortest paths of the books before and after the change', () => {
    expect(changingMazeTruth('blocking').shortest).toEqual([10, 16])
    expect(changingMazeTruth('shortcut').shortest).toEqual([16, 10])
    expect(changingMazeTruth('blocking').changeAt).toBe(1000)
    expect(changingMazeTruth('shortcut').changeAt).toBe(3000)
  })

  it('are solved in exactly the shortest number of steps by following the optimal actions of the phase', () => {
    for (const kind of ['blocking', 'shortcut'] as const) {
      for (const phase of [0, 1] as const) {
        const truth = changingMazeTruth(kind, phase === 0 ? 1_000_000 : 0)
        const env = makeChangingMaze({ kind, changeAt: truth.changeAt })
        let r = env.reset(key(0))
        let steps = 0
        let done = false
        while (!done && steps < 100) {
          const a = truth.optimalActions(phase, r.state.cell)[0] ?? 0
          const s = env.step(r.state, a, key(steps))
          r = { state: s.state, observation: s.observation, info: s.info }
          done = s.terminated
          steps += 1
        }
        expect(done, `${kind} ${phase}`).toBe(true)
        expect(steps, `${kind} ${phase}`).toBe(truth.shortest[phase])
      }
    }
  })

  it('flag the change once, keep the clock across episodes and close the old gap', () => {
    const env = makeChangingMaze({ kind: 'blocking', changeAt: 3 })
    let r = env.reset(key(0))
    const changed: number[] = []
    for (let t = 0; t < 6; t++) {
      const s = env.step(r.state, 3, key(t)) // press left against the maze's edge
      changed.push(Number(s.info['changed']))
      r = { state: s.state, observation: s.observation, info: s.info }
    }
    expect(changed).toEqual([0, 0, 1, 0, 0, 0])
    expect(r.info['phase']).toBe(1)
    expect(env.resetFrom?.(r.state, key(1)).state.t).toBe(6)
    expect(env.reset(key(1)).state.t).toBe(0)
    // Before the change the gap at the right end admits (3, 8) -> (2, 8); after it, that cell is a wall.
    const gap = { cell: 3 * 9 + 8, t: 0 }
    expect(env.step(gap, 0, key(0)).state.cell).toBe(2 * 9 + 8)
    expect(env.step({ cell: gap.cell, t: 3 }, 0, key(0)).state.cell).toBe(gap.cell)
  })
})
