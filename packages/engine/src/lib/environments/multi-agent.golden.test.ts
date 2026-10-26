import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { key } from '../random/index.ts'
import { tensor } from '../tensor/index.ts'
import { cooperativeNavigation, type NavigationState } from './multi-agent.ts'

describe('cooperative navigation against the MPE transcription', () => {
  for (const c of loadFixture('methods/rl/environments/multi_agent').cases) {
    it(c.name, () => {
      const positions = c.params['positions'] as [number, number][]
      const n = positions.length
      const env = cooperativeNavigation.create({ agents: n })
      let s: NavigationState = {
        positions,
        velocities: positions.map(() => [0, 0]),
        landmarks: c.params['landmarks'] as [number, number][],
        t: 0,
      }
      const rows: number[][] = []
      for (const acts of c.params['actions'] as number[][]) {
        const r = env.step(s, Object.fromEntries(acts.map((a, i) => [`agent_${i}`, a])), key(0))
        s = r.state
        rows.push([
          ...s.positions.flat(),
          ...s.velocities.flat(),
          ...Array.from({ length: n }, (_, i) => r.rewards[`agent_${i}`] ?? 0),
        ])
      }
      expectMatches(tensor(rows), c.output, c.rtol, c.atol, c.name)
    })
  }
})
