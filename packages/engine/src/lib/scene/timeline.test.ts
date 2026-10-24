import { describe, expect, it } from 'vitest'
import { scene } from './build.ts'
import { structure } from './structure.ts'
import { isTimeline, sceneTimeline, timeline } from './timeline.ts'

describe('timeline', () => {
  it('sorts frames by step and keeps structure overlays', () => {
    const s = structure('x', [{ id: 'a', label: 'a', role: 'input' }], [])
    const t = timeline(
      [
        { kind: 'structure', step: 2, structure: s, overlay: { active: ['a'] }, caption: 'later' },
        { kind: 'scene', step: 0, scene: scene(1, 1, []) },
      ],
      { fps: 4, title: 'demo' },
    )
    expect(t.frames.map((f) => f.step)).toEqual([0, 2])
    expect(t).toMatchObject({ fps: 4, title: 'demo' })
    expect(isTimeline(t)).toBe(true)
    expect(isTimeline({})).toBe(false)
  })

  it('numbers scene sequences from zero at the default rate', () => {
    const t = sceneTimeline([scene(1, 1, []), scene(1, 1, []), scene(1, 1, [])])
    expect(t.frames.map((f) => f.step)).toEqual([0, 1, 2])
    expect(t.fps).toBe(10)
  })
})
