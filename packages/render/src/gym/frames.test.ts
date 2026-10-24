import { describe, expect, it } from 'vitest'
import { scene } from 'whitebox-engine/scene'
import { episodeSteps, episodeTimeline, filmstrip, stepCaption } from './frames.ts'

const episode = {
  observations: [0, 1, 2],
  states: [0, 1, 2],
  actions: [1, 0],
  rewards: [1, 0.5],
  infos: [{}, {}],
  return: 1.5,
  length: 2,
  terminated: false,
  truncated: true,
  frames: [scene(10, 10, []), scene(10, 10, []), scene(10, 10, [])],
}

describe('episode frames', () => {
  it('lists the steps with running returns and how the episode ended', () => {
    const steps = episodeSteps(episode)
    expect(steps.map((s) => s.return)).toEqual([0, 1, 1.5])
    expect(steps[2]?.end).toBe('truncated')
    expect(stepCaption(steps[0])).toBe('t = 0 · reset')
    expect(stepCaption(steps[2])).toBe('t = 2 · action 0 · reward 0.5 · return 1.5 (truncated)')
  })

  it('makes a captioned timeline and a filmstrip', () => {
    const t = episodeTimeline(episode, { fps: 5 })
    expect(t.frames).toHaveLength(3)
    expect(t.fps).toBe(5)
    expect(filmstrip(episode.frames, { columns: 2, cell: [20, 20] }).width).toBe(40)
  })
})
