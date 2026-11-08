import { circle, group, path, rect, scene, sceneTimeline, structure, text, timeline } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import {
  easings,
  frameAt,
  interpolateScene,
  runTimeline,
  structureTimeline,
  sweepTimeline,
  tweenTimeline,
} from './timeline.ts'

describe('timeline', () => {
  it('eases from 0 to 1', () => {
    for (const e of Object.values(easings)) {
      expect(e(0)).toBeCloseTo(0)
      expect(e(1)).toBeCloseTo(1)
      expect(e(0.5)).toBeGreaterThan(0)
    }
  })
  it('interpolates matching nodes and switches the rest halfway', () => {
    const a = scene(10, 10, [
      rect(0, 0, 2, 2),
      path([
        [0, 0],
        [1, 1],
      ]),
      group([circle(0, 0, 1)], { translate: [0, 0] }),
      text(0, 0, 'a'),
    ])
    const b = scene(20, 10, [
      rect(2, 2, 4, 4),
      path([
        [2, 2],
        [3, 3],
      ]),
      group([circle(2, 0, 1)], { translate: [4, 2] }),
      circle(0, 0, 1),
    ])
    const m = interpolateScene(a, b, 0.5)
    expect(m.width).toBe(15)
    expect(m.children[0]).toMatchObject({ x: 1, width: 3 })
    expect(m.children[1]).toMatchObject({
      points: [
        [1, 1],
        [2, 2],
      ],
    })
    expect(m.children[2]).toMatchObject({ translate: [2, 1], children: [{ cx: 1 }] })
    expect(m.children[3]?.kind).toBe('circle')
    expect(interpolateScene(a, scene(1, 1, []), 0.2).children).toHaveLength(4)
  })
  it('tweens, sweeps and reads timelines', () => {
    const tl = sceneTimeline([scene(1, 1, [circle(0, 0, 1)]), scene(1, 1, [circle(1, 0, 1)])], { fps: 2, title: 't' })
    const tw = tweenTimeline(tl, 3, easings.linear)
    expect(tw.fps).toBe(6)
    expect(tw.frames.map((f) => f.step)).toEqual([0, 1, 2, 3])
    const mid = tw.frames[1]
    expect(mid?.kind === 'scene' ? mid.scene.children[0] : undefined).toMatchObject({ cx: 1 / 3 })
    expect(() => tweenTimeline(tl, 0)).toThrow()
    const s = structure('s', [], [])
    const mixed = timeline([
      { kind: 'structure', step: 0, structure: s, overlay: {} },
      ...tl.frames.map((f) => ({ ...f, step: f.step + 1 })),
    ])
    expect(tweenTimeline(mixed, 2).frames).toHaveLength(4)
    expect(
      sweepTimeline([1, 2], () => scene(1, 1, []), { caption: (v) => `r = ${v}`, title: 'sweep' }).frames[1]?.caption,
    ).toBe('r = 2')
    expect(structureTimeline(s, [{}], { captions: ['c'] }).frames[0]?.caption).toBe('c')
    expect(frameAt(timeline([]), 3)).toBe(0)
    const run = {
      schema: 'omai.run/1' as const,
      events: [
        { type: 'frame' as const, runId: 'r', step: 0, frame: { cells: [1] } },
        { type: 'frame' as const, runId: 'r', step: 1, frame: { other: true } },
      ],
    }
    const decoded = runTimeline(run, {
      decode: (p) => ('cells' in (p as object) ? scene(1, 1, []) : undefined),
      fps: 5,
    })
    expect(decoded.frames).toHaveLength(1)
    expect(decoded.fps).toBe(5)
  })
})
