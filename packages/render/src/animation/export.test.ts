import { describe, expect, it } from 'vitest'
import { circle, rect, scene, structure, timeline } from 'whitebox-engine/scene'
import { rasteriseScene, timelineFrames, timelineScenes } from './frames.ts'
import { animatedSvg } from './svg.ts'

const dots = [3, 5, 7].map((x) =>
  scene(10, 10, [rect(0, 0, 10, 10, { fill: '#000000' }), circle(x, 5, 2, { fill: '#ff0000' })]),
)

describe('animated SVG', () => {
  it('holds every frame in one document, each shown for its share of the cycle', () => {
    const svg = animatedSvg(dots, { fps: 3, width: 40 })
    expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"')).toBe(true)
    expect(svg).toContain('animation:wb-show 1s step-end infinite')
    expect(svg).toContain('33.3333%{visibility:hidden}')
    expect(svg).toContain('.wb-f2{animation-delay:0.6667s}')
    expect(svg.match(/<g class="wb-f wb-f\d+">/g)).toHaveLength(3)
    expect(animatedSvg(dots, { once: true })).toContain('animation:wb-show 0.3s step-end 1')
    expect(() => animatedSvg([])).toThrow(/no frames/)
  })

  it('draws structure timelines with a steady layout', () => {
    const s = structure(
      'tree',
      ['a', 'b'].map((id) => ({ id, label: id, role: 'internal' as const })),
      [{ from: 'a', to: 'b' }],
    )
    const tl = timeline([
      { kind: 'structure', step: 0, structure: s, overlay: { active: ['a'] } },
      { kind: 'structure', step: 1, structure: s, overlay: { active: ['b'], badges: { b: { value: 12.5 } } } },
    ])
    const [first, second] = timelineScenes(tl)
    expect([first?.width, first?.height]).toEqual([second?.width, second?.height])
    expect(animatedSvg(tl)).toContain('wb-f1')
  })
})

describe('frames', () => {
  it('rasterises scenes in Node with the software rasteriser, at the asked width', async () => {
    const px = await rasteriseScene(dots[0] ?? scene(1, 1, []), { width: 20 })
    expect([px.width, px.height]).toEqual([20, 20])
    // The red dot's centre at (3, 5) in scene units, (6, 10) in pixels.
    const at = 4 * (10 * 20 + 6)
    expect([px.rgba[at], px.rgba[at + 1], px.rgba[at + 2]]).toEqual([255, 0, 0])
    const frames = await timelineFrames(timeline(dots.map((sc, step) => ({ kind: 'scene' as const, step, scene: sc }))))
    expect(frames).toHaveLength(3)
    expect(frames.every((f) => f.width === 10 && f.height === 10)).toBe(true)
  })
})
