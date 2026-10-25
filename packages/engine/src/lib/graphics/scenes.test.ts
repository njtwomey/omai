import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { pathTrace } from './pathtracing.ts'
import { rayScene } from './raytracing.ts'
import { cornellBox } from './scenes.ts'

const mean = (rgb: Float64Array): number => rgb.reduce((s, v) => s + v, 0) / rgb.length

describe('cornellBox', () => {
  it('builds the room, the lamp and either contents, with one area light', () => {
    const spheres = cornellBox()
    const blocks = cornellBox({ contents: 'blocks' })
    expect(spheres.scene.objects.map((o) => o.name)).toContain('glass sphere')
    expect(blocks.scene.objects.map((o) => o.name)).toContain('tall block')
    expect(spheres.scene.lights).toHaveLength(1)
    expect(spheres.scene.objects.filter((o) => o.lightSurface === true)).toHaveLength(1)
    expect(() => cornellBox({ lamp: 0 })).toThrow()
  })

  it('renders the red wall on the left, the green wall on the right and the lamp at the top', () => {
    const { scene, camera } = cornellBox({ contents: 'blocks' })
    const w = 16
    const { image } = pathTrace(scene, camera, w, w, { key: key(1), samples: 16, maxBounces: 3 })
    const px = (x: number, y: number): number[] => [0, 1, 2].map((c) => image.rgb[3 * (y * w + x) + c] ?? 0)
    const [lr, lg] = px(0, 8)
    const [rr, rg] = px(w - 1, 8)
    expect(lr).toBeGreaterThan(2 * (lg ?? 0))
    expect(rg).toBeGreaterThan(2 * (rr ?? 0))
    expect(px(8, 1)[0]).toBeGreaterThan(5)
  })

  it('counts a light surface once: next-event estimation agrees with hitting the lamp by chance', () => {
    const { scene, camera } = cornellBox({ contents: 'blocks' })
    // The same room with no area light and a plain emissive lamp: only paths that hit the panel see the light.
    const unlit = rayScene(
      scene.objects.map((o) => ({ shape: o.shape, material: o.material })),
      { lights: [] },
    )
    const options = { samples: 600, maxBounces: 3 }
    const withLight = pathTrace(scene, camera, 6, 6, { key: key(2), ...options })
    const byChance = pathTrace(unlit, camera, 6, 6, { key: key(3), ...options })
    expect(mean(withLight.image.rgb)).toBeCloseTo(mean(byChance.image.rgb), 1)
  })
})
