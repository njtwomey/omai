import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { type PinholeCamera, pinholeCamera } from '../graphics/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { datasetRegistry } from './catalogue.ts'
import { hemisphereCameras, multiViewImages, multiViewScene, renderMultiView } from './multiview.ts'

describe('multiViewScene and renderMultiView', () => {
  it('renders the background where no ray hits and a shaded albedo where one does', () => {
    const scene = multiViewScene(1)
    const camera = pinholeCamera({ eye: [0, 0, 4], fov: 40 })
    const image = renderMultiView(scene, camera, 9, 9)
    const px = (i: number, j: number): number[] => Array.from(toFlat(image).slice(3 * (j * 9 + i), 3 * (j * 9 + i) + 3))
    expect(px(0, 0)).toEqual([1, 1, 1])
    // The red sphere (centre (-0.45, 0, 0), radius 0.45) is left of centre; its pixel is red-dominant and darker than its albedo.
    const hit = px(3, 4)
    expect(hit[0]).toBeGreaterThan((hit[1] ?? 0) * 2)
    expect(hit[0]).toBeLessThanOrEqual(0.9)
    expect(image.shape).toEqual([9, 9, 3])
  })

  it('shades by Lambert’s law with an ambient floor, and puts shadows on the sphere’s far side', () => {
    const scene = multiViewScene(1)
    // Looking along the light direction, the lit side faces the camera; looking against it, the camera sees the dark side.
    const toLight = scene.light
    const view = (sign: number): number => {
      const eye: [number, number, number] = [
        -0.45 + 4 * sign * toLight[0],
        4 * sign * toLight[1],
        4 * sign * toLight[2],
      ]
      const cam = pinholeCamera({ eye, target: [-0.45, 0, 0], up: [0, 1, 0], fov: 20 })
      const v = toFlat(renderMultiView(scene, cam, 5, 5))
      return v[3 * 12] as number
    }
    const lit = view(1)
    const dark = view(-1)
    expect(lit).toBeCloseTo(0.9, 2)
    expect(dark).toBeCloseTo(0.9 * scene.ambient, 2)
  })
})

describe('hemisphereCameras', () => {
  it('puts the cameras on a sphere above the scene, looking at the origin, spread in azimuth', () => {
    const cams = hemisphereCameras(key(0), 8, { radius: 4 })
    expect(cams.length).toBe(8)
    for (const c of cams) {
      expect(Math.hypot(...c.eye)).toBeCloseTo(4, 12)
      expect(c.eye[1]).toBeGreaterThan(0)
      expect(c.target).toEqual([0, 0, 0])
    }
    const azimuths = cams.map((c) => Math.atan2(c.eye[2], c.eye[0]))
    expect(new Set(azimuths.map((a) => Math.round((a * 4) / Math.PI))).size).toBeGreaterThanOrEqual(6)
  })
})

describe('multiViewImages', () => {
  const g = generate(multiViewImages, { key: key(0), knobs: { views: 5, resolution: 8, heldOut: 2, objects: 3 } })
  const extra = g.truth.extra as Record<string, unknown>

  it('has the training images, their cameras and the held-out views, with the scene as truth', () => {
    expect(g.data.images.shape).toEqual([5, 8, 8, 3])
    expect((extra['cameras'] as unknown[]).length).toBe(5)
    const held = extra['heldOut'] as { cameras: unknown[]; images: Tensor }
    expect(held.cameras.length).toBe(2)
    expect(held.images.shape).toEqual([2, 8, 8, 3])
    expect(extra['near']).toBe(2)
    expect(extra['far']).toBe(6)
  })

  it('can render any view exactly: its own views are reproduced, and a novel camera is rendered', () => {
    const render = extra['renderView'] as (c: PinholeCamera, w: number, h: number) => Tensor
    const cams = extra['cameras'] as PinholeCamera[]
    expect(Array.from(toFlat(render(cams[2] as PinholeCamera, 8, 8)))).toEqual(
      Array.from(toFlat(g.data.images).slice(2 * 192, 3 * 192)),
    )
    const novel = render(pinholeCamera({ eye: [3, 2, 1], fov: 40 }), 12, 6)
    expect(novel.shape).toEqual([6, 12, 3])
    expect(Array.from(toFlat(novel)).some((v) => v < 0.95)).toBe(true)
  })

  it('is deterministic in the key, white where nothing is hit, and registered as an image dataset', () => {
    const again = generate(multiViewImages, { key: key(0), knobs: { views: 5, resolution: 8, heldOut: 2, objects: 3 } })
    expect(Array.from(toFlat(again.data.images))).toEqual(Array.from(toFlat(g.data.images)))
    expect(Array.from(toFlat(g.data.images)).some((v) => v === 1)).toBe(true)
    const entry = datasetRegistry.entries.find((e) => e.name === 'multiViewImages')
    expect(entry?.kind).toBe('image')
    expect(entry?.modality).toBe('image')
  })
})
