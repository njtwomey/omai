import { describe, expect, it } from 'vitest'
import type { Vec3 } from '../base/index.ts'
import {
  bakeTexture,
  checkerboard,
  directionalLight,
  isBackFacing,
  material,
  mesh,
  orthographicCamera,
  pinholeCamera,
  rasteriseMesh,
} from './index.ts'

// A unit cube centred at the origin, faces wound counter-clockwise seen from outside.
function cube(): ReturnType<typeof mesh> {
  const p: Vec3[] = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
  ]
  const quads: [number, number, number, number][] = [
    [4, 5, 6, 7],
    [1, 0, 3, 2],
    [5, 1, 2, 6],
    [0, 4, 7, 3],
    [7, 6, 2, 3],
    [0, 1, 5, 4],
  ]
  return mesh(
    p,
    quads.flatMap(([a, b, c, d]) => [[a, b, c] as const, [a, c, d] as const]),
  )
}

describe('rasterising meshes', () => {
  const cam = pinholeCamera({ eye: [3, 2, 4] })

  it('culls exactly the faces turned away and draws the same image either way for a closed mesh', () => {
    const m = cube()
    const back = rasteriseMesh(m, cam, { width: 24, height: 18 })
    const none = rasteriseMesh(m, cam, { width: 24, height: 18, cull: 'none' })
    const front = rasteriseMesh(m, cam, { width: 24, height: 18, cull: 'front' })
    // Seen from (3, 2, 4) three faces (six triangles) of the cube face the eye.
    expect(back.triangles.filter((t) => t.culled)).toHaveLength(6)
    expect(front.triangles.filter((t) => t.culled)).toHaveLength(6)
    expect(Array.from(back.triangleIds)).toEqual(Array.from(none.triangleIds))
    for (const t of back.triangles) {
      const [a, b, c] = (m.triangles[t.index] ?? [0, 0, 0]).map((i) => m.positions[i] as Vec3) as [Vec3, Vec3, Vec3]
      expect(t.culled).toBe(isBackFacing(cam, a, b, c))
    }
    // With front faces culled the far, inside faces show: farther everywhere they cover.
    for (let i = 0; i < 24 * 18; i++) {
      if ((front.triangleIds[i] ?? -1) >= 0) expect(front.depth[i] ?? 0).toBeGreaterThanOrEqual(back.depth[i] ?? 0)
    }
    expect(back.triangles.reduce((s, t) => s + t.written, 0)).toBeGreaterThan(0)
  })

  it('clips a mesh that surrounds the camera instead of wrapping it around the screen', () => {
    const inside = rasteriseMesh(cube(), pinholeCamera({ eye: [0, 0, 0.2], target: [0, 0, -1], near: 0.05 }), {
      width: 16,
      height: 12,
      cull: 'none',
    })
    // From inside the cube, every pixel sees a wall.
    expect(Array.from(inside.triangleIds).every((id) => id >= 0)).toBe(true)
    expect(inside.triangles.some((t) => t.clippedVertices > 3)).toBe(true)
    const behind = rasteriseMesh(
      mesh(
        [
          [0, 0, 10],
          [1, 0, 10],
          [0, 1, 10],
        ],
        [[0, 1, 2]],
      ),
      cam,
      { width: 16, height: 12, cull: 'none' },
    )
    expect(behind.triangles[0]?.clippedVertices).toBe(0)
    expect(Array.from(behind.triangleIds).every((id) => id === -1)).toBe(true)
  })

  it('textures and shades the visible surface, orthographic or perspective', () => {
    const quad = mesh(
      [
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
      ],
      [
        [0, 1, 2],
        [0, 2, 3],
      ],
      {
        uvs: [
          [0, 1],
          [1, 1],
          [1, 0],
          [0, 0],
        ],
      },
    )
    const tex = bakeTexture(8, 8, (u, v) => checkerboard(u, v, { scale: 2, a: [1, 1, 1], b: [0, 0, 0] }))
    const ortho = orthographicCamera({ eye: [0, 0, 2], height: 1 })
    const r = rasteriseMesh(quad, ortho, { width: 8, height: 8, texture: tex, background: [0.5, 0.5, 0.5] })
    // The quad fills the view; its four quadrants alternate white and black.
    expect(r.image.rgb[0]).toBeCloseTo(1, 9)
    expect(r.image.rgb[3 * 7]).toBeCloseTo(0, 9)
    expect(r.image.rgb[3 * 63]).toBeCloseTo(1, 9)
    const lit = rasteriseMesh(quad, ortho, {
      width: 8,
      height: 8,
      shading: {
        material: material({ albedo: [1, 1, 1] }),
        lights: [directionalLight([0, 0, -1], [Math.PI, Math.PI, Math.PI])],
      },
    })
    for (let i = 0; i < 64; i++) expect(lit.image.rgb[3 * i]).toBeCloseTo(1, 12)
    const persp = rasteriseMesh(quad, pinholeCamera({ eye: [0, 0, 3] }), {
      width: 8,
      height: 8,
      background: [0.5, 0, 0],
    })
    expect(persp.image.rgb[0]).toBe(0.5)
    expect(persp.image.rgb[3 * 36]).toBe(1)
  })

  it('refuses bad meshes', () => {
    expect(() => mesh([[0, 0, 0]], [[0, 1, 2]])).toThrow()
    expect(() =>
      mesh(
        [
          [0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
        ],
        [[0, 1, 2]],
        { colours: [[1, 0, 0]] },
      ),
    ).toThrow()
  })
})
