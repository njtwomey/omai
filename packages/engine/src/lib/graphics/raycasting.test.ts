import { describe, expect, it } from 'vitest'
import { policies, recorder } from '../trace/index.ts'
import { fromAscii, world } from '../worlds/index.ts'
import { castColumns, columnAngles, ddaCast, floatImage, renderRaycast, viewConeRays, viewConeScene } from './index.ts'

const room = fromAscii(['#######', '#.....#', '#.....#', '#.....#', '#######'])
const pose = { x: 1.5, y: 2.5, heading: 0 }

describe('grid ray casting', () => {
  it('stops at thin walls, the map edge and when it starts inside a wall', () => {
    const corridor = world({ topology: { kind: 'square', rows: 1, cols: 3 }, walls: [[1, 2]] })
    const thin = ddaCast(corridor, [0.5, 0.5], 0)
    expect([thin.hit?.distance, thin.hit?.cell, thin.hit?.thin, thin.hit?.face, thin.cells]).toEqual([
      1.5,
      2,
      true,
      3,
      [0, 1],
    ])
    const back = ddaCast(corridor, [0.5, 0.5], Math.PI)
    expect([back.hit?.distance, back.hit?.cell, back.hit?.thin]).toEqual([0.5, -1, false])
    expect(ddaCast(room, [0.5, 0.5], 0).hit?.distance).toBe(0)
    expect(ddaCast(room, [-1, 2], 0).hit?.cell).toBe(-1)
    expect(ddaCast(room, [1.5, 2.5], 0, { maxDistance: 2 }).hit).toBeNull()
    expect(ddaCast(room, [1.5, 2.5], 0, { maxSteps: 2 }).hit).toBeNull()
    // A non-finite direction cannot loop: it stops at once without a hit.
    expect(ddaCast(room, [1.5, 2.5], Number.NaN).hit).toBeNull()
    expect(() => ddaCast(fromAscii(['...'], { topology: 'hex' }), [0.5, 0.5], 0)).toThrow()
  })

  it('reports texture columns left to right as the viewer sees each face', () => {
    // Facing east, the wall's u grows with y (to the viewer's right); facing west, it falls.
    const east = ddaCast(room, [1.5, 2.25], 0).hit
    const west = ddaCast(room, [4.5, 2.25], Math.PI).hit
    expect(east?.u).toBeCloseTo(0.25, 12)
    expect(west?.u).toBeCloseTo(0.75, 12)
    const south = ddaCast(room, [2.25, 1.5], Math.PI / 2).hit
    const north = ddaCast(room, [2.25, 3.5], -Math.PI / 2).hit
    expect([south?.side, south?.face, north?.face]).toEqual(['y', 0, 2])
    expect(south?.u).toBeCloseTo(0.75, 9)
    expect(north?.u).toBeCloseTo(0.25, 9)
  })

  it('corrects the fish-eye so a flat wall has one depth across the view', () => {
    const cols = castColumns(room, pose, { width: 16, height: 12, fov: Math.PI / 4 })
    for (const c of cols) {
      expect(c.hit?.cell).toBeGreaterThanOrEqual(0)
      if (c.hit?.side === 'x') expect(c.perpendicular).toBeCloseTo(4.5, 9)
    }
    // Wall slices are symmetric about the horizon when the eye is at half height.
    const mid = cols[8]
    expect(((mid?.top ?? 0) + (mid?.bottom ?? 0)) / 2).toBeCloseTo(6, 12)
    expect(columnAngles(0, 3, Math.PI / 2)[1]).toBe(0)
  })

  it('casts as a traced algorithm with one step and one hit message per column', () => {
    const rec = recorder()
    castColumns(room, pose, { width: 10, tracer: rec, policy: policies.teaching })
    expect(rec.events.filter((e) => e.type === 'message' && e.kind === 'hit')).toHaveLength(10)
    expect(rec.events.filter((e) => e.type === 'metric' && e.name === 'perpendicular').length).toBeGreaterThanOrEqual(
      10,
    )
  })

  it('renders textured walls, floors and ceilings, fog and depth-sorted sprites', () => {
    const tex = floatImage(4, 4, (x) => [x / 3, 0, 0])
    const f = renderRaycast(room, pose, {
      width: 24,
      height: 16,
      walls: { wall: tex },
      floor: [0, 1, 0],
      ceiling: [0, 0, 1],
      sideShade: 0.5,
    })
    expect(Array.from(f.image.rgb.slice(0, 3))).toEqual([0, 0, 1])
    expect(Array.from(f.image.rgb.slice(3 * 24 * 15, 3 * 24 * 15 + 3))).toEqual([0, 1, 0])
    // Across the far wall the texture column (red) grows left to right within each face.
    const row = 8
    const red = Array.from({ length: 24 }, (_, x) => f.image.rgb[3 * (row * 24 + x)] ?? 0)
    expect(red.some((v) => v > 0)).toBe(true)
    expect(f.depth).toHaveLength(24)
    // A sprite behind a wall is hidden; one in the open is drawn; the nearer of two overlapping sprites wins.
    const sprites = renderRaycast(room, pose, {
      width: 24,
      height: 16,
      floor: [0, 0, 0],
      sprites: [
        { x: 4, y: 2.5, colour: [1, 0, 0] },
        { x: 3, y: 2.5, colour: [0, 1, 1] },
        { x: 9, y: 2.5, colour: [1, 1, 1] },
      ],
    })
    expect(sprites.sprites.map((s) => s.index)).toEqual([2, 0, 1])
    expect(sprites.sprites.map((s) => s.visible > 0)).toEqual([false, true, true])
    const centre = 3 * (9 * 24 + 12)
    expect(Array.from(sprites.image.rgb.slice(centre, centre + 3))).toEqual([0, 1, 1])
    const masked = renderRaycast(room, pose, {
      width: 24,
      height: 16,
      sprites: [
        { x: 3, y: 2.5, texture: { ...floatImage(2, 2, () => [1, 1, 1]), alpha: Float64Array.from([0, 0, 0, 0]) } },
      ],
    })
    const plain = renderRaycast(room, pose, { width: 24, height: 16 })
    expect(masked.image.rgb).toEqual(plain.image.rgb)
    const foggy = renderRaycast(room, pose, { width: 24, height: 16, fog: { colour: [1, 1, 1], density: 10 } })
    expect(foggy.image.rgb[3 * (8 * 24 + 12)]).toBeGreaterThan(0.999)
    const terrain = fromAscii(['#####', '#.~.#', '#####'])
    const water = renderRaycast(
      terrain,
      { x: 1.5, y: 1.5, heading: 0 },
      { width: 8, height: 8, floors: { water: [0, 0, 1] } },
    )
    expect(water.image.rgb.some((v, i) => i % 3 === 2 && v === 1)).toBe(true)
  })

  it('draws the view cone and the highlighted ray’s DDA cells from above', () => {
    const rays = viewConeRays(room, pose, { rays: 5, fov: Math.PI / 2 })
    expect(rays).toHaveLength(5)
    for (const r of rays) expect(r.cells[0]).toBe(2 * 7 + 1)
    const s = viewConeScene(room, pose, { rays: 5, cell: 10, highlight: 2 })
    expect([s.width, s.height]).toEqual([70, 50])
    const cells = s.children.filter((n) => n.id?.startsWith('dda-cell-')).length
    expect(cells).toBe(rays[2]?.cells.length)
    expect(s.children.some((n) => n.id === 'cone')).toBe(true)
    expect(viewConeScene(room, pose, { highlight: -1 }).children.some((n) => n.id?.startsWith('dda-cell-'))).toBe(false)
  })
})
