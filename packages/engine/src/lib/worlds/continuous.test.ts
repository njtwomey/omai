import { describe, expect, it } from 'vitest'
import {
  castRay,
  distanceToWalls,
  fieldOfView,
  fromAscii,
  moveCircle,
  stepPose,
  wallDistanceField,
  wallSegments,
  world,
  worldGeometry,
} from './index.ts'

describe('continuous worlds', () => {
  it('builds solid and opaque segments, with thin walls and see-through but solid cells kept apart', () => {
    const w = fromAscii(['#####', '#.#%#', '#####'], { format: 'walls' })
    expect(wallSegments(w)).toHaveLength(7)
    // Forest blocks sight but not motion: its sides join the opaque set.
    expect(wallSegments(w, 'opaque')).toHaveLength(4)
    expect(() => worldGeometry(world({ topology: { kind: 'graph', n: 1, neighbours: [[]] } }))).toThrow()
  })

  it('stops without sliding, slides along walls, and refuses negative radii', () => {
    const g = worldGeometry(fromAscii(['...', '...']))
    const stop = moveCircle(g, [0.5, 0.5], [2, -2], 0.25, { slide: false })
    expect(stop.collided).toBe(true)
    expect(stop.position[1]).toBeCloseTo(0.25, 6)
    expect(stop.path).toHaveLength(2)
    const slide = moveCircle(g, [0.5, 0.5], [2, -2], 0.25)
    expect(slide.position[0]).toBeCloseTo(2.5, 6)
    expect(slide.position[1]).toBeCloseTo(0.25, 6)
    expect(slide.normal?.[1]).toBeCloseTo(1, 12)
    expect(() => moveCircle(g, [0.5, 0.5], [1, 0], -1)).toThrow()
    const corner = moveCircle(g, [0.5, 0.5], [-3, -3], 0.25)
    expect(corner.position[0]).toBeCloseTo(0.25, 6)
    expect(corner.position[1]).toBeCloseTo(0.25, 6)
    expect(moveCircle(g, [1, 1], [0, 0], 0.2)).toEqual({
      position: [1, 1],
      collided: false,
      normal: null,
      travelled: 0,
      path: [[1, 1]],
    })
  })

  it('turns, strafes and moves poses', () => {
    const g = worldGeometry(fromAscii(['....', '....']))
    const r = stepPose(g, { x: 1, y: 1, heading: 0 }, { turn: Math.PI / 2, forward: 0.5, strafe: 0.5 }, 0.1)
    expect(r.pose.heading).toBeCloseTo(Math.PI / 2, 12)
    expect(r.pose.x).toBeCloseTo(0.5, 12)
    expect(r.pose.y).toBeCloseTo(1.5, 12)
    expect(stepPose(g, { x: 1, y: 1, heading: Math.PI }, { forward: 5 }).collided).toBe(true)
  })

  it('casts rays within range, and sees what is in the cone', () => {
    const g = worldGeometry(fromAscii(['.....']))
    expect(castRay(g, [0.5, 0.5], 0, { maxDistance: 2 })).toBeNull()
    expect(castRay(g, [0.5, 0.5], Math.PI)?.cell).toBe(-1)
    const fov = fieldOfView(g, { x: 0.5, y: 0.5, heading: 0 }, { angle: 0.2, range: 2.6, rays: 4 })
    expect(fov.cells).toEqual([0, 1, 2])
    expect(fov.polygon).toHaveLength(6)
    const back = fieldOfView(g, { x: 2.5, y: 0.5, heading: Math.PI }, { angle: 0.1 })
    expect(back.cells).toEqual([0, 1, 2])
  })

  it('measures distances to walls and fields of distances on squares and hexagons', () => {
    const blocked = worldGeometry(fromAscii(['##']))
    expect(distanceToWalls(blocked, [1, 0.5])).toBe(Number.POSITIVE_INFINITY)
    expect(Array.from(wallDistanceField(fromAscii(['...'])))).toEqual([Infinity, Infinity, Infinity])
    expect(Array.from(wallDistanceField(fromAscii(['...']), { border: true, metric: 'taxicab' }))).toEqual([1, 1, 1])
    expect(Array.from(wallDistanceField(fromAscii(['#..', '...', '...']), { metric: 'taxicab' }))).toEqual([
      0, 1, 2, 1, 2, 3, 2, 3, 4,
    ])
    const hex = wallDistanceField(fromAscii(['...', '...'], { topology: 'hex' }), { metric: 'exact' })
    for (const d of hex) expect(d).toBeCloseTo(0.5, 12)
    expect(() => wallDistanceField(fromAscii(['..'], { topology: 'hex' }))).toThrow()
    expect(wallDistanceField(fromAscii(['.#']), { metric: 'exact' })[1]).toBe(0)
  })
})
