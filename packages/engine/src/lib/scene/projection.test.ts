import { describe, expect, it } from 'vitest'
import { scene } from './build.ts'
import { camera, fitCamera, layer3d, orbit, project, projectLayers, projectPoint } from './projection.ts'
import {
  arrow3d,
  axes3d,
  face3d,
  label3d,
  point3d,
  polyline3d,
  scene3d,
  segment3d,
  surface3d,
  triangle3d,
} from './three-d.ts'
import type { GroupNode, LineNode, PathNode } from './types.ts'

const front = camera({ azimuth: -90, elevation: 0, projection: 'orthographic', extent: 1 })
const viewport = { width: 100, height: 100 }

describe('three-d items', () => {
  it('builds every item kind', () => {
    expect(
      face3d([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ]).points,
    ).toHaveLength(3)
    expect(() =>
      face3d([
        [0, 0, 0],
        [1, 0, 0],
      ]),
    ).toThrow()
    expect(triangle3d([0, 0, 0], [1, 0, 0], [0, 1, 0]).kind).toBe('face')
    expect(label3d([0, 0, 0], '\\alpha', { tex: true }).tex).toBe(true)
    expect(axes3d()).toHaveLength(6)
  })

  it('builds surfaces from matrices and functions, with fills by height', () => {
    const faces = surface3d(
      [0, 1, 2],
      [0, 1],
      [
        [0, 1, 2],
        [1, 2, 3],
      ],
      { fill: (z) => (z > 1.5 ? 'tab1' : 'tab0') },
    )
    expect(faces.map((f) => f.fill)).toEqual(['tab0', 'tab1'])
    expect(faces[0]?.points[2]).toEqual([1, 1, 2])
    expect(() => surface3d([0], [0, 1], () => 0)).toThrow()
    expect(() => surface3d([0, 1], [0, 1], [[0, 1]])).toThrow()
  })
})

describe('camera and projection', () => {
  it('looks along +y from the front: x to the right, z up', () => {
    expect(projectPoint(front, [1, 0, 0], viewport)).toEqual({ x: 100, y: 50, depth: 5 })
    const up = projectPoint(front, [0, 0, 1], viewport)
    expect(up?.x).toBeCloseTo(50, 12)
    expect(up?.y).toBeCloseTo(0, 12)
    expect(projectPoint(front, [0, 1, 0], viewport)?.depth).toBeCloseTo(6, 12)
  })

  it('perspective shrinks far things and drops points behind the eye', () => {
    const cam = camera({ azimuth: -90, elevation: 0, distance: 5 })
    const nearX = (projectPoint(cam, [1, -2, 0], viewport)?.x ?? 0) - 50
    const farX = (projectPoint(cam, [1, 2, 0], viewport)?.x ?? 0) - 50
    expect(nearX).toBeGreaterThan(farX)
    expect(projectPoint(cam, [0, -6, 0], viewport)).toBeUndefined()
    const s = project(scene3d([segment3d([0, -6, 0], [0, 0, 0]), point3d([0, 0, 0])]), cam, viewport)
    expect(s.children.map((n) => n.kind)).toEqual(['circle'])
  })

  it('validates and orbits cameras', () => {
    expect(() => camera({ distance: 0 })).toThrow()
    expect(() => camera({ fov: 180 })).toThrow()
    expect(() => camera({ extent: -1 })).toThrow()
    expect(orbit(camera(), 10, -200)).toMatchObject({ azimuth: -50, elevation: -90 })
  })

  it('fitCamera frames the bounding sphere', () => {
    const s = scene3d([point3d([-1, -1, -1]), point3d([1, 1, 1])])
    const cam = fitCamera(camera({ projection: 'orthographic' }), s, 1)
    expect(cam.extent).toBeCloseTo(Math.sqrt(3), 12)
    for (const p of [
      [-1, -1, -1],
      [1, 1, 1],
    ] as const) {
      const q = projectPoint(cam, p, viewport)
      expect(Math.hypot((q?.x ?? 0) - 50, (q?.y ?? 0) - 50)).toBeLessThanOrEqual(50 + 1e-9)
    }
    expect(fitCamera(cam, scene3d([]))).toBe(cam)
  })

  it('splits polylines into segments, keeps arrows, and puts labels on top', () => {
    const s = scene3d([
      label3d([0, 5, 0], 'far'),
      polyline3d(
        [
          [0, 0, 0],
          [1, 0, 0],
          [1, 0, 1],
        ],
        { closed: true },
      ),
      arrow3d([0, 0, 0], [0, 0, 1]),
    ])
    const out = project(s, front, viewport)
    expect(out.children.map((n) => n.kind)).toEqual(['line', 'line', 'line', 'line', 'text'])
    expect(out.children.filter((n) => n.kind === 'line' && (n as LineNode).arrow === true)).toHaveLength(1)
    const mixed = project(s, front, { ...viewport, labelsOnTop: false })
    expect(mixed.children[0]?.kind).toBe('text')
    expect(project(scene3d([label3d([0, 0, 0], 'x', { tex: true })]), front).children[0]?.kind).toBe('tex')
  })

  it('culls faces seen from behind', () => {
    // Counter-clockwise as seen from the front camera (at -y looking towards +y, x right, z up).
    const ccw = face3d([
      [0, 0, 0],
      [1, 0, 0],
      [0, 0, 1],
    ])
    const cw = face3d([
      [0, 0, 0],
      [0, 0, 1],
      [1, 0, 0],
    ])
    const out = project(scene3d([ccw, cw]), front, { ...viewport, cull: true })
    expect(out.children).toHaveLength(1)
    expect((out.children[0] as PathNode).points[1]).toEqual([100, 50])
    expect(project(scene3d([ccw, cw]), front, viewport).children).toHaveLength(2)
  })

  it('projects 3-D layers inside groups', () => {
    const layer = layer3d(scene3d([point3d([0, 0, 0])]), front, { x: 10, y: 20, width: 100, height: 100 }, { id: 'l' })
    const flat = projectLayers(scene(200, 200, [{ kind: 'group', children: [layer] }], { title: 't' }))
    const inner = (flat.children[0] as GroupNode).children[0] as GroupNode
    expect(inner).toMatchObject({ kind: 'group', translate: [10, 20], id: 'l' })
    expect(inner.children[0]).toMatchObject({ kind: 'circle', cx: 50, cy: 50 })
    expect(flat.title).toBe('t')
    expect(project(scene3d([], { title: 'x' }), front, { background: 'bg' })).toMatchObject({
      title: 'x',
      background: 'bg',
    })
  })
})
