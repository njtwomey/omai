import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import type { Vec3 } from '../base/index.ts'
import { camera, project, projectPoint } from './projection.ts'
import { face3d, point3d, scene3d, segment3d } from './three-d.ts'

const coord = fc.double({ min: -3, max: 3, noNaN: true, noDefaultInfinity: true })
const vec: fc.Arbitrary<Vec3> = fc.tuple(coord, coord, coord)
const angles = fc.tuple(fc.double({ min: -180, max: 180, noNaN: true }), fc.double({ min: -89, max: 89, noNaN: true }))
const viewport = { width: 200, height: 120 }

describe('projection properties', () => {
  it('the target always lands in the centre of the viewport', () => {
    fc.assert(
      fc.property(
        angles,
        vec,
        fc.constantFrom('perspective', 'orthographic') as fc.Arbitrary<'perspective' | 'orthographic'>,
        ([azimuth, elevation], target, projection) => {
          const p = projectPoint(camera({ azimuth, elevation, target, projection }), target, viewport)
          expect(p?.x).toBeCloseTo(100, 9)
          expect(p?.y).toBeCloseTo(60, 9)
        },
      ),
    )
  })

  it('orthographic projection is affine: midpoints map to midpoints', () => {
    fc.assert(
      fc.property(angles, vec, vec, ([azimuth, elevation], a, b) => {
        const cam = camera({ azimuth, elevation, projection: 'orthographic' })
        const mid: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
        const [pa, pb, pm] = [a, b, mid].map((p) => projectPoint(cam, p, viewport))
        expect(pm?.x).toBeCloseTo(((pa?.x ?? 0) + (pb?.x ?? 0)) / 2, 8)
        expect(pm?.y).toBeCloseTo(((pa?.y ?? 0) + (pb?.y ?? 0)) / 2, 8)
      }),
    )
  })

  it('orthographic images do not depend on the camera distance', () => {
    fc.assert(
      fc.property(angles, vec, fc.double({ min: 1, max: 100, noNaN: true }), ([azimuth, elevation], p, distance) => {
        const near = projectPoint(camera({ azimuth, elevation, projection: 'orthographic', extent: 2 }), p, viewport)
        const far = projectPoint(
          camera({ azimuth, elevation, projection: 'orthographic', extent: 2, distance }),
          p,
          viewport,
        )
        expect(far?.x).toBeCloseTo(near?.x ?? 0, 9)
        expect(far?.y).toBeCloseTo(near?.y ?? 0, 9)
      }),
    )
  })

  it('a full turn of azimuth gives the same image', () => {
    fc.assert(
      fc.property(angles, vec, ([azimuth, elevation], p) => {
        const a = projectPoint(camera({ azimuth, elevation }), p, viewport)
        const b = projectPoint(camera({ azimuth: azimuth + 360, elevation }), p, viewport)
        expect(b?.x).toBeCloseTo(a?.x ?? 0, 8)
        expect(b?.depth).toBeCloseTo(a?.depth ?? 0, 8)
      }),
    )
  })

  it('perspective points on a ray from the eye project to one image point', () => {
    fc.assert(
      fc.property(angles, vec, fc.double({ min: 0.2, max: 0.9, noNaN: true }), ([azimuth, elevation], p, t) => {
        const cam = camera({ azimuth, elevation, distance: 20 })
        const az = (azimuth * Math.PI) / 180
        const el = (elevation * Math.PI) / 180
        const eye: Vec3 = [20 * Math.cos(el) * Math.cos(az), 20 * Math.cos(el) * Math.sin(az), 20 * Math.sin(el)]
        const q: Vec3 = [eye[0] + t * (p[0] - eye[0]), eye[1] + t * (p[1] - eye[1]), eye[2] + t * (p[2] - eye[2])]
        const pp = projectPoint(cam, p, viewport)
        const pq = projectPoint(cam, q, viewport)
        expect(pq?.x).toBeCloseTo(pp?.x ?? 0, 7)
        expect(pq?.y).toBeCloseTo(pp?.y ?? 0, 7)
        expect(pq?.depth ?? 0).toBeLessThan(pp?.depth ?? 0)
      }),
    )
  })

  it('projected scenes are painted back to front', () => {
    fc.assert(
      fc.property(angles, fc.array(vec, { minLength: 1, maxLength: 12 }), ([azimuth, elevation], points) => {
        const cam = camera({ azimuth, elevation, distance: 30 })
        const s = scene3d(points.map((p, i) => point3d(p, { id: String(i) })))
        const out = project(s, cam, viewport)
        const depths = out.children.map((n) => projectPoint(cam, points[Number(n.id)] as Vec3, viewport)?.depth ?? 0)
        for (let i = 1; i < depths.length; i++) expect(depths[i]).toBeLessThanOrEqual((depths[i - 1] as number) + 1e-12)
        expect(out.children).toHaveLength(points.length)
      }),
    )
  })

  it('every item projects to a node of the matching kind', () => {
    fc.assert(
      fc.property(angles, vec, vec, vec, ([azimuth, elevation], a, b, c) => {
        const out = project(scene3d([segment3d(a, b), face3d([a, b, c])]), camera({ azimuth, elevation, distance: 30 }))
        expect(out.children.map((n) => n.kind).sort()).toEqual(['line', 'path'])
      }),
    )
  })
})
