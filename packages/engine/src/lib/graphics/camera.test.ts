import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { camera, projectPoint, type Vec3 } from '../scene/index.ts'
import { toArray, type Value } from '../tensor/index.ts'
import {
  boxInFrustum,
  cameraBasis,
  cameraFrustum,
  cameraRays,
  fromSceneCamera,
  orthographicCamera,
  perspective,
  pinholeCamera,
  pixelRay,
  pointInFrustum,
  projectPoints,
  projectToPixel,
  sphereInFrustum,
  transformPoint,
  viewProjection,
} from './index.ts'

const coord = fc.double({ min: -3, max: 3, noNaN: true })
const point = fc.tuple(coord, coord, coord) as fc.Arbitrary<Vec3>

describe('cameras', () => {
  it('projects exactly as the scene orbit camera it was built from', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -180, max: 180, noNaN: true }),
        fc.double({ min: -80, max: 80, noNaN: true }),
        fc.constantFrom('perspective' as const, 'orthographic' as const),
        fc.constantFrom([120, 80] as const, [60, 90] as const),
        point,
        (azimuth, elevation, projection, [width, height], p) => {
          const sc = camera({ azimuth, elevation, projection, distance: 8, fov: 40 })
          const want = projectPoint(sc, p, { width, height })
          const got = projectToPixel(fromSceneCamera(sc, { width, height }), p, width, height)
          if (want === undefined) return
          expect(got?.x).toBeCloseTo(want.x, 8)
          expect(got?.y).toBeCloseTo(want.y, 8)
          expect(got?.depth).toBeCloseTo(want.depth, 8)
        },
      ),
    )
  })

  it('casts each pixel ray through the pixel it came from', () => {
    const cams = [
      pinholeCamera({ eye: [2, 1, 4], target: [0, 0.5, 0], fov: 50 }),
      orthographicCamera({ eye: [1, 3, 2], target: [0, 0, 0], up: [0, 0, 1], height: 2 }),
    ]
    for (const cam of cams) {
      fc.assert(
        fc.property(
          fc.double({ min: 0, max: 40, noNaN: true }),
          fc.double({ min: 0, max: 30, noNaN: true }),
          fc.double({ min: 1, max: 6, noNaN: true }),
          (x, y, t) => {
            const r = pixelRay(cam, 40, 30, x, y)
            const p = projectToPixel(
              cam,
              [r.origin[0] + t * r.direction[0], r.origin[1] + t * r.direction[1], r.origin[2] + t * r.direction[2]],
              40,
              30,
            )
            expect(p?.x).toBeCloseTo(x, 8)
            expect(p?.y).toBeCloseTo(y, 8)
          },
        ),
      )
    }
    const { right, up, forward } = cameraBasis(cams[0] as never)
    expect(right[0] * up[0] + right[1] * up[1] + right[2] * up[2]).toBeCloseTo(0, 14)
    expect(forward[0] * up[0] + forward[1] * up[1] + forward[2] * up[2]).toBeCloseTo(0, 14)
  })

  it('generates ray bundles at pixel centres or jittered inside each pixel', () => {
    const cam = pinholeCamera()
    const centre = cameraRays(cam, 5, 3)
    const r = pixelRay(cam, 5, 3, 2.5, 1.5)
    expect(Array.from(centre.directions.slice(3 * 7, 3 * 8))).toEqual(r.direction)
    const a = cameraRays(cam, 5, 3, { key: key(1) })
    expect(cameraRays(cam, 5, 3, { key: key(1) })).toEqual(a)
    for (let k = 0; k < 15; k++) {
      const d = Array.from(a.directions.slice(3 * k, 3 * k + 3)) as unknown as Vec3
      const p = projectToPixel(cam, [d[0], d[1], 5 + d[2]], 5, 3)
      expect(Math.floor(p?.x ?? -1)).toBe(k % 5)
      expect(Math.floor(p?.y ?? -1)).toBe(Math.floor(k / 5))
    }
    expect(() => cameraRays(cam, 0, 3)).toThrow()
  })

  it('tests points, spheres and boxes against the frustum consistently with clip space', () => {
    const cam = pinholeCamera({ eye: [1, 2, 6], target: [0, 0, 0], fov: 60, near: 0.5, far: 12 })
    const planes = cameraFrustum(cam, 1.5)
    const m = viewProjection(cam, 1.5)
    fc.assert(
      fc.property(
        fc.tuple(
          fc.double({ min: -10, max: 10, noNaN: true }),
          fc.double({ min: -10, max: 10, noNaN: true }),
          fc.double({ min: -10, max: 10, noNaN: true }),
        ),
        (p) => {
          const c = transformPoint(m, p as Vec3)
          const w = c[3]
          const inClip = w > 0 && Math.abs(c[0]) <= w && Math.abs(c[1]) <= w && Math.abs(c[2]) <= w
          // Points within rounding of a plane may go either way.
          const margin = Math.min(w - Math.abs(c[0]), w - Math.abs(c[1]), w - Math.abs(c[2]))
          if (Math.abs(margin) > 1e-9) expect(pointInFrustum(planes, p as Vec3)).toBe(inClip)
        },
      ),
    )
    fc.assert(
      fc.property(point, fc.double({ min: 0.1, max: 3, noNaN: true }), (centre, r) => {
        const verdict = sphereInFrustum(planes, centre, r)
        const box = boxInFrustum(
          planes,
          [centre[0] - r, centre[1] - r, centre[2] - r],
          [centre[0] + r, centre[1] + r, centre[2] + r],
        )
        for (const [i, j, k] of [
          [1, 0, 0],
          [-1, 0, 0],
          [0, 1, 0],
          [0, -1, 0],
          [0, 0, 1],
          [0, 0, -1],
          [0, 0, 0],
        ]) {
          const q: Vec3 = [centre[0] + r * (i ?? 0), centre[1] + r * (j ?? 0), centre[2] + r * (k ?? 0)]
          if (verdict === 'outside' || box === 'outside') expect(pointInFrustum(planes, q)).toBe(false)
          if (verdict === 'inside' || box === 'inside') expect(pointInFrustum(planes, q)).toBe(true)
        }
      }),
    )
  })

  it('projects batches differentiably, as single points', () => {
    const cam = pinholeCamera({ eye: [1, 2, 6], fov: 60 })
    const pts = [
      [0, 0, 0],
      [0.5, -1, 1],
    ]
    const got = toArray(projectPoints(cam, pts, 80, 60)) as number[][]
    for (const [i, p] of pts.entries()) {
      const want = projectToPixel(cam, p as unknown as Vec3, 80, 60)
      expect(got[i]?.[0]).toBeCloseTo(want?.x ?? 0, 10)
      expect(got[i]?.[1]).toBeCloseTo(want?.y ?? 0, 10)
      expect(got[i]?.[2]).toBeCloseTo(want?.ndcZ ?? 0, 10)
    }
    expect(checkPrimitive((x: Value) => projectPoints(cam, x, 80, 60), [pts]).failures).toEqual([])
    expect(projectToPixel(cam, [1, 2, 7], 80, 60)).toBeUndefined()
  })

  it('refuses degenerate cameras and matrices', () => {
    expect(() => pinholeCamera({ eye: [0, 0, 0] })).toThrow()
    expect(() => pinholeCamera({ up: [0, 0, 1] })).toThrow()
    expect(() => pinholeCamera({ fov: 180 })).toThrow()
    expect(() => pinholeCamera({ near: 2, far: 1 })).toThrow()
    expect(() => orthographicCamera({ height: 0 })).toThrow()
    expect(() => perspective(60, 0, 1, 2)).toThrow()
  })
})
