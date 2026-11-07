import { describe, expect, it } from 'vitest'
import type { Vec3 } from '../base/index.ts'
import { key } from '../random/index.ts'
import { areaLight, directionalLight, lightSamples, material, pointLight, shadePoints } from './index.ts'

// Irradiance at a point with normal n from a uniformly emitting polygon of radiance L (Lambert's contour formula):
// E = L / 2 sum_i angle_i (n . unit(r_i x r_{i+1})), r_i the vectors from the point to the vertices.
function polygonIrradiance(point: Vec3, normal: Vec3, vertices: readonly Vec3[], radiance: number): number {
  let s = 0
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i] as Vec3
    const b = vertices[(i + 1) % vertices.length] as Vec3
    const ra = [a[0] - point[0], a[1] - point[1], a[2] - point[2]]
    const rb = [b[0] - point[0], b[1] - point[1], b[2] - point[2]]
    const c = [
      (ra[1] ?? 0) * (rb[2] ?? 0) - (ra[2] ?? 0) * (rb[1] ?? 0),
      (ra[2] ?? 0) * (rb[0] ?? 0) - (ra[0] ?? 0) * (rb[2] ?? 0),
      (ra[0] ?? 0) * (rb[1] ?? 0) - (ra[1] ?? 0) * (rb[0] ?? 0),
    ]
    const cn = Math.hypot(c[0] ?? 0, c[1] ?? 0, c[2] ?? 0)
    const angle = Math.atan2(
      cn,
      (ra[0] ?? 0) * (rb[0] ?? 0) + (ra[1] ?? 0) * (rb[1] ?? 0) + (ra[2] ?? 0) * (rb[2] ?? 0),
    )
    s += (angle * ((c[0] ?? 0) * normal[0] + (c[1] ?? 0) * normal[1] + (c[2] ?? 0) * normal[2])) / cn
  }
  return (radiance * Math.abs(s)) / 2
}

describe('lights', () => {
  it('falls off with the inverse square from a point light and not at all from a directional one', () => {
    const [s] = lightSamples(pointLight([0, 0, 0], [8, 8, 8]), [2, 0, 0])
    expect(s).toEqual({ direction: [-1, 0, 0], distance: 2, irradiance: [2, 2, 2] })
    expect(lightSamples(pointLight([0, 0, 0], [1, 1, 1]), [0, 0, 0])).toEqual([])
    const [d] = lightSamples(directionalLight([0, -1, 0], [3, 2, 1]), [5, 5, 5])
    expect(d?.direction).toEqual([-0, 1, -0])
    expect(d?.distance).toBe(Number.POSITIVE_INFINITY)
  })

  it('integrates an area light to the closed-form irradiance of a rectangle', () => {
    const light = areaLight([-0.5, -0.5, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], { samples: 24 })
    const corners: Vec3[] = [
      [-0.5, -0.5, 1],
      [-0.5, 0.5, 1],
      [0.5, 0.5, 1],
      [0.5, -0.5, 1],
    ]
    const m = material({ albedo: [Math.PI, Math.PI, Math.PI] })
    for (const p of [
      [0, 0, 0],
      [0.7, 0.2, 0.1],
      [1.5, -1, 0.5],
    ] as Vec3[]) {
      const got = shadePoints(m, p, [0, 0, 1], [0, 0, 1], [light])[0] ?? 0
      expect(got).toBeCloseTo(polygonIrradiance(p, [0, 0, 1], corners, 1), 3)
    }
    // Seen from behind (above), the one-sided light delivers nothing.
    expect(lightSamples(light, [0, 0, 2]).every((s) => s.irradiance[0] === 0)).toBe(true)
    // Jittered samples are a pure function of the key and estimate the same irradiance.
    const a = lightSamples(light, [0, 0, 0], key(3))
    expect(lightSamples(light, [0, 0, 0], key(3))).toEqual(a)
    const sum = a.reduce((t, s) => t + s.irradiance[0] * s.direction[2], 0)
    expect(sum).toBeCloseTo(polygonIrradiance([0, 0, 0], [0, 0, 1], corners, 1), 2)
    expect(() => areaLight([0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 1])).toThrow()
    expect(() => areaLight([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], { samples: 0 })).toThrow()
  })

  it('adds emission and ambient light and respects visibility', () => {
    const m = material({ albedo: [0.5, 0.5, 0.5], emission: [0.1, 0, 0] })
    const lights = [pointLight([0, 0, 1], [1, 1, 1])]
    const lit = shadePoints(m, [0, 0, 0, 5, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1], lights, {
      ambient: [0.2, 0.2, 0.2],
      visible: (i) => i === 0,
    })
    expect(lit[0]).toBeCloseTo(0.1 + 0.1 + 0.5 / Math.PI, 14)
    expect(lit[3]).toBeCloseTo(0.1 + 0.1, 14)
    expect(lit[4]).toBeCloseTo(0.1, 14)
  })
})
