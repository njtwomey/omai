import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { recorder } from '../trace/index.ts'
import {
  areaLight,
  cosineHemisphere,
  material,
  pathTrace,
  pinholeCamera,
  plane,
  pointLight,
  rayScene,
  renderWhitted,
  sampleCosineHemisphere,
  sphere,
  tracePath,
} from './index.ts'

describe('path tracing', () => {
  it('samples the hemisphere with density cos(theta) / pi', () => {
    const n: [number, number, number] = [0.6, 0, 0.8]
    const dirs = sampleCosineHemisphere(key(4), n, 20000)
    let meanCos = 0
    let meanCos2 = 0
    for (const d of dirs) {
      expect(Math.hypot(d[0], d[1], d[2])).toBeCloseTo(1, 12)
      const c = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]
      expect(c).toBeGreaterThanOrEqual(-1e-12)
      meanCos += c / dirs.length
      meanCos2 += (c * c) / dirs.length
    }
    // Under cos / pi, E[cos] = 2 / 3 and E[cos^2] = 1 / 2.
    expect(meanCos).toBeCloseTo(2 / 3, 2)
    expect(meanCos2).toBeCloseTo(1 / 2, 2)
    expect(cosineHemisphere(0.25, 0.25)).toEqual([0.5 * Math.cos(Math.PI / 2), 0.5, Math.sqrt(0.75)])
  })

  it('passes the white furnace test exactly: a convex grey object in uniform light shows albedo times the light', () => {
    const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ albedo: [0.3, 0.6, 0.9] }) }], {
      background: [2, 2, 2],
    })
    const img = pathTrace(s, pinholeCamera(), 8, 8, { key: key(1), samples: 3, jitter: false }).image
    const centre = 3 * (4 * 8 + 4)
    expect(Array.from(img.rgb.slice(centre, centre + 3)).map((x) => Math.round(x * 1e12) / 1e12)).toEqual([
      0.6, 1.2, 1.8,
    ])
    expect(img.rgb[0]).toBe(2)
  })

  it('equals Whitted direct lighting where there is no indirect light', () => {
    const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ albedo: [0.7, 0.7, 0.7] }) }], {
      lights: [pointLight([2, 3, 4], [30, 30, 30])],
    })
    const cam = pinholeCamera()
    const a = renderWhitted(s, cam, 6, 6).image.rgb
    const b = pathTrace(s, cam, 6, 6, { key: key(2), samples: 2, jitter: false }).image.rgb
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i] ?? 0, 12)
  })

  it('converges to the stratified estimate of an area light, with a shrinking standard error', () => {
    const s = rayScene([{ shape: plane([0, 0, 0], [0, 1, 0]), material: material({ albedo: [1, 1, 1] }) }], {
      lights: [areaLight([-0.5, 2, -0.5], [1, 0, 0], [0, 0, 1], [5, 5, 5], { samples: 2 })],
    })
    const cam = pinholeCamera({ eye: [0, 3, 3], target: [0, 0, 0] })
    const reference = renderWhitted(
      rayScene(s.objects, { lights: [areaLight([-0.5, 2, -0.5], [1, 0, 0], [0, 0, 1], [5, 5, 5], { samples: 24 })] }),
      cam,
      4,
      4,
    ).image.rgb
    const rec = recorder()
    const img = pathTrace(s, cam, 4, 4, { key: key(3), samples: 64, maxBounces: 0, jitter: false, tracer: rec }).image
      .rgb
    expect(reference.reduce((m, x) => Math.min(m, x), Infinity)).toBeGreaterThan(0.02)
    for (let i = 0; i < img.length; i++) expect(Math.abs((img[i] ?? 0) - (reference[i] ?? 0))).toBeLessThan(0.01)
    const se = rec.events
      .filter((e) => e.type === 'metric' && e.name === 'standardError')
      .map((e) => (e.type === 'metric' ? e.value : 0))
    expect(se.length).toBeGreaterThan(1)
    expect(se[se.length - 1] ?? 1).toBeLessThan((se[4] ?? 0) / 2)
  })

  it('is a pure function of its key and follows single paths through mirrors and glass', () => {
    const s = rayScene(
      [
        { shape: sphere([0, 0, 0], 1), material: material({ transparency: 1 }) },
        { shape: sphere([2.5, 0, 0], 1), material: material({ mirror: 1 }) },
        { shape: plane([0, -1, 0], [0, 1, 0]), material: material({ albedo: [0.5, 0.5, 0.5] }) },
      ],
      { background: [1, 1, 1] },
    )
    const cam = pinholeCamera({ eye: [0, 1, 6] })
    const a = pathTrace(s, cam, 6, 4, { key: key(5), samples: 2 })
    expect(pathTrace(s, cam, 6, 4, { key: key(5), samples: 2 })).toEqual(a)
    expect(pathTrace(s, cam, 6, 4, { key: key(6), samples: 2 })).not.toEqual(a)
    const p = tracePath(s, { origin: [2.5, 0, 5], direction: [0, 0, -1] }, key(0))
    expect(p.vertices[0]?.event).toBe('mirror')
    const g = tracePath(s, { origin: [0, 0, 5], direction: [0, 0, -1] }, key(0), { maxBounces: 8 })
    expect(g.vertices.map((v) => v.event).filter((e) => e === 'refract' || e === 'reflect').length).toBeGreaterThan(0)
    expect(tracePath(s, { origin: [0, 5, 0], direction: [0, -1, 0] }, key(0), { maxBounces: 0 }).vertices).toHaveLength(
      1,
    )
    expect(() => pathTrace(s, cam, 6, 4, { key: key(5), samples: 0 })).toThrow()
  })
})
