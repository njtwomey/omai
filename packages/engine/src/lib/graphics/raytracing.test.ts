import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import type { Vec3 } from '../base/index.ts'
import { item, toFlat, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import {
  checkerboard,
  material,
  pinholeCamera,
  plane,
  pointLight,
  rayScene,
  refract,
  renderWhitted,
  schlick,
  sphere,
  splitWeights,
  traceRay,
} from './index.ts'

const unit = (v: Vec3): Vec3 => {
  const n = Math.hypot(v[0], v[1], v[2])
  return [v[0] / n, v[1] / n, v[2] / n]
}

describe('Whitted ray tracing', () => {
  const floor = plane([0, 0, 0], [0, 1, 0])
  const ball = sphere([0, 1, 0], 1)
  const light = pointLight([0, 5, 0], [25, 25, 25])

  it('shades a Lambertian floor by the closed form, and darkens it in the shadow of a sphere', () => {
    const s = rayScene([{ shape: floor, material: material({ albedo: [0.5, 0.5, 0.5] }) }], { lights: [light] })
    const origin: Vec3 = [3, 4, 0]
    const target: Vec3 = [2, 0, 0]
    const r = traceRay(s, { origin, direction: unit([target[0] - origin[0], target[1] - origin[1], 0]) })
    const d2 = 2 * 2 + 5 * 5
    const cos = 5 / Math.sqrt(d2)
    expect(r.radiance[0]).toBeCloseTo((0.5 / Math.PI) * (25 / d2) * cos, 12)
    const shadowed = rayScene(
      [
        { shape: floor, material: material({ albedo: [0.5, 0.5, 0.5] }) },
        { shape: ball, material: material() },
      ],
      { lights: [light], ambient: [0.1, 0.1, 0.1] },
    )
    const below = traceRay(shadowed, { origin: [5, 0.3, 0], direction: unit([-4.2, -0.3, 0]) })
    expect(below.radiance[0]).toBeCloseTo(0.05, 12)
    expect(below.records[0]?.shadows.every((x) => x.blocked)).toBe(true)
  })

  it('splits energy into local, mirror and refraction weights that sum to one', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 1.01, max: 2.5, noNaN: true }),
        fc.boolean(),
        (a, b, cos, ior, entering) => {
          const mirror = a * (1 - b)
          const transparency = b * (1 - mirror)
          const w = splitWeights(material({ mirror, transparency, ior }), cos, entering)
          expect(w.local + w.reflect + w.refract).toBeCloseTo(1, 12)
          for (const x of [w.local, w.reflect, w.refract]) {
            expect(x).toBeGreaterThanOrEqual(-1e-15)
            expect(x).toBeLessThanOrEqual(1 + 1e-15)
          }
        },
      ),
    )
  })

  it('passes a uniform background through glass without gaining energy', () => {
    const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ transparency: 1, ior: 1.5 }) }], {
      background: [1, 1, 1],
    })
    for (const x of [0, 0.3, 0.7, 0.95]) {
      const r = traceRay(s, { origin: [x, 0, 5], direction: [0, 0, -1] }, { maxDepth: 12, minWeight: 1e-6 })
      expect(r.radiance[0]).toBeLessThanOrEqual(1 + 1e-12)
      expect(r.radiance[0]).toBeGreaterThan(0.97)
      // Every spawned ray carries no more than its parent.
      for (const rec of r.records) {
        const parent = r.records.find((p) => p.id === rec.parent)
        if (parent !== undefined) expect(rec.weight[0]).toBeLessThanOrEqual(parent.weight[0] + 1e-15)
      }
    }
    // Through the centre the ray goes straight through, refracting twice.
    const centre = traceRay(s, { origin: [0, 0, 5], direction: [0, 0, -1] })
    expect(centre.records.filter((r) => r.kind === 'refraction').length).toBeGreaterThanOrEqual(2)
  })

  it('stops between facing mirrors at the depth cap', () => {
    const s = rayScene(
      [
        { shape: plane([0, 0, -1], [0, 0, 1]), material: material({ mirror: 0.9 }) },
        { shape: plane([0, 0, 1], [0, 0, -1]), material: material({ mirror: 0.9 }) },
      ],
      { lights: [pointLight([0, 0, 0], [1, 1, 1])] },
    )
    const r = traceRay(s, { origin: [0, 0, 0], direction: unit([0.1, 0, 1]) }, { maxDepth: 6 })
    expect(r.records).toHaveLength(7)
    expect(r.records.reduce((m, x) => Math.max(m, x.depth), 0)).toBe(6)
    expect(Number.isFinite(r.radiance[0])).toBe(true)
    expect(() => traceRay(s, { origin: [0, 0, 0], direction: [0, 0, 0] })).toThrow()
  })

  it('renders as a traced algorithm with one step per generation and records chosen pixels', () => {
    const s = rayScene(
      [
        {
          shape: floor,
          material: material({ albedo: [1, 1, 1] }),
          albedoMap: (u, v) => checkerboard(u, v, { scale: 1 }),
        },
        { shape: ball, material: material({ model: 'blinn-phong', mirror: 0.3, albedo: [0.8, 0.2, 0.2] }) },
      ],
      { lights: [light], background: [0.2, 0.3, 0.4] },
    )
    const rec = recorder()
    const cam = pinholeCamera({ eye: [0, 2, 6], target: [0, 1, 0] })
    const out = renderWhitted(s, cam, 12, 9, { tracer: rec, record: [4 * 12 + 6] })
    expect(out.image.rgb.every((x) => Number.isFinite(x) && x >= 0)).toBe(true)
    expect(out.stats.rays).toBeGreaterThan(12 * 9)
    const events = rec.events.filter((e) => e.type === 'metric' && e.name === 'rays')
    expect(events.length).toBeGreaterThanOrEqual(out.stats.generations)
    expect(out.records.every((r) => r.pixel === 4 * 12 + 6)).toBe(true)
    expect(out.records.some((r) => r.kind === 'reflection')).toBe(true)
    // The same pixel traced on its own gives the same radiance.
    const again = renderWhitted(s, cam, 12, 9)
    expect(Array.from(again.image.rgb)).toEqual(Array.from(out.image.rgb))
  })
})

describe('reflection and refraction', () => {
  it('obeys Snell and keeps refracted directions unit length', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1.5, noNaN: true }),
        fc.double({ min: 0.4, max: 2.5, noNaN: true }),
        (theta, eta) => {
          const d: Vec3 = [Math.sin(theta), -Math.cos(theta), 0]
          const t = toFlat(refract(d, [0, 1, 0], eta))
          const sinT = eta * Math.sin(theta)
          if (sinT > 1) {
            expect(t).toEqual([0, 0, 0])
            expect(item(schlick(Math.cos(theta), eta, 1))).toBe(1)
            return
          }
          expect(Math.hypot(t[0] ?? 0, t[1] ?? 0, t[2] ?? 0)).toBeCloseTo(1, 12)
          expect(t[0]).toBeCloseTo(sinT, 12)
          expect(t[1]).toBeLessThanOrEqual(0)
          const r = item(schlick(Math.cos(theta), eta, 1))
          expect(r).toBeGreaterThanOrEqual(0)
          expect(r).toBeLessThanOrEqual(1)
        },
      ),
    )
  })

  it('refract and schlick pass checkPrimitive', () => {
    const f = (d: Value, n: Value, e: Value): Value => refract(d, n, e)
    expect(checkPrimitive(f, [[0.3, -0.9, 0.1], [0.05, 1, 0.02], 0.7]).failures).toEqual([])
    const g = (c: Value, a: Value, b: Value): Value => schlick(c, a, b)
    expect(
      checkPrimitive(g, [
        [0.9, 0.4],
        [1, 1.1],
        [1.5, 1.6],
      ]).failures,
    ).toEqual([])
    expect(
      checkPrimitive(g, [
        [0.95, 0.9],
        [1.5, 1.4],
        [1, 1.05],
      ]).failures,
    ).toEqual([])
  })
})
