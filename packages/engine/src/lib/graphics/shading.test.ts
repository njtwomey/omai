import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { gaussLegendre } from '../numerics/quadrature/index.ts'
import { concreteOrThrow, fromStorage, item, toFlat, type Value } from '../tensor/index.ts'
import {
  blinnPhong,
  blinnPhongBrdf,
  blinnPhongDistribution,
  brdf,
  cookTorrance,
  fresnelDielectric,
  fresnelSchlick,
  ggxDistribution,
  halfVector,
  lambert,
  lambertBrdf,
  material,
  metallicRoughness,
  phong,
  phongBrdf,
  reflect,
  shade,
  smithG,
  smithG1,
} from './index.ts'

// Directions over the upper hemisphere (z up) on a product Gauss-Legendre rule in (theta, phi), with the solid-angle
// weights sin(theta) dtheta dphi, so sum(weights * f(directions)) approximates the integral of f over the hemisphere.
function hemisphere(nTheta: number, nPhi: number): { dirs: Value; weights: Float64Array; cos: Float64Array } {
  const gt = gaussLegendre(nTheta)
  const gp = gaussLegendre(nPhi)
  const n = nTheta * nPhi
  const dirs = new Float64Array(3 * n)
  const weights = new Float64Array(n)
  const cos = new Float64Array(n)
  let k = 0
  for (let i = 0; i < nTheta; i++) {
    const theta = ((gt.nodes.data[i] ?? 0) + 1) * (Math.PI / 4)
    const wt = (gt.weights.data[i] ?? 0) * (Math.PI / 4)
    for (let j = 0; j < nPhi; j++) {
      const phi = ((gp.nodes.data[j] ?? 0) + 1) * Math.PI
      const wp = (gp.weights.data[j] ?? 0) * Math.PI
      dirs.set([Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)], 3 * k)
      weights[k] = wt * wp * Math.sin(theta)
      cos[k] = Math.cos(theta)
      k++
    }
  }
  return { dirs: fromStorage(dirs, [n, 3], 'float64'), weights, cos }
}

const integrate = (values: ArrayLike<number>, weights: Float64Array, stride = 1, offset = 0): number => {
  let s = 0
  for (let k = 0; k < weights.length; k++) s += (weights[k] ?? 0) * (values[stride * k + offset] ?? 0)
  return s
}

const N: [number, number, number] = [0, 0, 1]

describe('shading terms', () => {
  it('reflects, halves and clamps as the textbook formulas', () => {
    expect(toFlat(reflect([0.6, -0.8, 0], [0, 1, 0]))).toEqual([0.6, 0.8, 0])
    expect(toFlat(halfVector([1, 0, 0], [1, 0, 0]))).toEqual([1, 0, 0])
    expect(item(lambert(N, [0, 0.6, 0.8]))).toBeCloseTo(0.8, 15)
    expect(item(lambert(N, [0, 0.6, -0.8]))).toBe(0)
    // The mirror of l = (0.6, 0, 0.8) is r = (-0.6, 0, 0.8); a viewer at angle a from r sees cos(a)^s.
    const v = [-0.8, 0, 0.6]
    expect(item(phong(N, [0.6, 0, 0.8], v, 3))).toBeCloseTo((-0.6 * -0.8 + 0.8 * 0.6) ** 3, 14)
    expect(item(phong(N, [0.6, 0, 0.8], [0.6, 0, -0.8], 3))).toBe(0)
    expect(item(blinnPhong(N, [1, 0, 0], [0, 0, 1], 2))).toBeCloseTo(Math.cos(Math.PI / 4) ** 2, 14)
  })

  it('integrates the Lambertian BRDF to the albedo and the normalised Phong lobe to one at normal incidence', () => {
    const { dirs, weights, cos } = hemisphere(96, 32)
    const fl = concreteOrThrow(lambertBrdf([0.2, 0.5, 0.9]), 'test').data
    for (let c = 0; c < 3; c++) {
      expect(integrate(cos, weights) * (fl[c] ?? 0)).toBeCloseTo([0.2, 0.5, 0.9][c] ?? 0, 10)
    }
    for (const s of [0, 1, 8, 40]) {
      const f = concreteOrThrow(phongBrdf(N, N, dirs, s), 'test').data
      const albedo = integrate(
        Float64Array.from(f, (x, k) => x * (cos[k] ?? 0)),
        weights,
      )
      expect(albedo, `shininess ${s}`).toBeCloseTo(1, 6)
      // At oblique incidence part of the lobe falls below the horizon, so less is reflected.
      const l = [Math.sin(1), 0, Math.cos(1)]
      const g = concreteOrThrow(phongBrdf(N, l, dirs, s), 'test').data
      expect(
        integrate(
          Float64Array.from(g, (x, k) => x * (cos[k] ?? 0)),
          weights,
        ),
      ).toBeLessThanOrEqual(1 + 1e-9)
    }
  })

  it('normalises the Blinn-Phong and GGX distributions of microfacet normals', () => {
    const { weights, cos } = hemisphere(400, 4)
    for (const s of [0, 4, 64]) {
      const d = concreteOrThrow(blinnPhongDistribution(fromStorage(cos, [cos.length], 'float64'), s), 'test').data
      expect(
        integrate(
          Float64Array.from(d, (x, k) => x * (cos[k] ?? 0)),
          weights,
        ),
        `s ${s}`,
      ).toBeCloseTo(1, 8)
    }
    for (const alpha of [0.2, 0.5, 1]) {
      const d = concreteOrThrow(ggxDistribution(fromStorage(cos, [cos.length], 'float64'), alpha), 'test').data
      expect(
        integrate(
          Float64Array.from(d, (x, k) => x * (cos[k] ?? 0)),
          weights,
        ),
        `alpha ${alpha}`,
      ).toBeCloseTo(1, 6)
    }
  })

  it('matches Smith masking with its closed form through Lambda, and keeps it in [0, 1]', () => {
    for (const alpha of [0.1, 0.4, 0.9]) {
      for (const theta of [0, 0.3, 0.9, 1.4]) {
        const c = Math.cos(theta)
        const lambda = (-1 + Math.sqrt(1 + alpha * alpha * Math.tan(theta) ** 2)) / 2
        const g1 = item(smithG1(c, alpha))
        expect(g1).toBeCloseTo(1 / (1 + lambda), 12)
        expect(g1).toBeGreaterThanOrEqual(0)
        expect(g1).toBeLessThanOrEqual(1 + 1e-15)
      }
    }
    expect(item(smithG(0.5, 0.7, 0.3))).toBeCloseTo(item(smithG1(0.5, 0.3)) * item(smithG1(0.7, 0.3)), 15)
  })

  it('agrees on Fresnel at normal and grazing incidence and stays in [0, 1]', () => {
    const f0 = ((1 - 1.5) / (1 + 1.5)) ** 2
    expect(item(fresnelDielectric(1, 1, 1.5))).toBeCloseTo(f0, 14)
    expect(item(fresnelSchlick(1, f0))).toBeCloseTo(f0, 15)
    expect(item(fresnelDielectric(1e-9, 1, 1.5))).toBeCloseTo(1, 6)
    for (let k = 0; k <= 20; k++) {
      const c = k / 20
      const exact = item(fresnelDielectric(c, 1, 1.5))
      const schlick = item(fresnelSchlick(c, f0))
      expect(exact).toBeGreaterThanOrEqual(0)
      expect(exact).toBeLessThanOrEqual(1)
      expect(Math.abs(exact - schlick)).toBeLessThan(0.05)
      // From the dense side, beyond the critical angle everything is reflected.
      const inside = item(fresnelDielectric(c, 1.5, 1))
      if (1.5 * Math.sqrt(1 - c * c) > 1) expect(inside).toBe(1)
      else expect(inside).toBeLessThanOrEqual(1)
    }
  })

  it('never reflects more than arrives (white furnace bounds for the microfacet BRDFs)', () => {
    const { dirs, weights, cos } = hemisphere(64, 48)
    for (const roughness of [0.3, 0.7, 1]) {
      for (const theta of [0, 0.6, 1.2]) {
        const l = [Math.sin(theta), 0, Math.cos(theta)]
        const spec = concreteOrThrow(cookTorrance(N, l, dirs, roughness, 1), 'test').data
        const a = integrate(
          Float64Array.from(spec, (x, k) => x * (cos[k] ?? 0)),
          weights,
        )
        expect(a).toBeGreaterThan(0)
        expect(a, `spec ${roughness} ${theta}`).toBeLessThanOrEqual(1 + 1e-6)
        const full = concreteOrThrow(metallicRoughness(N, l, dirs, [1, 1, 1], roughness, 0), 'test').data
        const b = integrate(
          Float64Array.from({ length: weights.length }, (_, k) => (full[3 * k] ?? 0) * (cos[k] ?? 0)),
          weights,
        )
        // Weighting the diffuse lobe by 1 - F(h . v) is the usual approximation; it overshoots slightly at grazing angles.
        expect(b, `full ${roughness} ${theta}`).toBeLessThanOrEqual(1.03)
      }
    }
    const view = concreteOrThrow(blinnPhongBrdf(N, N, dirs, 20), 'test').data
    expect(
      integrate(
        Float64Array.from(view, (x, k) => x * (cos[k] ?? 0)),
        weights,
      ),
    ).toBeLessThan(1.1)
  })

  it('shades with every model and refuses bad materials', () => {
    const l = [0, 0.6, 0.8]
    const v = [0, -0.6, 0.8]
    for (const model of ['lambert', 'phong', 'blinn-phong', 'cook-torrance'] as const) {
      const m = material({ model, albedo: [0.5, 0.5, 0.5] })
      const f = toFlat(brdf(m, N, l, v))
      expect(f).toHaveLength(3)
      expect(f[0]).toBeGreaterThanOrEqual(0.5 / Math.PI - 0.1)
      const rad = toFlat(shade(m, [N, N], [l, l], [v, v], [1, 1, 1]))
      expect(rad).toHaveLength(6)
      expect(rad[0]).toBeCloseTo((f[0] ?? 0) * 0.8, 12)
    }
    expect(() => material({ mirror: 0.7, transparency: 0.7 })).toThrow()
    expect(() => material({ roughness: 0 })).toThrow()
    expect(() => material({ albedo: [1, 1] as unknown as [number, number, number] })).toThrow()
  })
})

describe('shading derivatives', () => {
  const n = [0.1, 0.2, 0.97]
  const l = [0.3, 0.5, 0.8]
  const v = [-0.2, -0.4, 0.9]
  const cases: [string, (...xs: Value[]) => Value, readonly (number | number[])[]][] = [
    ['reflect', (d, nn) => reflect(d as Value, nn as Value), [l, n]],
    ['lambert', (a, b) => lambert(a as Value, b as Value), [n, l]],
    ['phong', (a, b, c, s) => phong(a as Value, b as Value, c as Value, s as Value), [n, l, [-0.3, -0.5, 0.8], 7]],
    ['blinnPhong', (a, b, c, s) => blinnPhong(a as Value, b as Value, c as Value, s as Value), [n, l, v, 12]],
    [
      'ggxDistribution',
      (c, a) => ggxDistribution(c as Value, a as Value),
      [
        [0.9, 0.6],
        [0.4, 0.7],
      ],
    ],
    [
      'smithG1',
      (c, a) => smithG1(c as Value, a as Value),
      [
        [0.9, 0.3],
        [0.4, 0.7],
      ],
    ],
    [
      'fresnelSchlick',
      (c, f) => fresnelSchlick(c as Value, f as Value),
      [
        [0.9, 0.3],
        [0.04, 0.5],
      ],
    ],
    [
      'fresnelDielectric',
      (c, a, b) => fresnelDielectric(c as Value, a as Value, b as Value),
      [
        [0.9, 0.3],
        [1, 1.2],
        [1.5, 1.6],
      ],
    ],
    [
      'cookTorrance',
      (a, b, c, r) => cookTorrance(a as Value, b as Value, c as Value, r as Value, [0.9, 0.6, 0.3]),
      [n, l, v, 0.5],
    ],
    [
      'metallicRoughness',
      (a, b, c, alb, r, m) =>
        metallicRoughness(a as Value, b as Value, c as Value, alb as Value, r as Value, m as Value),
      [n, l, v, [0.8, 0.4, 0.2], 0.6, 0.3],
    ],
  ]
  for (const [name, f, inputs] of cases) {
    it(`${name} passes checkPrimitive`, () => {
      expect(checkPrimitive(f, inputs, { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    })
  }
})
