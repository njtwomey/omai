import { describe, expect, it } from 'vitest'
import { gradCheck } from '../autodiff/index.ts'
import { cameraBasis, pinholeCamera, projectToPixel } from '../graphics/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, mul, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  colourToCoefficient,
  concatGaussians,
  covariance3d,
  type GaussianSet,
  gaussianColour,
  gaussianCount,
  gaussiansFromPoints,
  largestScale,
  projectGaussians,
  quaternionToMatrix,
  randomGaussians,
  rasterise,
  renderSplats,
  sampleGaussians,
  selectGaussians,
} from './index.ts'

const camera = (z = 4, fov = 90): ReturnType<typeof pinholeCamera> => pinholeCamera({ eye: [0, 0, z], fov })
const one = (options: { scale?: number; opacity?: number; at?: number[]; rgb?: number[] } = {}): GaussianSet =>
  gaussiansFromPoints([options.at ?? [0, 0, 0]], [options.rgb ?? [0.2, 0.4, 0.6]], {
    scale: options.scale ?? 0.5,
    opacity: options.opacity ?? 0.5,
  })
const rounded = (v: ArrayLike<number>, digits = 9): number[] =>
  Array.from(v, (x) => Math.round(x * 10 ** digits) / 10 ** digits + 0)

describe('Gaussian sets', () => {
  it('turns quaternions into rotation matrices: orthonormal, identity for (1, 0, 0, 0), a quarter turn about z', () => {
    const q = tensor([
      [1, 0, 0, 0],
      [0.3, -0.5, 0.2, 0.9],
      [Math.SQRT1_2, 0, 0, Math.SQRT1_2],
    ])
    const R = toFlat(quaternionToMatrix(q))
    expect(rounded(R.slice(0, 9))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1])
    const m = R.slice(9, 18)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let d = 0
        for (let k = 0; k < 3; k++) d += (m[3 * i + k] as number) * (m[3 * j + k] as number)
        expect(d).toBeCloseTo(i === j ? 1 : 0, 12)
      }
    expect(rounded(R.slice(18, 27))).toEqual([0, -1, 0, 1, 0, 0, 0, 0, 1])
  })

  it('has the covariance R S S^T R^T: diag(s^2) for the identity rotation, permuted for a quarter turn about z', () => {
    const g: GaussianSet = {
      ...gaussiansFromPoints(
        [
          [0, 0, 0],
          [0, 0, 0],
        ],
        undefined,
        { scale: 1 },
      ),
      logScales: tensor([
        [Math.log(1), Math.log(2), Math.log(3)],
        [Math.log(1), Math.log(2), Math.log(3)],
      ]),
      rotations: tensor([
        [1, 0, 0, 0],
        [Math.SQRT1_2, 0, 0, Math.SQRT1_2],
      ]),
    }
    const S = toFlat(covariance3d(g))
    expect(rounded(S.slice(0, 9))).toEqual([1, 0, 0, 0, 4, 0, 0, 0, 9])
    // Turned about z, the x and y variances swap places.
    expect(rounded(S.slice(9, 18))).toEqual([4, 0, 0, 0, 1, 0, 0, 0, 9])
    expect(rounded(toFlat(largestScale(g)))).toEqual([3, 3])
  })

  it('initialises scales from the nearest points, colours from the point colours, and at random from a key', () => {
    const g = gaussiansFromPoints(
      [
        [0, 0, 0],
        [1, 0, 0],
        [3, 0, 0],
      ],
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      { neighbours: 2 },
    )
    expect(
      rounded(
        toFlat(g.logScales)
          .filter((_, i) => i % 3 === 0)
          .map((v) => Math.exp(v)),
      ),
    ).toEqual([2, 1.5, 2.5])
    expect(rounded(toFlat(gaussianColour(g)))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1])
    const r = randomGaussians(key(2), 6, { boxHalfWidth: 0.5 })
    expect(toFlat(r.means).every((v) => Math.abs(v) <= 0.5)).toBe(true)
    expect(toFlat(randomGaussians(key(2), 6, { boxHalfWidth: 0.5 }).means)).toEqual(toFlat(r.means))
    expect(() => gaussiansFromPoints([[0, 0]])).toThrow(/\[n, 3\]/)
  })

  it('makes colour depend on the view through degree-1 harmonics, and selects, joins and samples sets', () => {
    const g = gaussiansFromPoints([[0, 0, 0]], [[0.5, 0.5, 0.5]], { shDegree: 1 })
    const withSh: GaussianSet = {
      ...g,
      sh: fromStorage(Float64Array.from([1, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 3, 3], 'float64'),
    }
    const a = toFlat(gaussianColour(withSh, [[1, 0, 0]]))
    const b = toFlat(gaussianColour(withSh, [[0, 1, 0]]))
    expect(rounded(a)).toEqual([0.5, 0.5, 0.5])
    expect(b[0]).toBeCloseTo(0.5 - 0.4886025119029199, 12)
    const both = concatGaussians(randomGaussians(key(0), 2), randomGaussians(key(1), 3))
    expect(gaussianCount(both)).toBe(5)
    expect(gaussianCount(selectGaussians(both, [4, 4, 0]))).toBe(3)
    expect(toFlat(selectGaussians(both, [4]).means)).toEqual(toFlat(both.means).slice(12, 15))
    const draws = Array.from(
      { length: 400 },
      (_, i) => toFlat(sampleGaussians(key(i), one({ scale: 0.3 })))[0] as number,
    )
    const sd = Math.sqrt(draws.reduce((s, v) => s + v * v, 0) / draws.length)
    expect(sd).toBeGreaterThan(0.26)
    expect(sd).toBeLessThan(0.34)
    expect(rounded(toFlat(colourToCoefficient([[0.5, 0.5, 0.5]])))).toEqual([0, 0, 0])
  })
})

describe('projectGaussians', () => {
  it('projects an isotropic Gaussian to a circle whose radius is f s / depth, plus the dilation', () => {
    const f = 32 // a 90 degree field of view over 64 pixels
    for (const depth of [2, 4, 8]) {
      const p = projectGaussians(one({ scale: 0.5 }), camera(depth), 64, 64)
      const [a, b, c] = toFlat(p.cov2d)
      expect(toFlat(p.mean2d)).toEqual([32, 32])
      expect(a).toBeCloseTo((f * 0.5) ** 2 / depth ** 2 + 0.3, 9)
      expect(c).toBeCloseTo(a as number, 12)
      expect(b).toBeCloseTo(0, 12)
      expect(toFlat(p.depth)[0]).toBeCloseTo(depth, 12)
    }
    // The radius of the circle, without the dilation, halves when the depth doubles.
    const radius = (z: number): number =>
      Math.sqrt((toFlat(projectGaussians(one(), camera(z), 64, 64).cov2d)[0] as number) - 0.3)
    expect(radius(4) / radius(8)).toBeCloseTo(2, 9)
  })

  it('agrees with the projection of the graphics library for the centre, and with the EWA formula J W Sigma W^T J^T off axis', () => {
    const cam = pinholeCamera({ eye: [1, 2, 4], target: [0.2, 0, 0], up: [0, 1, 0], fov: 50 })
    const at = [0.3, -0.2, 0.4]
    const g: GaussianSet = {
      ...gaussiansFromPoints([at], undefined, { scale: 1 }),
      logScales: tensor([[Math.log(0.2), Math.log(0.1), Math.log(0.3)]]),
      rotations: tensor([[0.8, 0.1, -0.4, 0.3]]),
    }
    const p = projectGaussians(g, cam, 80, 60)
    const ref = projectToPixel(cam, at as [number, number, number], 80, 60)
    expect(toFlat(p.mean2d)[0]).toBeCloseTo(ref?.x as number, 9)
    expect(toFlat(p.mean2d)[1]).toBeCloseTo(ref?.y as number, 9)
    expect(toFlat(p.depth)[0]).toBeCloseTo(ref?.depth as number, 9)
    // The covariance by hand: camera-space Sigma and the perspective Jacobian at the centre.
    const { right, up, forward } = cameraBasis(cam)
    const W = [right, up.map((v) => -v), forward]
    const Sigma = toFlat(covariance3d(g))
    const d = at.map((v, i) => v - (cam.eye[i] as number))
    const t = W.map((row) => row.reduce((s, v, i) => s + v * (d[i] as number), 0)) as [number, number, number]
    const f = 60 / (2 * Math.tan((50 * Math.PI) / 360))
    const J = [
      [f / t[2], 0, (-f * t[0]) / t[2] ** 2],
      [0, f / t[2], (-f * t[1]) / t[2] ** 2],
    ]
    const A = J.map((row) => [0, 1, 2].map((k) => row.reduce((s, v, i) => s + v * (W[i]?.[k] as number), 0)))
    const cov = (i: number, j: number): number => {
      let s = 0
      for (let k = 0; k < 3; k++)
        for (let l = 0; l < 3; l++) s += (A[i]?.[k] as number) * (Sigma[3 * k + l] as number) * (A[j]?.[l] as number)
      return s
    }
    const [a, b, c] = toFlat(p.cov2d)
    expect(a).toBeCloseTo(cov(0, 0) + 0.3, 9)
    expect(b).toBeCloseTo(cov(0, 1), 9)
    expect(c).toBeCloseTo(cov(1, 1) + 0.3, 9)
  })

  it('culls Gaussians behind the camera and outside the image, and keeps all on request', () => {
    const pts = [
      [0, 0, 0],
      [0, 0, 10],
      [50, 0, 0],
      [1, 1, 0],
    ]
    const g = gaussiansFromPoints(pts, undefined, { scale: 0.2 })
    const p = projectGaussians(g, camera(4), 64, 64)
    expect(Array.from(p.index)).toEqual([0, 3])
    expect(p.mean2d.shape).toEqual([2, 2])
    expect(Array.from(projectGaussians(g, camera(4), 64, 64, { keepAll: true }).index)).toEqual([0, 1, 2, 3])
    expect(p.radius.every((r) => r >= 3)).toBe(true)
  })
})

describe('rasterise and renderSplats', () => {
  it('gives each pixel the opacity o exp(-r^2 / (2 s^2)) of the projected circle, and the Gaussian’s colour', () => {
    const g = one({ scale: 0.5, opacity: 0.8 })
    const p = projectGaussians(g, camera(4), 64, 64)
    const variance = (32 * 0.5) ** 2 / 16 + 0.3
    const xs = [32.5, 34.5, 36.5, 40.5]
    const r = rasterise(
      p,
      xs,
      xs.map(() => 32.5),
    )
    xs.forEach((x, k) => {
      const d2 = (x - 32) ** 2 + 0.5 ** 2
      expect(toFlat(r.alpha)[k]).toBeCloseTo(0.8 * Math.exp(-d2 / (2 * variance)), 12)
    })
    // One Gaussian, black background: colour is alpha times its colour (0.2, 0.4, 0.6).
    const a0 = toFlat(r.alpha)[0] as number
    expect(rounded(toFlat(r.colour).slice(0, 3), 12)).toEqual(rounded([0.2 * a0, 0.4 * a0, 0.6 * a0], 12))
    // The depth is alpha times the Gaussian's depth.
    expect(toFlat(r.depth)[0]).toBeCloseTo(4 * a0, 12)
  })

  it('composites front to back: the nearer opaque Gaussian hides the further, whatever the order in the set', () => {
    const red = gaussiansFromPoints([[0, 0, 1]], [[1, 0, 0]], { scale: 3, opacity: 0.98 })
    const blue = gaussiansFromPoints([[0, 0, -1]], [[0, 0, 1]], { scale: 3, opacity: 0.98 })
    const view = (set: GaussianSet): number[] =>
      toFlat(rasterise(projectGaussians(set, camera(4), 64, 64), [32.5], [32.5]).colour)
    const a = view(concatGaussians(red, blue))
    const b = view(concatGaussians(blue, red))
    expect(a).toEqual(b)
    // The red one is nearer (z = 1 against z = -1, the camera at z = 4): mostly red, a little blue behind it.
    expect(a[0]).toBeGreaterThan(0.9)
    expect(a[2] as number).toBeLessThan(0.05)
    // Seen from the other side the blue one is nearer.
    const back = pinholeCamera({ eye: [0, 0, -4], fov: 90 })
    const c = toFlat(rasterise(projectGaussians(concatGaussians(red, blue), back, 64, 64), [32.5], [32.5]).colour)
    expect(c[2]).toBeGreaterThan(0.9)
  })

  it('lets a Gaussian of opacity zero contribute nothing, and accumulates alpha as 1 - prod(1 - alpha_i)', () => {
    const empty = one({ opacity: 1e-9, scale: 3 })
    const r = rasterise(projectGaussians(empty, camera(4), 64, 64), [32.5], [32.5], { background: [0.1, 0.2, 0.3] })
    expect(toFlat(r.alpha)).toEqual([0])
    expect(toFlat(r.colour)).toEqual([0.1, 0.2, 0.3])
    // Three Gaussians at different depths, with the same footprint.
    const parts = [0.3, 0.5, 0.2].map((o, i) =>
      gaussiansFromPoints([[0, 0, i - 1]], [[0.5, 0.5, 0.5]], { scale: 2, opacity: o }),
    )
    const alphaAt = (set: GaussianSet): number =>
      toFlat(rasterise(projectGaussians(set, camera(6), 64, 64), [40.5], [30.5]).alpha)[0] as number
    // Each one's alpha alone, from its own footprint (the three have different depths, so different footprints).
    const single = parts.map(alphaAt)
    const all = alphaAt(parts.reduce(concatGaussians))
    expect(all).toBeCloseTo(1 - single.reduce((p, a) => p * (1 - a), 1), 12)
  })

  it('stops compositing before the Gaussian that would take the transmittance below 1e-4', () => {
    const slab = gaussiansFromPoints(
      Array.from({ length: 40 }, (_, i) => [0, 0, -i * 0.01]),
      undefined,
      { scale: 5, opacity: 0.5 },
    )
    const alpha1 = toFlat(
      rasterise(projectGaussians(selectGaussians(slab, [0]), camera(4), 64, 64), [32.5], [32.5]).alpha,
    )[0] as number
    const k = Math.floor(Math.log(1e-4) / Math.log(1 - alpha1))
    const total = toFlat(rasterise(projectGaussians(slab, camera(4), 64, 64), [32.5], [32.5]).alpha)[0] as number
    expect(k).toBeLessThan(40)
    // The k-th Gaussian still fits above the floor; the next would cross it.
    expect(total).toBeGreaterThan(1 - (1 - alpha1) ** k - 1e-3)
    expect(total).toBeLessThan(1 - (1 - alpha1) ** (k + 1) + 1e-3)
    expect(1 - total).toBeGreaterThanOrEqual(1e-4 - 1e-9)
  })

  it('renders a full image with the background where nothing is, in chunks that change nothing', () => {
    const g = one({ scale: 0.4, opacity: 0.9, rgb: [1, 0, 0] })
    const img = renderSplats(g, camera(4), 12, 8, { background: [0, 0, 1] })
    expect(img.colour.shape).toEqual([8, 12, 3])
    expect(img.depth.shape).toEqual([8, 12])
    const c = toFlat(img.colour)
    expect(c.slice(0, 3)).toEqual([0, 0, 1])
    const middle = 3 * (4 * 12 + 6)
    expect(c[middle]).toBeGreaterThan(0.5)
    const chunked = renderSplats(g, camera(4), 12, 8, { background: [0, 0, 1], chunk: 7 })
    expect(toFlat(chunked.colour)).toEqual(c)
  })

  it('is differentiable in the means, scales, rotations, opacities and colours (checked against finite differences)', () => {
    const base: GaussianSet = {
      ...gaussiansFromPoints(
        [
          [0.2, 0, 0],
          [-0.25, 0.1, -0.3],
        ],
        [
          [0.9, 0.2, 0.1],
          [0.1, 0.6, 0.8],
        ],
        { scale: 0.5, opacity: 0.6 },
      ),
      rotations: tensor([
        [0.9, 0.1, 0.2, 0.1],
        [0.8, -0.3, 0.1, 0.4],
      ]),
    }
    const xs = [30.5, 33.5, 35.5, 31.5, 36.5, 29.5]
    const ys = [31.5, 29.5, 33.5, 34.5, 31.5, 28.5]
    const weights = tensor([
      [1, 2, 3],
      [-1, 0.5, 2],
      [0.3, 0.7, -0.4],
      [2, 1, 0.5],
      [0.1, 0.2, 0.3],
      [1, -1, 1],
    ])
    const full = (set: GaussianSet): Value => {
      const r = rasterise(projectGaussians(set, camera(4), 64, 64), xs, ys, { background: [0.3, 0.3, 0.3] })
      return sum(mul(r.colour, weights))
    }
    expect(gradCheck((v: Value) => full({ ...base, means: v }), base.means, { eps: 1e-6 }).passed).toBe(true)
    expect(gradCheck((v: Value) => full({ ...base, logScales: v }), base.logScales).passed).toBe(true)
    expect(gradCheck((v: Value) => full({ ...base, rotations: v }), base.rotations).passed).toBe(true)
    expect(gradCheck((v: Value) => full({ ...base, opacityLogits: v }), base.opacityLogits).passed).toBe(true)
    expect(gradCheck((v: Value) => full({ ...base, colours: v }), base.colours).passed).toBe(true)
  })
})
