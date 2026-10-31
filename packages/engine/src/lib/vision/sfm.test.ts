import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  addView,
  bundleAdjustmentAlgorithm,
  bundleReprojectionError,
  calibrateCamera,
  calibrationAlgorithm,
  essentialRansacAlgorithm,
  findEssentialMat,
  projectPoints,
  rodrigues,
  twoViewReconstruction,
} from './index.ts'

const K = [
  [300, 0, 160],
  [0, 300, 120],
  [0, 0, 1],
]
const N = 48
const world: number[][] = Array.from({ length: N }, (_, i) => [
  3.6 * Math.sin(2.1 * i + 0.3) * 0.5,
  2.6 * Math.cos(1.3 * i + 0.7) * 0.5,
  5 + 2 * (0.5 + 0.5 * Math.sin(0.77 * i + 1)),
])
const poses = [
  { rvec: [0, 0, 0], tvec: [0, 0, 0] },
  { rvec: [0.02, 0.15, 0.01], tvec: [-1, 0.05, 0.1] },
  { rvec: [-0.03, -0.1, 0.02], tvec: [0.8, -0.1, 0.2] },
] as { rvec: [number, number, number]; tvec: [number, number, number] }[]
const noise = toFlat(normal(key(21), [3, N, 2]))
const sigma = 0.3
const observed = poses.map((p, c) =>
  projectPoints(world, p, K).map(
    ([u, v], j) =>
      [u + sigma * (noise[(c * N + j) * 2] as number), v + sigma * (noise[(c * N + j) * 2 + 1] as number)] as [
        number,
        number,
      ],
  ),
)
const exact = poses.map((p) => projectPoints(world, p, K))
const norm3 = (v: readonly number[]): number => Math.hypot(...v)

describe('essential matrix from correspondences', () => {
  it('is recovered from exact correspondences to numerical precision', () => {
    const e = findEssentialMat(exact[0] as [number, number][], exact[1] as [number, number][], K, {
      method: 'all',
    }).model
    const R = rodrigues(poses[1]?.rvec as number[])
    const t = poses[1]?.tvec as number[]
    const tx = [
      [0, -(t[2] as number), t[1] as number],
      [t[2] as number, 0, -(t[0] as number)],
      [-(t[1] as number), t[0] as number, 0],
    ]
    const truth = tx.map((row) => [0, 1, 2].map((c) => row.reduce((s, v, k) => s + v * (R[k]?.[c] ?? 0), 0)))
    const fro = (m: number[][]): number => Math.hypot(...m.flat())
    const a = truth.map((r) => r.map((v) => v / fro(truth)))
    const sign = Math.sign(e.flat().reduce((s, v, i) => s + v * (a.flat()[i] as number), 0))
    const b = e.map((r) => r.map((v) => (sign * v) / fro(e)))
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(Math.abs((a[i]?.[j] as number) - (b[i]?.[j] as number))).toBeLessThan(1e-8)
  })
  it('RANSAC finds the inliers among planted outliers, one hypothesis per step', () => {
    const p2 = (observed[1] as [number, number][]).map((p, j): [number, number] =>
      j % 5 === 0 ? [p[0] + 40 + 3 * j, p[1] - 30] : p,
    )
    const alg = essentialRansacAlgorithm(observed[0] as [number, number][], p2, K, {
      threshold: 3,
      maxTrials: 150,
      key: key(2),
    })
    let s = alg.init(key(0))
    const sizes: number[] = []
    for (let t = 0; t < 150; t++) {
      s = alg.step(s, {
        t,
        key: key(100 + t),
        log: () => undefined,
        wants: () => false,
        emit: () => undefined,
      } as never)
      sizes.push(s.bestInliers.length)
      if (alg.done?.(s, t) !== undefined) break
    }
    for (let i = 1; i < sizes.length; i++) expect(sizes[i] as number).toBeGreaterThanOrEqual(sizes[i - 1] as number)
    const planted = Array.from({ length: N }, (_, j) => j).filter((j) => j % 5 !== 0)
    const found = new Set(s.bestInliers)
    expect(planted.filter((j) => found.has(j)).length).toBeGreaterThanOrEqual(planted.length - 3)
    expect(s.bestInliers.filter((j) => j % 5 === 0).length).toBeLessThanOrEqual(2)
  })
})

describe('structure from motion', () => {
  const ransac = runAlgorithm(
    essentialRansacAlgorithm(observed[0] as [number, number][], observed[1] as [number, number][], K, {
      threshold: 4,
      maxTrials: 100,
      key: key(4),
    }),
    {
      maxSteps: 100,
      seed: 4,
    },
  ).state
  const two = twoViewReconstruction(ransac, observed[0] as [number, number][], observed[1] as [number, number][], K)
  const baseline = norm3(poses[1]?.tvec as number[])

  it('recovers the second camera and the structure up to scale', () => {
    const truthR = rodrigues(poses[1]?.rvec as number[])
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        expect(Math.abs((two.R[i]?.[j] as number) - (truthR[i]?.[j] as number))).toBeLessThan(3e-2)
    const t = poses[1]?.tvec as number[]
    const cosine = two.t.reduce((s, v, i) => s + v * (t[i] as number), 0) / baseline
    expect(cosine).toBeGreaterThan(0.995)
    expect(two.points.length).toBeGreaterThanOrEqual(N - 6)
    // Points are the truth divided by the baseline length (unit-translation scale).
    const errors: number[] = []
    two.indices.forEach((j, k) => {
      const p = two.points[k] as [number, number, number]
      const q = world[j] as number[]
      errors.push(norm3(p.map((v, i) => v * baseline - (q[i] as number))))
    })
    errors.sort((x, y) => x - y)
    // Two noisy views fix depth only to about depth times the angular error of the baseline (here a few degrees).
    expect(errors[Math.floor(errors.length / 2)]).toBeLessThan(0.1 * 6 * 3)
  })

  it('recovers exact correspondences completely: pose and structure to numerical precision', () => {
    const r = runAlgorithm(
      essentialRansacAlgorithm(exact[0] as [number, number][], exact[1] as [number, number][], K, {
        threshold: 1e-3,
        key: key(6),
      }),
      {
        maxSteps: 20,
        seed: 6,
      },
    ).state
    const clean = twoViewReconstruction(r, exact[0] as [number, number][], exact[1] as [number, number][], K)
    const truthR = rodrigues(poses[1]?.rvec as number[])
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        expect(Math.abs((clean.R[i]?.[j] as number) - (truthR[i]?.[j] as number))).toBeLessThan(1e-8)
    expect(clean.points.length).toBe(N)
    clean.indices.forEach((j, k) => {
      const p = clean.points[k] as [number, number, number]
      expect(norm3(p.map((v, i) => v * baseline - ((world[j] as number[])[i] as number)))).toBeLessThan(1e-6)
    })
  })

  it('bundle adjustment lowers the reprojection error step by step and tightens the structure', () => {
    const idx = [...two.indices]
    const points = two.points.map((p) => p.map((v) => v * baseline))
    const cam2 = addView(
      points,
      idx.map((j) => observed[2]?.[j] as [number, number]),
      K,
    )
    const scene = {
      poses: [
        poses[0] as never,
        {
          rvec: (poses[1]?.rvec ?? []).map((v) => v * 1.1),
          tvec: two.t.map((v) => v * baseline * 1.03),
        } as never,
        cam2,
      ],
      points,
      observations: observed.map((o) => idx.map((j) => o[j] as [number, number])),
    }
    const before = bundleReprojectionError(scene, K)
    const alg = bundleAdjustmentAlgorithm(scene, K, { iterations: 30 })
    let s = alg.init(key(0))
    const rms = [s.rms]
    const ctx = { t: 0, key: key(1), log: () => undefined, wants: () => false, emit: () => undefined } as never
    for (let t = 0; t < 30 && alg.done?.(s, t) === undefined; t++) {
      s = alg.step(s, ctx)
      rms.push(s.rms)
    }
    expect(rms[0]).toBeCloseTo(before, 6)
    for (let i = 1; i < rms.length; i++) expect(rms[i] as number).toBeLessThanOrEqual((rms[i - 1] as number) + 1e-12)
    const last = rms[rms.length - 1] as number
    expect(last).toBeLessThan(0.5 * before)
    // The noise floor: residuals of a good fit are about the observation noise (sigma per coordinate, a little less with the fitted parameters).
    expect(last).toBeLessThan(2 * sigma * Math.SQRT2)
    // Structure against the truth, after bundle adjustment, is within the noise-limited accuracy.
    const dist = (pts: readonly (readonly number[])[]): number => {
      let sum = 0
      idx.forEach((j, k) => {
        sum += norm3((pts[k] as number[]).map((v, i) => v - ((world[j] as number[])[i] as number))) ** 2
      })
      return Math.sqrt(sum / idx.length)
    }
    // Align the scale (camera 0 fixed leaves the scale free).
    let num = 0
    let den = 0
    idx.forEach((j, k) => {
      for (let i = 0; i < 3; i++) {
        num += (s.decoded.points[k]?.[i] as number) * (world[j]?.[i] as number)
        den += (s.decoded.points[k]?.[i] as number) ** 2
      }
    })
    const scale = num / den
    expect(dist(s.decoded.points.map((p) => p.map((v) => v * scale)))).toBeLessThan(dist(points))
    expect(dist(s.decoded.points.map((p) => p.map((v) => v * scale)))).toBeLessThan(0.08)
  })

  it('serialises a bundle adjustment state', () => {
    const idx = [...two.indices].slice(0, 20)
    const points = two.points.slice(0, 20).map((p) => p.map((v) => v * baseline))
    const scene = {
      poses: [poses[0] as never, poses[1] as never, poses[2] as never],
      points,
      observations: observed.map((o) => idx.map((j) => o[j] as [number, number])),
    }
    const alg = bundleAdjustmentAlgorithm(scene, K, { iterations: 5 })
    const mid = runAlgorithm(alg, { maxSteps: 2 })
    const back = alg.deserialise?.(alg.serialise?.(mid.state) as never)
    expect(back?.iteration).toBe(2)
    expect(back?.rms).toBeCloseTo(mid.state.rms, 12)
    const rest = runAlgorithm(alg, { maxSteps: 5, start: { state: back, step: 2 } })
    expect(rest.state.rms).toBeLessThanOrEqual(mid.state.rms + 1e-12)
  })
})

describe('calibration iterations', () => {
  const Ktrue = [
    [420, 0, 316],
    [0, 400, 242],
    [0, 0, 1],
  ]
  const dist = [-0.18, 0.05, 0.001, -0.0005, 0]
  const board = Array.from({ length: 35 }, (_, i) => [(i % 7) * 0.5, Math.floor(i / 7) * 0.5, 0])
  const views = [
    { rvec: [0.3, -0.2, 0.05], tvec: [-1.5, -1, 6] },
    { rvec: [-0.25, 0.3, 0.1], tvec: [-1.2, -0.8, 7] },
    { rvec: [0.1, 0.45, -0.1], tvec: [-1.4, -1.1, 5.5] },
    { rvec: [-0.4, -0.15, 0.2], tvec: [-1, -1.2, 6.5] },
  ] as { rvec: [number, number, number]; tvec: [number, number, number] }[]
  const images = views.map((p) => projectPoints(board, p, Ktrue, dist))

  it('starts from the closed form and descends to the planted intrinsics', () => {
    const alg = calibrationAlgorithm(
      views.map(() => board),
      images,
      { fixK3: true, iterations: 100 },
    )
    let s = alg.init(key(0))
    const start = s.decoded
    expect(start.distortion.every((v) => v === 0)).toBe(true)
    const errors = [s.rms]
    const ctx = { t: 0, key: key(1), log: () => undefined, wants: () => false, emit: () => undefined } as never
    for (let t = 0; t < 100 && alg.done?.(s, t) === undefined; t++) {
      s = alg.step(s, ctx)
      errors.push(s.rms)
    }
    for (let i = 1; i < errors.length; i++)
      expect(errors[i] as number).toBeLessThanOrEqual((errors[i - 1] as number) + 1e-12)
    expect(errors[0] as number).toBeGreaterThan(0.1)
    expect(s.rms).toBeLessThan(1e-4)
    const c = s.decoded
    expect(c.cameraMatrix[0]?.[0]).toBeCloseTo(420, 2)
    expect(c.cameraMatrix[1]?.[1]).toBeCloseTo(400, 2)
    expect(c.cameraMatrix[0]?.[2]).toBeCloseTo(316, 2)
    expect(c.cameraMatrix[1]?.[2]).toBeCloseTo(242, 2)
    for (const [i, d] of dist.slice(0, 4).entries()) expect(c.distortion[i]).toBeCloseTo(d, 4)
    // The same optimum as the one-shot calibration.
    const one = calibrateCamera(
      views.map(() => board),
      images,
      { fixK3: true },
    )
    expect(c.cameraMatrix[0]?.[0]).toBeCloseTo(one.cameraMatrix[0]?.[0] as number, 5)
  })
})
