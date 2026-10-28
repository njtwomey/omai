import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import * as V from './index.ts'

const K = [
  [250, 0, 80],
  [0, 240, 60],
  [0, 0, 1],
]

const homographyArb = fc
  .tuple(
    fc.double({ min: 0.8, max: 1.2, noNaN: true }),
    fc.double({ min: -0.2, max: 0.2, noNaN: true }),
    fc.double({ min: -20, max: 20, noNaN: true }),
    fc.double({ min: -0.2, max: 0.2, noNaN: true }),
    fc.double({ min: 0.8, max: 1.2, noNaN: true }),
    fc.double({ min: -20, max: 20, noNaN: true }),
    fc.double({ min: -5e-4, max: 5e-4, noNaN: true }),
    fc.double({ min: -5e-4, max: 5e-4, noNaN: true }),
  )
  .map(([a, b, c, d, e, f, g, h]) => [
    [a, b, c],
    [d, e, f],
    [g, h, 1],
  ])

const grid: [number, number][] = Array.from({ length: 12 }, (_, i) => [(i % 4) * 40 + 10, Math.floor(i / 4) * 50 + 5])

describe('planar geometry', () => {
  it('homographies round-trip: estimate from mapped points, map back', () => {
    fc.assert(
      fc.property(homographyArb, (H) => {
        const dst = V.perspectiveTransform(grid, H)
        const est = V.findHomography(grid, dst, { method: 'all' }).model
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(est[i]?.[j]).toBeCloseTo(H[i]?.[j] ?? 0, 7)
        const four = [0, 3, 8, 11]
        const exact = V.getPerspectiveTransform(
          four.map((i) => grid[i] as [number, number]),
          four.map((i) => dst[i] as [number, number]),
        )
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(exact[i]?.[j]).toBeCloseTo(H[i]?.[j] ?? 0, 7)
        const robust = V.findHomography(
          grid,
          dst.map((p, i) => (i === 3 ? [p[0] + 50, p[1] - 40] : p)),
          { key: key(1) },
        )
        expect(robust.mask[3]).toBe(0)
        expect(robust.mask.reduce((s, v) => s + v, 0)).toBe(11)
      }),
      { numRuns: 15 },
    )
  })

  it('recovers affine maps and keeps x2^T F x1 = 0', () => {
    const A = [
      [1.1, -0.1, 4],
      [0.2, 0.9, -2],
    ]
    const dst = grid.map(([x, y]): [number, number] => [
      (A[0]?.[0] ?? 0) * x + (A[0]?.[1] ?? 0) * y + (A[0]?.[2] ?? 0),
      (A[1]?.[0] ?? 0) * x + (A[1]?.[1] ?? 0) * y + (A[1]?.[2] ?? 0),
    ])
    const est = V.estimateAffine2D(grid, dst, { key: key(2) }).model
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) expect(est[i]?.[j]).toBeCloseTo(A[i]?.[j] ?? 0, 8)
    const tri = [0, 1, 4]
    const pick = (ps: [number, number][]): [number, number][] => tri.map((i) => ps[i] as [number, number])
    expect(V.getAffineTransform(pick(grid), pick(dst))[0]?.[0]).toBeCloseTo(1.1, 10)
    const X = Array.from({ length: 14 }, (_, i) => [Math.sin(i) * 2, Math.cos(1.3 * i), 5 + (i % 5)])
    const pose = {
      rvec: [0.02, -0.15, 0.01] as [number, number, number],
      tvec: [-1, 0.2, 0.1] as [number, number, number],
    }
    const p1 = V.projectPoints(X, { rvec: [0, 0, 0], tvec: [0, 0, 0] }, K)
    const p2 = V.projectPoints(X, pose, K)
    const F = V.findFundamentalMat(p1, p2, { method: 'all' }).model
    for (const [i, a] of p1.entries()) {
      const b = p2[i] as [number, number]
      expect(V.sampsonDistance(a, b, F)).toBeLessThan(1e-12)
    }
    const lines = V.computeCorrespondEpilines(p1, 1, F)
    for (const [i, [a, b, c]] of lines.entries()) {
      const [x, y] = p2[i] as [number, number]
      expect(Math.abs(a * x + b * y + c)).toBeLessThan(1e-6)
    }
  })
})

describe('calibrated geometry and the camera', () => {
  it('recovers the relative pose and triangulates exactly', () => {
    const X = Array.from({ length: 12 }, (_, i) => [Math.sin(2 * i) * 2, Math.cos(1.7 * i), 5 + (i % 4)])
    const pose = {
      rvec: [0.05, -0.1, 0.02] as [number, number, number],
      tvec: [-1, 0, 0.1] as [number, number, number],
    }
    const p1 = V.projectPoints(X, { rvec: [0, 0, 0], tvec: [0, 0, 0] }, K)
    const p2 = V.projectPoints(X, pose, K)
    const E = V.findEssentialMat(p1, p2, K, { method: 'all' }).model
    const r = V.recoverPose(E, p1, p2, K)
    expect(r.good).toBe(12)
    const R = V.rodrigues(pose.rvec)
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(r.R[i]?.[j]).toBeCloseTo(R[i]?.[j] ?? 0, 7)
    const n = Math.hypot(...pose.tvec)
    for (let i = 0; i < 3; i++) expect(r.t[i]).toBeCloseTo((pose.tvec[i] ?? 0) / n, 7)
    const { R1, R2 } = V.decomposeEssentialMat(E)
    for (const Q of [R1, R2]) {
      const qtq = [0, 1, 2].map((i) =>
        [0, 1, 2].map((j) => [0, 1, 2].reduce((s, k) => s + (Q[k]?.[i] ?? 0) * (Q[k]?.[j] ?? 0), 0)),
      )
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(qtq[i]?.[j]).toBeCloseTo(i === j ? 1 : 0, 10)
    }
    const P1 = K.map((row) => [...row, 0])
    const Rt = R.map((row, i) => [...row, pose.tvec[i] ?? 0])
    const P2 = K.map((row) => [0, 1, 2, 3].map((j) => row.reduce((s, v, k) => s + v * (Rt[k]?.[j] ?? 0), 0)))
    const Xt = V.triangulatePoints(P1, P2, p1, p2)
    for (const [i, p] of Xt.entries()) for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(X[i]?.[k] ?? 0, 7)
    const F = V.fundamentalFromEssential(E, K)
    const E2 = V.essentialFromFundamental(F, K)
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(E2[i]?.[j]).toBeCloseTo(E[i]?.[j] ?? 0, 10)
  })

  it('undistorts what it distorts and solves for poses', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -0.3, max: 0.1, noNaN: true }),
        fc.double({ min: -0.003, max: 0.003, noNaN: true }),
        (k1, p1) => {
          const d = [k1, 0.02, p1, -p1 / 2]
          const X = Array.from({ length: 10 }, (_, i) => [(i % 5) * 0.2 - 0.4, Math.floor(i / 5) * 0.3 - 0.15, 0])
          const pose = {
            rvec: [0.1, 0.2, -0.05] as [number, number, number],
            tvec: [0, 0, 2] as [number, number, number],
          }
          const img = V.projectPoints(X, pose, K, d)
          const und = V.undistortPoints(img, K, d, { iterations: 50 })
          for (const [i, [x, y]] of und.entries()) {
            const [a, b, c] = X[i] as number[]
            const R = V.rodrigues(pose.rvec)
            const xc = (R[0]?.[0] ?? 0) * (a ?? 0) + (R[0]?.[1] ?? 0) * (b ?? 0) + (R[0]?.[2] ?? 0) * (c ?? 0) + 0
            const yc = (R[1]?.[0] ?? 0) * (a ?? 0) + (R[1]?.[1] ?? 0) * (b ?? 0) + (R[1]?.[2] ?? 0) * (c ?? 0) + 0
            const zc = (R[2]?.[0] ?? 0) * (a ?? 0) + (R[2]?.[1] ?? 0) * (b ?? 0) + (R[2]?.[2] ?? 0) * (c ?? 0) + 2
            expect(x).toBeCloseTo(xc / zc, 6)
            expect(y).toBeCloseTo(yc / zc, 6)
          }
          const est = V.solvePnP(X, img, K, d)
          for (let i = 0; i < 3; i++) expect(est.tvec[i]).toBeCloseTo(pose.tvec[i] ?? 0, 6)
          expect(V.reprojectionError(X, img, est, K, d)).toBeLessThan(1e-6)
        },
      ),
      { numRuns: 8 },
    )
  })

  it('maps, remaps and rectifies', () => {
    const { mapX, mapY } = V.initUndistortRectifyMap(K, [], [6, 4])
    const img = Array.from({ length: 4 }, (_, i) => Array.from({ length: 6 }, (_, j) => i * 6 + j))
    expect(V.remap(img, mapX, mapY).shape).toEqual([4, 6])
    expect(V.undistortImage(img, K, []).shape).toEqual([4, 6])
    const r = V.stereoRectify(K, [], K, [], [160, 120], V.rodrigues([0, 0.05, 0]), [-0.1, 0, 0])
    expect(r.P2[0]?.[3]).toBeLessThan(0)
    expect(r.Q[3]?.[2]).toBeGreaterThan(0)
  })
})
