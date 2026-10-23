import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { item, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as G from './index.ts'

const arr = (t: Tensor): number[][] => {
  const f = toFlat(t)
  const c = t.shape[1] ?? 1
  return Array.from({ length: t.shape[0] ?? 0 }, (_, i) => f.slice(i * c, i * c + c))
}

describe('polygons and hulls against shapely and scipy', () => {
  for (const c of loadFixture('numerics/geometry/polygons').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const a = t[0] as Tensor
      const actual: Value | number[] | number =
        c.fn === 'area'
          ? G.polygonArea(a)
          : c.fn === 'centroid'
            ? G.polygonCentroid(a)
            : c.fn === 'perimeter'
              ? G.polygonPerimeter(a)
              : c.fn === 'pointInPolygon'
                ? G.pointInPolygon(a, t[1] as Tensor)
                : c.fn === 'clipArea'
                  ? G.polygonArea(G.polygonClip(a, t[1] as Tensor))
                  : c.fn === 'hull2'
                    ? [...G.convexHull(a)].sort((x, y) => x - y)
                    : c.fn === 'hull2Area'
                      ? G.polygonArea(G.convexHull(a).map((i) => arr(a)[i] as number[]))
                      : c.fn === 'hull3'
                        ? [...G.convexHull3(a).vertices]
                        : c.fn === 'hull3Volume'
                          ? G.convexHull3(a).volume
                          : c.fn === 'hull3Area'
                            ? G.convexHull3(a).area
                            : G.polygonArea(G.minkowskiSum(a, t[1] as Tensor))
      expectMatches(
        typeof actual === 'number' || Array.isArray(actual) ? tensor(actual) : actual,
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})

describe('distances against scipy.spatial.distance', () => {
  for (const c of loadFixture('numerics/geometry/distances').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const metric = p['metric'] as G.Metric
      const actual =
        c.fn === 'cdist'
          ? G.cdist(t[0] as Tensor, t[1] as Tensor, metric, p['p'] === undefined ? {} : { p: p['p'] as number })
          : c.fn === 'pdist'
            ? G.pdist(t[0] as Tensor, metric)
            : c.fn === 'cdistWeighted'
              ? G.cdist(t[0] as Tensor, t[1] as Tensor, 'minkowski', { p: 3, w: t[2] as Tensor })
              : G.squareform(t[0] as Tensor)
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('transforms against scipy and skimage', () => {
  for (const c of loadFixture('numerics/geometry/transforms').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const q = (i: number): G.Rotation => G.rotationFromQuat(toFlat(t[i] as Tensor))
      const p = c.params
      const actual: Value | number[] | number[][] =
        c.fn === 'asMatrix'
          ? q(0).asMatrix()
          : c.fn === 'asRotvec'
            ? q(0).asRotvec()
            : c.fn === 'apply'
              ? q(0).apply(t[1] as Tensor)
              : c.fn === 'fromMatrix'
                ? G.rotationFromMatrix(t[0] as Tensor).asQuat(true)
                : c.fn === 'fromRotvec'
                  ? G.rotationFromRotvec(toFlat(t[0] as Tensor)).asQuat(true)
                  : c.fn === 'asEuler'
                    ? q(0).asEuler(p['seq'] as string)
                    : c.fn === 'fromEuler'
                      ? G.rotationFromEuler(p['seq'] as string, toFlat(t[0] as Tensor)).asQuat(true)
                      : c.fn === 'compose'
                        ? q(0).compose(q(1)).asQuat(true)
                        : c.fn === 'inverse'
                          ? q(0).inv().asQuat(true)
                          : c.fn === 'slerp'
                            ? G.slerp(q(0), q(1), p['t'] as number).asQuat(true)
                            : c.fn === 'orthogonalProcrustes'
                              ? G.orthogonalProcrustes(t[0] as Tensor, t[1] as Tensor).rotation
                              : c.fn === 'orthogonalProcrustesScale'
                                ? G.orthogonalProcrustes(t[0] as Tensor, t[1] as Tensor).scale
                                : c.fn === 'procrustesMtx2'
                                  ? G.procrustes(t[0] as Tensor, t[1] as Tensor).mtx2
                                  : c.fn === 'procrustesDisparity'
                                    ? G.procrustes(t[0] as Tensor, t[1] as Tensor).disparity
                                    : c.fn === 'kabsch'
                                      ? G.kabsch(t[0] as Tensor, t[1] as Tensor).rotation
                                      : c.fn === 'kabschRmsd'
                                        ? G.kabsch(t[0] as Tensor, t[1] as Tensor).rmsd
                                        : c.fn === 'homography'
                                          ? G.homography(t[0] as Tensor, t[1] as Tensor)
                                          : G.fundamentalMatrix(t[0] as Tensor, t[1] as Tensor)
      expectMatches(Array.isArray(actual) ? tensor(actual as number[]) : actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('contours against skimage.measure.find_contours', () => {
  for (const c of loadFixture('numerics/geometry/contours').cases) {
    it(c.name, () => {
      const z = decode(c.inputs[0]) as Tensor
      const paths = G.contourLines(z, c.params['level'] as number, {
        fullyConnected: c.params['connected'] as 'low' | 'high',
      })
      const actual = c.fn === 'count' ? tensor(paths.length) : tensor(paths[c.params['index'] as number] ?? [])
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('geometry properties', () => {
  it('area and centroid are differentiable in the vertices', () => {
    const square = [
      [0, 0],
      [2, 0.1],
      [2.2, 1.9],
      [-0.1, 2],
    ]
    expect(checkPrimitive((v) => G.polygonArea(v), [square]).failures).toEqual([])
    expect(checkPrimitive((v) => G.polygonCentroid(v), [square]).failures).toEqual([])
    expect(
      checkPrimitive(
        (a, b) => G.cdist(a, b, 'euclidean'),
        [
          [
            [0, 1],
            [2, 3],
          ],
          [
            [1, 1],
            [0.5, 2],
          ],
        ],
      ).failures,
    ).toEqual([])
  })

  it('rotations round-trip through every representation', () => {
    const r = G.rotationFromEuler('ZYX', [0.3, -0.7, 1.2])
    for (const seq of ['xyz', 'ZYX', 'zxz', 'XZX']) {
      const back = G.rotationFromEuler(seq, r.asEuler(seq))
      for (const [i, v] of back.asQuat(true).entries()) expect(v).toBeCloseTo(r.asQuat(true)[i] ?? 0, 12)
    }
    const m = G.rotationFromMatrix(r.asMatrix())
    expect(m.compose(r.inv()).magnitude()).toBeCloseTo(0, 12)
    expect(G.rotationFromRotvec(r.asRotvec()).compose(r.inv()).magnitude()).toBeCloseTo(0, 12)
  })

  it('homographies and affine maps invert; Kabsch recovers a rigid motion exactly', () => {
    const src = [
      [0, 0],
      [3, 0],
      [3, 2],
      [0, 2],
      [1, 1],
    ]
    const h = [
      [1.2, 0.1, 2],
      [0, 0.8, -1],
      [0.02, 0.01, 1],
    ]
    const estimate = G.homography(src, G.applyHomography(h, src))
    for (const [i, row] of estimate.entries())
      for (const [j, v] of row.entries()) expect(v).toBeCloseTo(h[i]?.[j] ?? 0, 10)
    const map = G.affine(
      [
        [2, 1],
        [0, 1],
      ],
      [1, -1],
    )
    expect(toFlat(map.inverse().apply(map.apply([[0.5, 0.25]])))).toEqual([0.5, 0.25])
    const p = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
    ]
    const rot = G.rotationFromEuler('xyz', [0.4, 0.1, -0.3])
    const moved = G.affine(rot.asMatrix(), [1, 2, 3]).apply(p)
    const k = G.kabsch(p, moved)
    expect(item(k.rmsd)).toBeLessThan(1e-12)
    for (const [i, v] of toFlat(k.translation).entries()) expect(v).toBeCloseTo(i + 1, 12)
  })

  it('contour levels are nice and ellipses hold their probability', () => {
    expect(G.contourLevels([[-1.3, 4.2]], 5)).toEqual([0, 2, 4])
    expect(G.contourLevels([[-1.3, 4.2]], 6)).toEqual([-1, 0, 1, 2, 3, 4])
    const e = G.covarianceEllipse(
      [1, 2],
      [
        [2, 0.6],
        [0.6, 1],
      ],
      { confidence: 0.95, points: 64 },
    )
    // Every point satisfies (x - mu)^T Sigma^-1 (x - mu) = chi2(0.95, 2).
    // The inverse covariance [[a, b], [b, c]].
    const d = 2 - 0.36
    const [a, b, c] = [1 / d, -0.6 / d, 2 / d]
    const pts = toFlat(e.points)
    for (let i = 0; i < 64; i++) {
      const dx = (pts[2 * i] ?? 0) - 1
      const dy = (pts[2 * i + 1] ?? 0) - 2
      expect(a * dx * dx + 2 * b * dx * dy + c * dy * dy).toBeCloseTo(-2 * Math.log(0.05), 10)
    }
  })
})
