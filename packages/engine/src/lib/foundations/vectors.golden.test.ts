import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as F from './index.ts'

const v = (t: Tensor | undefined): number[] => (t === undefined ? [] : toFlat(t))
const projector = (rows: readonly (readonly number[])[], n: number): Tensor =>
  tensor(
    Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => rows.reduce((s, q) => s + (q[i] ?? 0) * (q[j] ?? 0), 0)),
    ),
  )
const plane = (p: Tensor | undefined, n: Tensor | undefined): F.Plane => ({ point: v(p), normal: v(n) })
const line = (p: Tensor | undefined, d: Tensor | undefined): F.Line => ({ point: v(p), direction: v(d) })

goldenSuite('vectors, lines and planes against numpy, scipy and sympy.geometry', 'foundations/vectors', {
  dotProduct: ([a, b]) => F.dotProduct(a, b),
  crossProduct: ([a, b]) => F.crossProduct(a, b),
  tripleProduct: ([a, b, c]) => F.tripleProduct(a, b, c as Tensor),
  vectorTripleProduct: ([a, b, c]) => F.vectorTripleProduct(a, b, c as Tensor),
  magnitude: ([a]) => F.magnitude(a),
  angleBetween: ([a, b]) => F.angleBetween(a, b),
  projection: ([a, b]) => F.projection(a, b),
  rejection: ([a, b]) => F.rejection(a, b),
  spanProjector: ([m]) => projector(F.orthonormalise(m).basis, m.shape[1] ?? 0),
  spanDimension: ([m]) => F.spanDimension(m),
  pointLineDistance: ([x, p, d]) => F.pointLineDistance(x, line(p, d)),
  pointPlaneDistance: ([x, p, n]) => F.pointPlaneDistance(x, plane(p, n)),
  linePlaneIntersection: ([p, d, q, n]) => tensor(F.linePlaneIntersection(line(p, d), plane(q, n))?.point ?? []),
  lineLineDistance: ([p, d, q, e]) => F.closestPoints(line(p, d), line(q, e)).distance,
  planePlanePoint: ([p, n, q, m]) => tensor(F.planeIntersection(plane(p, n), plane(q, m))?.point ?? []),
  planePlaneProjector: ([p, n, q, m]) => {
    const l = F.planeIntersection(plane(p, n), plane(q, m))
    const d = l?.direction ?? []
    const dd = d.reduce((s, x) => s + x * x, 0)
    return projector([d.map((x) => x / Math.sqrt(dd))], 3)
  },
})
