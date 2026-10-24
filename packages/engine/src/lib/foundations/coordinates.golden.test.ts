import { goldenSuite } from '../../testing/suite.ts'
import * as F from './index.ts'

goldenSuite('coordinates against numpy, scipy.spatial and sympy', 'foundations/coordinates', {
  cartesianToPolar: ([p]) => F.cartesianToPolar(p),
  polarToCartesian: ([p]) => F.polarToCartesian(p),
  cartesianToCylindrical: ([p]) => F.cartesianToCylindrical(p),
  cartesianToSpherical: ([p]) => F.cartesianToSpherical(p),
  sphericalToCartesian: ([p]) => F.sphericalToCartesian(p),
  barycentric: ([q, simplex]) => F.barycentric(q, simplex),
  coordinatesInBasis: ([v, b]) => F.coordinatesInBasis(v, b),
  changeOfBasis: ([a, b]) => F.changeOfBasis(a, b),
  matrixInBasis: ([a, b]) => F.matrixInBasis(a, b),
  jacobianDeterminant: ([p], o) => F.coordinateJacobianDeterminant(o['kind'] as F.CoordinateSystem, p),
})
