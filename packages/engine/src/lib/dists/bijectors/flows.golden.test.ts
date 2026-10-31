import { goldenSuite } from '../../../testing/suite.ts'
import { matmul, neg, type Tensor, tanh, type Value } from '../../tensor/index.ts'
import * as B from './index.ts'

const parameters = (uw: Tensor | undefined, uh: Tensor | undefined, ud: Tensor | undefined): B.SplineParameters => ({
  widths: uw as Tensor,
  heights: uh as Tensor,
  derivatives: ud as Tensor,
})

goldenSuite(
  'flow layers against nflows-style splines, scipy LU and finite-difference log-determinants',
  'dists/bijectors/flows',
  {
    'spline.forward': ([x, uw, uh, ud]) => B.rationalQuadraticSplineMap(x, parameters(uw, uh, ud), false).value,
    'spline.forwardLogDet': ([x, uw, uh, ud]) =>
      B.rationalQuadraticSplineMap(x, parameters(uw, uh, ud), false).logDerivative,
    'spline.inverse': ([y, uw, uh, ud]) => B.rationalQuadraticSplineMap(y, parameters(uw, uh, ud), true).value,
    'spline.inverseLogDet': ([y, uw, uh, ud]) => {
      const r = B.rationalQuadraticSplineMap(y, parameters(uw, uh, ud), true)
      return neg(r.logDerivative)
    },
    'lu.logDet': ([m]) => B.luLinear(B.luParameters(m)).forwardLogDetJacobian([0, 0, 0, 0, 0]),
    'lu.forward': ([m, rows]) => B.luLinear(B.luParameters(m)).forward(rows as Tensor),
    'residual.logDet': ([x, w]) =>
      B.residualFlow((v: Value) => tanh(matmul(v, w as Tensor)), { terms: 80 }).forwardLogDetJacobian(x),
  },
)
