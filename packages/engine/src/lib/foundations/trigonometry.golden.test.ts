import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as F from './index.ts'

const asParts = (t: F.Triangle | undefined): Tensor =>
  tensor(t === undefined ? [] : [t.a, t.b, t.c, t.A, t.B, t.C, t.area])

goldenSuite('trigonometry against numpy and sympy.geometry', 'foundations/trigonometry', {
  radians: ([x]) => F.radians(x),
  degrees: ([x]) => F.degrees(x),
  wrapSigned: ([x]) => F.wrapAngle(x, 'signed'),
  wrapPositive: ([x]) => F.wrapAngle(x, 'positive'),
  directionAngle: ([y, x]) => F.directionAngle(y, x),
  unitCirclePoint: ([x]) => F.unitCirclePoint(x),
  sss: ([s]) => {
    // sympy's sss is (c, a, b) in the labelling of the fixture.
    const [c, a, b] = toFlat(s) as [number, number, number]
    return asParts(F.solveTriangle({ a, b, c })[0])
  },
  sas: ([s]) => {
    const [b, A, c] = toFlat(s) as [number, number, number]
    return asParts(F.solveTriangle({ b, A, c })[0])
  },
  asa: ([s]) => {
    const [A, c, B] = toFlat(s) as [number, number, number]
    return asParts(F.solveTriangle({ A, c, B })[0])
  },
})
