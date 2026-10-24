import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const canonical = (q: Value): Tensor => {
  const rows = (q.shape.length === 1 ? [toFlat(q)] : (toArray(q) as number[][])).map((r) =>
    (r[3] ?? 0) < 0 ? r.map((v) => -v) : r,
  )
  return tensor(q.shape.length === 1 ? (rows[0] ?? []) : rows)
}

goldenSuite('rotations against scipy.spatial.transform', 'foundations/rotations', {
  elementaryX: ([a]) => F.elementaryRotation('x', a),
  elementaryY: ([a]) => F.elementaryRotation('y', a),
  elementaryZ: ([a]) => F.elementaryRotation('z', a),
  axisAngleToMatrix: ([axis, angle]) => F.axisAngleToMatrix(axis, angle),
  rotateAboutAxis: ([v, axis, angle]) => F.rotateAboutAxis(v, axis, angle as Tensor),
  matrixToRotvec: ([m]) => {
    const { axis, angle } = F.matrixToAxisAngle(m)
    return tensor(axis.map((v) => v * angle))
  },
  eulerToMatrix: ([a], p) => F.eulerToMatrix(p['seq'] as string, a),
  matrixToEuler: ([m], p) => tensor(F.matrixToEuler(p['seq'] as string, m)),
  quatMultiply: ([p, q]) => canonical(F.quatMultiply(p, q)),
  quatRotate: ([q, v]) => F.quatRotate(q, v),
  quatToMatrix: ([q]) => F.quatToMatrix(q),
  quatFromAxisAngle: ([axis, angle]) => canonical(F.quatFromAxisAngle(axis, angle)),
  matrixToQuat: ([m]) => tensor(F.matrixToQuat(m)),
  quatSlerp: ([p, q], o) => canonical(F.quatSlerp(p, q, o['t'] as number)),
  rotationDistance: ([p, q]) => F.rotationDistance(p, q),
  meanRotation: ([q]) => tensor(F.meanRotation(q)),
  meanRotationWeighted: ([q, w]) => tensor(F.meanRotation(q, w)),
})
