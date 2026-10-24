import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, zeros } from '../tensor/index.ts'
import * as F from './index.ts'

const rows = (vs: readonly (readonly number[])[], n: number): Tensor =>
  vs.length === 0 ? zeros([0, n]) : tensor(vs.map((v) => [...v]))
const width = (a: Tensor): number => a.shape[1] ?? 0
const height = (a: Tensor): number => a.shape[0] ?? 0

const eigenSorted = (a: Tensor): { re: number[]; im: number[] } => {
  const pairs = F.smallEigen(a).flatMap((p) =>
    Array.from({ length: p.algebraicMultiplicity }, () => [p.re, Math.abs(p.im)] as const),
  )
  const seen = [...pairs].sort((x, y) => x[0] - y[0] || x[1] - y[1])
  return { re: seen.map((p) => p[0]), im: seen.map((p) => p[1]) }
}

goldenSuite('linear algebra against sympy, scipy.linalg and numpy', 'foundations/linear-algebra', {
  rref: ([a]) => tensor(F.rref(a).matrix),
  pivotColumns: ([a]) => tensor(F.rref(a).pivotColumns),
  rank: ([a]) => F.rref(a).rank,
  nullSpace: ([a]) => rows(F.fundamentalSubspaces(a).nullSpace, width(a)),
  columnSpace: ([a]) => rows(F.fundamentalSubspaces(a).columnSpace, height(a)),
  leftNullSpace: ([a]) => rows(F.fundamentalSubspaces(a).leftNullSpace, height(a)),
  luP: ([a]) => tensor(F.luSteps(a).p),
  luL: ([a]) => tensor(F.luSteps(a).l),
  luU: ([a]) => tensor(F.luSteps(a).u),
  echelonPartial: ([a]) => tensor(F.rowEchelon(a, { tol: 0 }).matrix),
  detByElimination: ([a]) => F.detByElimination(a).value,
  detByCofactors: ([a]) => F.detByCofactors(a).value,
  characteristicPolynomial: ([a]) => tensor(F.characteristicPolynomial(a).coefficients),
  particular: ([a, b]) => tensor(F.solveByElimination(a, b).particular ?? []),
  leastSquares: ([a, b]) => tensor(F.leastSquaresProjection(a, b).x),
  leastSquaresResidual: ([a, b]) => tensor(F.leastSquaresProjection(a, b).residual),
  projectionMatrix: ([a]) => F.projectionMatrix(a),
  eigenRe: ([a]) => tensor(eigenSorted(a).re),
  eigenIm: ([a]) => tensor(eigenSorted(a).im),
  principalValues: ([a]) => tensor(F.principalAxes(a).eigenvalues),
  quadraticForm: ([a, x]) => F.quadraticForm(a, x),
  singularValues: ([a]) => tensor(F.svdGeometry(a).singularValues),
  svdAxisLengths: ([a]) => tensor(F.svdGeometry(a).axes.map((v) => Math.hypot(...v))),
})
