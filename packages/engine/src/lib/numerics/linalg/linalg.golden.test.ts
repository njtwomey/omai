import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import * as L from './index.ts'

type Params = Readonly<Record<string, unknown>>
type Fn = (xs: Tensor[], p: Params) => Value | number

const order = (p: Params): L.NormOrder | undefined => {
  const o = p['ord'] ?? p['p']
  if (o === undefined) return undefined
  if (o === 'inf') return Number.POSITIVE_INFINITY
  if (o === '-inf') return Number.NEGATIVE_INFINITY
  return typeof o === 'string' && o !== 'fro' && o !== 'nuc' ? Number(o) : (o as L.NormOrder)
}

const FNS: Readonly<Record<string, Fn>> = {
  solveTriangular: ([a, b], p) =>
    L.solveTriangular(a as Tensor, b as Tensor, {
      lower: p['lower'] === true,
      transpose: p['transpose'] === true,
      unitDiagonal: p['unitDiagonal'] === true,
    }),
  cholesky: ([a]) => L.cholesky(a as Tensor),
  choSolve: ([l, b]) => L.choSolve(l as Tensor, b as Tensor),
  choleskyLogdet: ([l]) => L.choleskyLogdet(l as Tensor),
  solve: ([a, b]) => L.solve(a as Tensor, b as Tensor),
  inv: ([a]) => L.inv(a as Tensor),
  lu: ([a], p) => L.lu(a as Tensor)[p['part'] as 'p' | 'l' | 'u'],
  det: ([a]) => L.det(a as Tensor),
  slogdet: ([a], p) => L.slogdet(a as Tensor)[p['part'] as 'sign' | 'logabsdet'],
  cond: ([a], p) => L.cond(a as Tensor, order(p)),
  matrixPower: ([a], p) => L.matrixPower(a as Tensor, p['k'] as number),
  norm: ([a], p) => L.norm(a as Tensor, order(p)),
  matrixRank: ([a]) => L.matrixRank(a as Tensor),
  qr: ([a], p) => L.qr(a as Tensor)[p['part'] as 'q' | 'r'],
  lstsq: ([a, b]) => L.lstsq(a as Tensor, b as Tensor),
  eigh: ([a], p) => L.eigh(a as Tensor)[p['part'] as 'eigenvalues' | 'eigenvectors'],
  eigvalsh: ([a]) => L.eigvalsh(a as Tensor),
  svd: ([a], p) => L.svd(a as Tensor)[p['part'] as 'u' | 's' | 'vh'],
  svdvals: ([a]) => L.svdvals(a as Tensor),
  pinv: ([a]) => L.pinv(a as Tensor),
  lowRankApproximation: ([a], p) => L.lowRankApproximation(a as Tensor, p['rank'] as number),
}

for (const file of ['triangular', 'cholesky', 'lu', 'basics', 'qr', 'eigh', 'svd']) {
  describe(`linalg/${file} against numpy and scipy`, () => {
    for (const c of loadFixture(`numerics/linalg/${file}`).cases) {
      it(c.name, () => {
        const f = FNS[c.fn]
        if (f === undefined) throw new Error(`no function ${c.fn}`)
        expectMatches(
          f(
            c.inputs.map((x) => decode(x) as Tensor),
            c.params,
          ),
          c.output,
          c.rtol,
          c.atol,
          c.name,
        )
      })
    }
  })
}
