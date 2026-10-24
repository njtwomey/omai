import { goldenSuite } from '../../testing/suite.ts'
import { maxOf } from '../base/index.ts'
import { type Tensor, tensor, toFlat, zeros } from '../tensor/index.ts'
import * as F from './index.ts'

const ints = (t: Tensor): number[] => toFlat(t)
const pairs = (t: Tensor): [number, number][] => {
  const v = ints(t)
  return Array.from({ length: v.length / 2 }, (_, i) => [v[2 * i] ?? 0, v[2 * i + 1] ?? 0])
}
const text = (xs: readonly bigint[]): string => xs.join(',')
const grid = (rows: number[][], width: number): Tensor => (rows.length === 0 ? zeros([0, width]) : tensor(rows))
const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i)

goldenSuite('combinatorics against sympy and itertools', 'foundations/combinatorics', {
  factorial: ([n]) => text(ints(n).map(F.factorialExact)),
  binomial: ([nk]) => text(pairs(nk).map(([n, k]) => F.binomialExact(n, k))),
  multinomial: ([ks]) => F.multinomialExact(ints(ks)).toString(),
  permutations: ([nr]) => {
    const [n = 0, r = 0] = ints(nr)
    return grid(F.permutations(range(n), r), r)
  },
  combinations: ([nr]) => {
    const [n = 0, r = 0] = ints(nr)
    return grid(F.combinations(range(n), r), r)
  },
  permutationRank: ([p]) => F.permutationRank(ints(p)).rank.toString(),
  permutationUnrank: ([nr]) => {
    const [n = 0, r = 0] = ints(nr)
    return tensor(F.permutationUnrank(n, r))
  },
  combinationRank: ([v]) => {
    const [n = 0, ...subset] = ints(v)
    return F.combinationRank(subset, n).toString()
  },
  combinationUnrank: ([v]) => {
    const [n = 0, k = 0, r = 0] = ints(v)
    return tensor(F.combinationUnrank(n, k, r))
  },
  derangements: ([n]) => text(ints(n).map(F.derangements)),
  stirling2: ([nk]) => text(pairs(nk).map(([n, k]) => F.stirling2(n, k))),
  stirling1: ([nk]) => text(pairs(nk).map(([n, k]) => F.stirling1(n, k))),
  stirling1Signed: ([nk]) => text(pairs(nk).map(([n, k]) => F.stirling1(n, k, true))),
  bell: ([n]) => text(ints(n).map(F.bell)),
  catalan: ([n]) => text(ints(n).map(F.catalan)),
  partitionCount: ([n]) => text(ints(n).map(F.partitionCount)),
  integerPartitions: ([n]) => {
    const parts = F.integerPartitions(ints(n)[0] ?? 0)
    const width = maxOf(parts.map((p) => p.length))
    return tensor(parts.map((p) => [...p, ...Array.from({ length: width - p.length }, () => 0)]))
  },
  inclusionExclusion: ([flat]) => {
    const v = ints(flat)
    const families: number[][] = []
    for (let i = 0; i < v.length; ) {
      const len = v[i] ?? 0
      families.push(v.slice(i + 1, i + 1 + len))
      i += len + 1
    }
    return F.inclusionExclusion(families).union
  },
  pascal: ([n]) => tensor(F.pascalTriangle(ints(n)[0] ?? 0).flat()),
  pascalMod3: ([n]) => tensor(F.pascalTriangle(ints(n)[0] ?? 0, 3).flat()),
})
