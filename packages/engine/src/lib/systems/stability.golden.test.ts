import { describe, it } from 'vitest'
import { type GoldenFn, runFixture, sortedParts } from '../../testing/systems.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const FNS: Readonly<Record<string, GoldenFn>> = {
  'poles-ss': ([a, b, c, d]) => sortedParts(S.poles(S.ss(a as Tensor, b as Tensor, c as Tensor, d))),
  'zeros-ss': ([a, b, c, d]) => sortedParts(S.zeros(S.ss(a as Tensor, b as Tensor, c as Tensor, d))),
  'poles-tf': ([n, d]) => sortedParts(S.poles(S.tf(n as Tensor, d as Tensor))),
  'zeros-tf': ([n, d]) => sortedParts(S.zeros(S.tf(n as Tensor, d as Tensor))),
  rootLocus: ([n, d], p) => {
    const r = S.rootLocus(S.tf(n as Tensor, d as Tensor), [p['gain'] as number])
    return sortedParts(r.roots[0] ?? { re: [], im: [] })
  },
  routh: ([c]) => tensor(S.routh(c as Tensor).signChanges),
  jury: ([c]) => tensor(S.jury(c as Tensor).stable ? 1 : 0),
}

describe('systems/stability against python-control and numpy', () => {
  runFixture('systems/stability', FNS, it)
})
