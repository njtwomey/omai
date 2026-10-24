import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, type Value, zeros } from '../../tensor/index.ts'
import {
  dualAveragingInit,
  dualAveragingUpdate,
  massMatrixFinal,
  massMatrixInit,
  massMatrixUpdate,
  windowSchedule,
} from './adaptation.ts'

type Params = Readonly<Record<string, unknown>>

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => Value>> = {
  dualAveraging: ([a], p) => {
    let s = dualAveragingInit(p['initialStepSize'] as number)
    const out: number[][] = []
    for (const alpha of (a as Tensor).data as ArrayLike<number> as Float64Array) {
      s = dualAveragingUpdate(s, alpha, { targetAcceptance: p['targetAcceptance'] as number })
      out.push([s.logStepSize, s.logStepSizeAvg])
    }
    return tensor(out)
  },
  windowSchedule: (_, p) => {
    const s = windowSchedule(p['numSteps'] as number)
    return s.length === 0 ? zeros([0, 2]) : tensor(s.map((x) => [x.slow ? 1 : 0, x.windowEnd ? 1 : 0]))
  },
  massMatrix: ([draws], p) => {
    const d = draws as Tensor
    const n = d.shape[0] ?? 0
    const dim = d.shape[1] ?? 0
    const dense = p['dense'] as boolean
    let s = massMatrixInit(dim, { dense })
    for (let i = 0; i < n; i++) s = massMatrixUpdate(s, (d.data as Float64Array).subarray(i * dim, (i + 1) * dim))
    const imm = Array.from(massMatrixFinal(s).inverseMassMatrix)
    return dense ? tensor(Array.from({ length: dim }, (_, i) => imm.slice(i * dim, (i + 1) * dim))) : tensor(imm)
  },
}

describe('sampling/hamiltonian/adaptation against BlackJAX', () => {
  for (const c of loadFixture('sampling/hamiltonian/adaptation').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(
        call(
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
