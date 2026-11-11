// The integral of gaussian-gamma-precision is kept in the engine's cache for the stretch it asks for (no module-global state): the
// evidence and the message of the same cavities share one grid, the values are the ones without a cache, and a cache is never needed.
import { describe, expect, it } from 'vitest'
import * as M from '../../index.ts'

const args = [
  M.valueArg(M.gaussian, 1),
  M.messageArg(M.gaussian, M.gaussianEta(0, 1)),
  M.messageArg(M.gamma, M.gammaEta(2, 1)),
]
const op = M.defaultOperators()['gaussian-gamma-precision'] as M.FactorOperator
const ctx = (to: number, memo?: M.Memo): M.OperatorContext => ({
  mode: 'ep',
  to,
  args,
  params: {},
  power: 1,
  ...(memo === undefined ? {} : { memo }),
})

describe('the cache of the precision tilt', () => {
  it('holds one integral for the evidence and the messages of the same cavities, with the values of no cache', () => {
    const memo: M.Memo = new Map()
    const withCache = [op.logEvidence?.(ctx(0, memo)), op.ep?.(ctx(1, memo)).message, op.ep?.(ctx(2, memo)).message]
    expect(memo.size).toBe(1)
    const without = [op.logEvidence?.(ctx(0)), op.ep?.(ctx(1)).message, op.ep?.(ctx(2)).message]
    expect(withCache).toEqual(without)
  })

  it('is keyed by the cavities: a different cavity is another entry', () => {
    const memo: M.Memo = new Map()
    op.ep?.(ctx(1, memo))
    op.ep?.({
      ...ctx(1, memo),
      args: [args[0] as M.Arg, M.messageArg(M.gaussian, M.gaussianEta(0.5, 1)), args[2] as M.Arg],
    })
    expect(memo.size).toBe(2)
  })
})
