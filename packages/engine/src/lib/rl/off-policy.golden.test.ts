import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { transitionsOf } from '../../testing/rl-tabular.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { clippedIps, directMethod, doublyRobust, ips, type LoggedBandit, snips, tabularFqe } from './off-policy.ts'

describe('off-policy evaluation against numpy formulas', () => {
  for (const c of loadFixture('rl/off_policy').cases) {
    it(c.name, () => {
      const input = (i: number): unknown => toArray(decode(c.inputs[i]) as Tensor)
      if (c.fn === 'tabularFqe') {
        const rows = (input(0) as number[][]).map((r) => [...r, 0])
        const policy = input(1) as number[][]
        const result = tabularFqe(transitionsOf(rows), policy, {
          nStates: 5,
          nActions: 2,
          gamma: c.params['gamma'] as number,
          tol: 1e-13,
        })
        expectMatches(tensor(result.Q.flat()), c.output, c.rtol, c.atol, c.name)
        return
      }
      const data: LoggedBandit<number> = {
        contexts: input(0) as number[],
        actions: input(1) as number[],
        rewards: input(2) as number[],
        propensities: input(3) as number[],
      }
      const target = input(4) as number[][]
      const model = input(5) as number[][]
      const pi = (x: number): readonly number[] => target[x] ?? []
      const rhat = (x: number, a: number): number => model[x]?.[a] ?? 0
      const maxWeight = c.params['maxWeight'] as number | undefined
      const est =
        c.fn === 'ips'
          ? ips(data, pi)
          : c.fn === 'snips'
            ? snips(data, pi)
            : c.fn === 'clippedIps'
              ? clippedIps(data, pi, { maxWeight: maxWeight ?? 10 })
              : c.fn === 'directMethod'
                ? directMethod(data, pi, rhat)
                : doublyRobust(data, pi, rhat, maxWeight === undefined ? {} : { maxWeight })
      expectMatches(tensor([est.value, est.standardError ?? Number.NaN]), c.output, c.rtol, c.atol, c.name)
    })
  }
})
