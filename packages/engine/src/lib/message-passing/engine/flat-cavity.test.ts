// The engine's one policy for a flat cavity (no information yet at the start of a run): an operator that `needsDensity` is shown a very
// wide Gaussian and its message is taken back against the flat cavity, so the first message is the factor's own predictive.
import { describe, expect, it } from 'vitest'
import * as M from '../index.ts'
import { FLAT_CAVITY_PRECISION } from './cavity.ts'
import { operatorMessage } from './dispatch.ts'
import { momentProjection } from './projections.ts'

const operators = M.defaultOperators()
const factor = (name: string, params: M.Params = {}): Parameters<typeof operatorMessage>[0] =>
  ({ operator: operators[name], params, power: 1, ports: [] }) as never

const flat = M.messageArg(M.gaussian, M.gaussian.flat)

describe('the flat-cavity policy of the engine', () => {
  it('is declared by exactly the operators that integrate over their cavities (a deliberate list)', () => {
    const names = Object.entries(operators)
      .filter(([, op]) => op.needsDensity === true)
      .map(([name]) => name)
      .sort()
    expect(names).toEqual(['gate', 'gaussian-gamma-precision', 'mixture-site', 'prior-site'])
  })

  it('gaussian-gamma-precision: the first message to a flat value is the predictive of N(u, 1 / lambda), lambda ~ Gamma(2, 1)', () => {
    const args = [flat, M.messageArg(M.gaussian, M.gaussianEta(1, 1)), M.messageArg(M.gamma, M.gammaEta(2, 1))]
    const p = operatorMessage(factor('gaussian-gamma-precision'), momentProjection, 0, args)
    expect(p.kind).toBe('message')
    if (p.kind !== 'message') return
    // E[1 / lambda] = 1 and u has variance 1: the value has variance 2 about 1, against a flat cavity.
    expect(p.eta[0]).toBeCloseTo(0.5, 4)
    expect(p.eta[1]).toBeCloseTo(0.5, 4)
  })

  it('mixture-site: the first message to a flat value is the mixture itself, moment matched', () => {
    const params = { weights: [0.5, 0.5], means: [0, 0], sds: [3, 0.1] }
    const p = operatorMessage(factor('mixture-site', params), momentProjection, 0, [flat])
    expect(p.kind).toBe('message')
    if (p.kind !== 'message') return
    expect(p.eta[0]).toBeCloseTo(0, 6)
    // variance 0.5 * 9 + 0.5 * 0.01, against a cavity 1e6 wide that the engine takes back
    expect(1 / (p.eta[1] as number)).toBeCloseTo(4.505, 3)
  })

  it('an operator that does not declare it sees the flat cavity as it is (and an improper one is still improper)', () => {
    const args = [M.messageArg(M.gaussian, [0, -1]), M.valueArg(M.gaussian, 1)]
    const p = operatorMessage(factor('mixture-site', { weights: [1], means: [0], sds: [1] }), momentProjection, 0, [
      args[0] as M.Arg,
    ])
    expect(p.kind).toBe('blocked')
    expect(FLAT_CAVITY_PRECISION).toBeLessThan(1e-5)
  })
})
