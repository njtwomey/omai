import { describe, expect, it } from 'vitest'
import { messageArg } from '../core/args.ts'
import { defineOperator, messageRule, siteRule } from '../core/define.ts'
import { gaussian, gaussianEta } from '../core/families.ts'
import type { OperatorContext, TiltedMoments } from '../core/types.ts'
import { ruleMessage } from './dispatch.ts'

const ctx = (mode: OperatorContext['mode'], power = 1): OperatorContext => ({
  mode,
  to: 0,
  args: [messageArg(gaussian, gaussianEta(0, 1))],
  params: {},
  power,
})
const ports = [{ name: 'x', accepts: ['gaussian' as const] }]

describe('ruleMessage', () => {
  it('calls the hook of the mode, and the derived hook of a site rule', () => {
    const op = defineOperator({
      name: 'site',
      description: 'd',
      ports,
      rules: { ep: siteRule(() => ({ mean: 1, variance: 0.5 })), vmp: messageRule(() => ({ message: [3, 4] })) },
      traits: { power: true },
    })
    expect(ruleMessage(op, ctx('ep', 2)).message).toEqual([1, 0.5])
    expect(ruleMessage(op, ctx('vmp')).message).toEqual([3, 4])
  })

  it('projects the moments of an operator with only `tilted`, divided by the cavity and the power', () => {
    const tilted = (): TiltedMoments => ({ moments: [1, 1.5], logZ: 0 })
    const op = {
      ...defineOperator({
        name: 'tilt',
        description: 'd',
        ports,
        rules: { vmp: messageRule(() => ({ message: [0, 0] })) },
      }),
      tilted,
    }
    // N(1, 0.5) against the cavity N(0, 1): [2, 1] for a power of 1.
    expect(ruleMessage(op, ctx('ep')).message).toEqual([2, 1])
    expect(ruleMessage(op, ctx('ep', 2)).message).toEqual([1, 0.5])
  })

  it('says which rule is missing', () => {
    const op = defineOperator({
      name: 'only-vmp',
      description: 'd',
      ports,
      rules: { vmp: messageRule(() => ({ message: [0, 0] })) },
    })
    expect(() => ruleMessage(op, ctx('bp'))).toThrow('only-vmp has no bp message rule')
  })
})
