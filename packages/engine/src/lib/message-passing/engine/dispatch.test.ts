import { describe, expect, it } from 'vitest'
import { messageArg } from '../core/args.ts'
import { defineOperator, messageRule, siteRule } from '../core/define.ts'
import { gaussian, gaussianEta } from '../core/families.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { OperatorContext, TiltedMoments } from '../core/types.ts'
import { operatorMessage, ruleMessage } from './dispatch.ts'
import { momentProjection } from './projections.ts'

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

describe('operatorMessage and a cavity with no density', () => {
  const reader = (improperTarget: boolean): CompiledGraph['factors'][number] =>
    ({
      id: 'f',
      operator: defineOperator({
        name: 'reader',
        description: 'reads its cavity',
        ports,
        rules: { ep: messageRule(() => ({ message: [1, 1] })) },
        traits: { readsTarget: true, ...(improperTarget ? { improperTarget: true as const } : {}) },
      }),
      params: {},
      power: 1,
      ports: [],
    }) as unknown as CompiledGraph['factors'][number]
  const send = (f: CompiledGraph['factors'][number], eta: readonly number[]): string =>
    operatorMessage(f, momentProjection, 0, [messageArg(gaussian, [...eta])]).kind

  it('sends nothing to a port whose cavity has no density when the operator reads it', () => {
    expect(send(reader(false), [0.2, -0.1])).toBe('blocked')
    expect(send(reader(false), [0, 0])).toBe('blocked')
  })

  it('shows an indefinite cavity to an operator that reads it only through its own posterior, but not a flat one', () => {
    expect(send(reader(true), [0.2, -0.1])).toBe('message')
    expect(send(reader(true), [0, 0])).toBe('blocked')
  })
})
