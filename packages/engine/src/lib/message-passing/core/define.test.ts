import { describe, expect, it } from 'vitest'
import { messageArg, siteMessage } from './args.ts'
import { defineOperator, messageRule, projectSite, siteRule, tiltedRule } from './define.ts'
import { ImproperMessageError, ParameterError } from './errors.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from './families.ts'
import { param } from './params.ts'
import type { OperatorContext } from './types.ts'

const port = { name: 'x', accepts: ['gaussian' as const] }
const ctx = (power = 1): OperatorContext => ({
  mode: 'ep',
  to: 0,
  args: [messageArg(gaussian, gaussianEta(0, 1))],
  params: {},
  power,
})

describe('projectSite', () => {
  it('is the projection divided by the cavity, then by the power', () => {
    const target = messageArg(gamma, gammaEta(2, 1))
    const moments = [0.4, 1.8]
    const site = siteMessage(gamma, gammaEta(2, 1), moments)
    expect(projectSite(target, { moments }, 1)).toEqual(site)
    expect(projectSite(target, { moments }, 2)).toEqual(site.map((v) => v / 2))
  })

  it('projects a Gaussian mean and variance as the quotient of natural parameters', () => {
    expect(projectSite(messageArg(gaussian, gaussianEta(0, 1)), { mean: 1, variance: 0.5 }, 1)).toEqual([2, 1])
  })

  it('refuses a variance with no density as an improper message', () => {
    expect(() => projectSite(messageArg(gaussian, gaussianEta(0, 1)), { mean: 1, variance: 0 }, 1)).toThrow(
      ImproperMessageError,
    )
  })
})

describe('defineOperator', () => {
  const declared = (power: boolean) =>
    defineOperator({
      name: 'toy',
      description: 'a toy site',
      ports: [port],
      params: {},
      rules: { ep: siteRule(() => ({ mean: 1, variance: 0.5 })) },
      evidence: {
        improperIsNaN: true,
        ep: (c) => {
          if (c.power === 3) throw new ImproperMessageError('no density')
          return 7
        },
      },
      traits: { readsTarget: true, power },
    })

  it('derives support from the kinds of the rules', () => {
    const op = defineOperator({
      name: 'mixed',
      description: 'one of each kind',
      ports: [port],
      rules: {
        ep: siteRule(() => ({ mean: 0, variance: 1 })),
        vmp: messageRule(() => ({ message: [0, 0] }), 'bound'),
        bp: messageRule(() => ({ message: [0, 0] })),
      },
    })
    expect(op.support).toEqual({ ep: 'numeric', vmp: 'bound', bp: 'exact' })
    expect(
      defineOperator({
        name: 'closed',
        description: 'd',
        ports: [port],
        rules: { ep: messageRule(() => ({ message: [0, 1] })) },
      }).support,
    ).toEqual({ ep: 'exact' })
  })

  it('keeps the tilted moments of a tilted rule as the `tilted` member, with no ep hook', () => {
    const op = defineOperator({
      name: 'tilt',
      description: 'd',
      ports: [port],
      rules: { ep: tiltedRule(() => ({ moments: [1, 1.5], logZ: 0 })) },
    })
    expect(op.support).toEqual({ ep: 'numeric' })
    expect(op.ep).toBeUndefined()
    expect(op.tilted?.(ctx()).moments).toEqual([1, 1.5])
  })

  it('derives the ep hook from a site rule, dividing by the power only for an operator that takes one', () => {
    expect(declared(true).ep?.(ctx(1)).message).toEqual([2, 1])
    expect(declared(true).ep?.(ctx(2)).message).toEqual([1, 0.5])
    expect(declared(false).ep?.(ctx(2)).message).toEqual([2, 1])
  })

  it('a power declared as a function of the parameters divides a site rule only where it is true', () => {
    const byParams = defineOperator({
      name: 'by-params',
      description: 'a toy site whose power depends on the parameters',
      ports: [port],
      params: { on: param.integer({ optional: true }) },
      rules: { ep: siteRule(() => ({ mean: 1, variance: 0.5 })) },
      traits: { readsTarget: true, power: (params) => params['on'] === 1 },
    })
    expect(byParams.ep?.({ ...ctx(2), params: { on: 1 } }).message).toEqual([1, 0.5])
    expect(byParams.ep?.({ ...ctx(2), params: { on: 0 } }).message).toEqual([2, 1])
  })

  it('gives NaN, not an error, for the evidence while a message has no density yet; other errors propagate', () => {
    expect(declared(true).logEvidence?.(ctx(1))).toBe(7)
    expect(declared(true).logEvidence?.(ctx(3))).toBeNaN()
    const broken = defineOperator({
      name: 'broken',
      description: 'd',
      ports: [port],
      rules: { ep: siteRule(() => ({ mean: 0, variance: 1 })) },
      evidence: {
        ep: () => {
          throw new ParameterError('a mistake')
        },
      },
    })
    expect(() => broken.logEvidence?.(ctx())).toThrow(ParameterError)
  })

  it('lets an improper evidence propagate unless the operator asks for NaN', () => {
    const plain = defineOperator({
      name: 'plain',
      description: 'd',
      ports: [port],
      rules: { ep: siteRule(() => ({ mean: 0, variance: 1 })) },
      evidence: {
        ep: () => {
          throw new ImproperMessageError('no density')
        },
      },
    })
    expect(() => plain.logEvidence?.(ctx())).toThrow(ImproperMessageError)
  })

  it('keeps only the members it was given', () => {
    const op = defineOperator({
      name: 'bare',
      description: 'd',
      ports: [port],
      rules: { vmp: messageRule(() => ({ message: [0, 0] })) },
    })
    expect(Object.keys(op).sort()).toEqual(['description', 'name', 'ports', 'support', 'vmp'])
  })

  it('refuses a contradictory declaration when the operator is built', () => {
    const base = { name: 'bad', description: 'd', ports: [port] }
    const rule = messageRule(() => ({ message: [0, 0] }))
    expect(() => defineOperator({ ...base, rules: {} })).toThrow(/no rule/)
    expect(() => defineOperator({ ...base, name: '', rules: { ep: rule } })).toThrow(/name is empty/)
    expect(() => defineOperator({ ...base, rules: { ep: rule }, noRule: { ep: 'because' } })).toThrow(/noRule\.ep/)
    expect(() => defineOperator({ ...base, rules: { ep: rule }, evidence: { vmp: () => 0 } })).toThrow(/vmp evidence/)
    expect(() => defineOperator({ ...base, rules: { vmp: rule }, evidence: { ep: () => 0 } })).toThrow(/ep evidence/)
    expect(() => defineOperator({ ...base, rules: { ep: rule }, traits: { defines: 0.5 } })).toThrow(/defines/)
  })
})
