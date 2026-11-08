import { classificationLosses, penalties } from 'omai-engine/losses'
import { defaults, dict } from 'omai-engine/space'
import { modes } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { greyOptions } from './space-controls.ts'
import { admissibleNames, modeSlot, settleAdmissible, tableRules, tableSlot } from './validity.ts'

const loss = tableSlot(classificationLosses, (p) =>
  p.smooth ? undefined : 'not differentiable: a gradient method stalls',
)
const penalty = tableSlot(penalties, (p, name) => (p.smooth ? undefined : `${name} has a kink at zero`))

describe('validity from the option tables', () => {
  it('reads the reason of an option from its properties in the table', () => {
    expect(loss.reason('logistic')).toBeUndefined()
    expect(loss.reason('hinge')).toMatch(/not differentiable/)
    expect(penalty.reason('l1')).toBe('l1 has a kink at zero')
    expect(penalty.reason('none')).toBeUndefined()
    expect(loss.reason('nonsense')).toBeUndefined()
  })

  it('lists the options nothing rules out, in the table order', () => {
    expect(admissibleNames(penalty)).toEqual(['none', 'l2'])
    expect(admissibleNames(loss)).toEqual(classificationLosses.where((p) => p.smooth))
  })

  it('greys exactly the inadmissible options of the right slot through the controls rule', () => {
    const rule = tableRules({ loss, penalty })
    const grey = greyOptions(penalties.names, ['penalty', 'kind'], rule)
    expect(grey).toHaveLength(penalties.names.length)
    expect(grey.filter((o) => typeof o === 'object').map((o) => (o as { value: string }).value)).toEqual([
      'l1',
      'elasticNet',
    ])
    expect(grey.find((o) => typeof o === 'object' && o.value === 'l1')).toMatchObject({
      disabled: true,
      description: 'l1 has a kink at zero',
    })
    // The same option names under another path are not ruled, nor is a knob of the slot.
    expect(greyOptions(['l1'], ['other', 'kind'], rule)).toEqual(['l1'])
    expect(greyOptions(['l1'], ['penalty', 'alpha'], rule)).toEqual(['l1'])
  })
})

describe('validity from a table of modes', () => {
  const noop = {
    space: dict({}),
    make: () => {
      throw new Error('unused')
    },
    params: () => ({}),
  }
  const solvers = modes<unknown, Record<string, never>>(
    {
      any: noop,
      prox: { ...noop, needs: { penalty: ['l1', 'elasticNet'], loss: ['squared'] } },
    } as never,
    { label: 'solver' },
  )

  it('rules out the modes whose needs the other slots do not meet, naming what they need', () => {
    const slot = modeSlot(solvers, { penalty: { kind: 'l2' }, loss: 'squared' } as never)
    expect(slot.reason('any')).toBeUndefined()
    expect(slot.reason('prox')).toBe('needs penalty: l1 or elasticNet, loss: squared')
    const met = modeSlot(solvers, { penalty: { kind: 'l1' }, loss: 'squared' } as never)
    expect(met.reason('prox')).toBeUndefined()
    expect(admissibleNames(slot)).toEqual(['any'])
  })
})

describe('settling a value onto the admissible options', () => {
  const space = dict({ loss: classificationLosses.space, penalty: penalties.space })

  it('moves a slot whose option no longer applies to the first that does, with that option’s defaults', () => {
    const sparse = tableSlot(penalties, (p) => (p.sparse ? undefined : 'dense'))
    const value = { loss: { kind: 'hinge' }, penalty: { kind: 'l2', alpha: 0.5 } }
    expect(settleAdmissible(space, value, { penalty: sparse })).toEqual({
      loss: { kind: 'hinge' },
      penalty: { ...(defaults(penalties.space.options['l1'] as never) as object), kind: 'l1' },
    })
  })

  it('leaves a value whose options all apply as it is', () => {
    const value = { loss: { kind: 'logistic' }, penalty: { kind: 'l1', alpha: 0.5 } }
    expect(
      settleAdmissible(space, value, { penalty: tableSlot(penalties, (p) => (p.sparse ? undefined : 'dense')) }),
    ).toBe(value)
  })
})
