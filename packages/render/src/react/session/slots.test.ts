import { smoothSolvers } from 'mdml-engine/optim'
import { choice, defaults, dict, union } from 'mdml-engine/space'
import type { Config } from 'mdml-engine/trace'
import { describe, expect, it } from 'vitest'
import { variantsOfSlot } from './slots.ts'
import type { Variant } from './useSession.ts'

const space = dict({
  loss: choice(['logistic', 'hinge']),
  solver: smoothSolvers.space,
  model: union({ a: dict({ inner: smoothSolvers.space }), b: dict({}) }, { tag: 'kind' }),
})

describe('variantsOfSlot', () => {
  it('gives one variant per option of the union, labelled by name, with the option’s defaults', () => {
    const variants = variantsOfSlot(space, 'solver')
    expect(variants.map((v) => v.label)).toEqual([...smoothSolvers.names])
    for (const [i, v] of variants.entries()) {
      const name = smoothSolvers.names[i] as string
      expect(Object.keys(v.config)).toEqual(['solver'])
      expect(v.config['solver']).toEqual({
        ...(defaults(smoothSolvers.space.options[name] as never) as object),
        kind: name,
      })
    }
  })

  it('keeps the base’s knobs for the option the base names, and restricts and labels on request', () => {
    const base = { solver: { kind: 'lbfgs', memory: 3 } }
    const variants = variantsOfSlot(space, ['solver'], base as never, {
      only: ['lbfgs', 'gradientDescent'],
      label: (n) => n.toUpperCase(),
    })
    expect(variants.map((v) => v.label)).toEqual(['LBFGS', 'GRADIENTDESCENT'])
    expect(((variants[0] as Variant<Config>).config['solver'] as { memory: number }).memory).toBe(3)
    expect(((variants[1] as Variant<Config>).config['solver'] as { memory?: number }).memory).toBeUndefined()
    expect(variantsOfSlot(space, 'solver', {}, { admissible: (n) => n !== 'lbfgs' })).toHaveLength(
      smoothSolvers.names.length - 1,
    )
  })

  it('replaces the whole top-level value for a union nested in another', () => {
    const [first] = variantsOfSlot(space, 'model.inner')
    expect(first?.config['model']).toMatchObject({ kind: 'a', inner: { kind: smoothSolvers.names[0] } })
  })

  it('refuses a path that is not a union', () => {
    expect(() => variantsOfSlot(space, 'loss')).toThrow(/not a union/)
    expect(() => variantsOfSlot(space, 'nothing')).toThrow(/not a union/)
  })
})
