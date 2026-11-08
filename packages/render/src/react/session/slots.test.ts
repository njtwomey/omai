import { smoothSolvers } from 'omai-engine/optim'
import { choice, defaults, dict, union } from 'omai-engine/space'
import type { Value } from 'omai-engine/tensor'
import type { Config } from 'omai-engine/trace'
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

describe('variantsOfSlot knobs', () => {
  it('lays the knobs the page holds for an option over its defaults', () => {
    const variants = variantsOfSlot(space, 'solver', {}, { knobs: { lbfgs: { memory: 4 } }, only: ['lbfgs', 'bfgs'] })
    expect((variants[0] as Variant<Config>).config['solver']).toMatchObject({ kind: 'lbfgs', memory: 4 })
    expect((variants[0] as Variant<Config>).config['solver']).toHaveProperty('gradTol')
    expect((variants[1] as Variant<Config>).config['solver']).toMatchObject({ kind: 'bfgs' })
  })
})

describe('a comparison over the solver slot of a method', () => {
  it('runs one session per solver on one objective, and every deterministic one reaches the same minimum', async () => {
    const { add, matmul, tensor } = await import('omai-engine/tensor')
    const { classificationLosses, penalties } = await import('omai-engine/losses')
    const { gradientContract, gradientDefaults, loggedSolver, solverContract, solverDefaults } = await import(
      'omai-engine/optim'
    )
    const { Log, defineMethod, metricSeries, session } = await import('omai-engine/trace')
    const x = [
      [-1.2, -0.8, 1],
      [-0.9, -1.1, 1],
      [-1.5, -0.3, 1],
      [0.9, 1.1, 1],
      [1.2, 0.7, 1],
      [0.6, 1.3, 1],
    ]
    const design = tensor(x)
    const signs = tensor([-1, -1, -1, 1, 1, 1])
    const loss = classificationLosses.make({ kind: 'logistic' })
    const penalty = penalties.make({ kind: 'l2', alpha: 0.1 })
    // A logistic loss with a ridge penalty, minimised by whichever solver the slot holds.
    const method = defineMethod({
      name: 'test/ridge-logistic',
      config: dict({ solver: smoothSolvers.space }),
      make: (c, key) =>
        loggedSolver(
          { objective: (w: Value): Value => add(loss(matmul(design, w), signs), penalty.value(w)), x0: [0, 0, 0] },
          c['solver'] as Config,
          key,
        ),
      contract: { ...solverContract, ...gradientContract },
      defaults: { ...solverDefaults, ...gradientDefaults, 'solver/objective': Log() },
    })
    const variants = variantsOfSlot(method.config, 'solver')
    expect(variants.map((v) => v.label)).toEqual([...smoothSolvers.names])
    // The stochastic mode at its default learning rate has not arrived in so few steps; the rest have.
    const finals = variants
      .filter((v) => v.label !== 'gradient')
      .map((v) => {
        const s = session(method, { mode: 'eager', steps: 300, config: v.config as never })
        return metricSeries(s.run(), 'solver/objective').values.at(-1) as number
      })
    for (const f of finals) expect(f).toBeCloseTo(finals[0] as number, 2)
  })
})
