// Unary functions of a random value (exp, square, tanh, relu, softplus, sigmoid) compiled to `apply` factors and run under EP,
// against the exact posterior of a model with two scalar latents on a fine grid (`gridPosterior` of the compiled program).
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
} from '../../../message-passing/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import { gridPosterior } from '../../validation/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, factorGraphVerdict, ops, toFactorGraph, toProgram } from '../index.ts'

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

const XS = [-1.6, -1, -0.5, 0, 0.4, 0.9, 1.5]

type Builder = Parameters<Parameters<typeof defineModel>[1]>[0]

/** A model `y_i ~ N(g(a + b x_i), noise)` with `a ~ N(0, 1)`, `b ~ N(0, 1.5)`. */
function curve(name: string, g: (u: never) => never, noise: number): ReturnType<typeof defineModel> {
  return defineModel(name, (m: Builder) => {
    const row = m.range('row')
    const x = m.data('x', { dims: [row] })
    const a = m.latent('a', d.normal(0, 1))
    const b = m.latent('b', d.normal(0, 1.5))
    m.observe('y', d.normal(g(a.add(b.mul(x)) as never), noise), m.data('y', { dims: [row] }))
  })
}

function run(
  def: ReturnType<typeof defineModel>,
  y: readonly number[],
): { means: Record<string, number>; sds: Record<string, number> } & {
  ref: Record<string, { mean: number; sd: number }>
} {
  const bound = def.bind({ x: XS, y })
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = graphOf(c.graph)
  expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-8, damping: 0.3 }), { maxSteps: 600 })
  const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
  const means: Record<string, number> = {}
  const sds: Record<string, number> = {}
  const ref: Record<string, { mean: number; sd: number }> = {}
  const grid = gridPosterior(toProgram(bound), undefined, { points: 81, maxRefinements: 1 })
  for (const name of ['a', 'b']) {
    means[name] = toFlat(post.mean(name))[0] as number
    sds[name] = post.summary().find((s) => s.site === name)?.sd as number
    const m = grid.marginal(name)
    ref[name] = { mean: m.mean, sd: m.sd }
  }
  return { means, sds, ref }
}

const cases: readonly { name: string; g: (u: never) => never; y: number[]; noise: number; sdLow?: number }[] = [
  {
    name: 'tanh',
    g: (u) => ops.tanh(u) as never,
    y: XS.map((x) => Math.tanh(0.3 + 1.2 * x) + 0.05 * Math.sin(7 * x)),
    noise: 0.25,
  },
  {
    name: 'exp',
    g: (u) => ops.exp(u) as never,
    y: XS.map((x) => Math.exp(0.2 + 0.7 * x) + 0.1 * Math.cos(5 * x)),
    noise: 0.4,
  },
  {
    name: 'sigmoid',
    g: (u) => ops.sigmoid(u) as never,
    y: XS.map((x) => 1 / (1 + Math.exp(-(0.5 + 1.5 * x)))),
    noise: 0.15,
  },
  {
    name: 'softplus',
    g: (u) => ops.softplus(u) as never,
    y: XS.map((x) => Math.log1p(Math.exp(0.1 + 1.1 * x))),
    noise: 0.3,
  },
  {
    name: 'relu',
    g: (u) => ops.maximum(u, 0) as never,
    y: XS.map((x) => Math.max(0.2 + 0.9 * x, 0) + 0.05 * Math.sin(9 * x)),
    noise: 0.3,
    sdLow: 0.55,
  },
]

describe('unary functions of random values under expectation propagation', () => {
  for (const c of cases)
    it(`${c.name}: EP agrees with the exact posterior of a two-latent curve (means within a third of an sd, sds within 30%${c.sdLow === undefined ? '' : ', a rectifier more overconfident: its posterior has a kink'})`, () => {
      const out = run(curve(`curve-${c.name}`, c.g, c.noise), c.y)
      for (const name of ['a', 'b']) {
        const ref = out.ref[name] as { mean: number; sd: number }
        expect(Math.abs((out.means[name] as number) - ref.mean) / ref.sd, `mean of ${name}`).toBeLessThan(
          c.sdLow === undefined ? 0.34 : 0.4,
        )
        expect((out.sds[name] as number) / ref.sd, `sd of ${name}`).toBeGreaterThan(c.sdLow ?? 0.7)
        expect((out.sds[name] as number) / ref.sd, `sd of ${name}`).toBeLessThan(1.3)
      }
    })

  it('a known operand makes min and max a rectifier: max(x, c) = c + relu(x - c)', () => {
    const def = curve('clip', (u) => ops.minimum(ops.maximum(u, -0.5), 1) as never, 0.3)
    const c = toFactorGraph(def.bind({ x: XS, y: XS.map((x) => Math.min(Math.max(0.2 + 0.9 * x, -0.5), 1)) }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'apply').length).toBeGreaterThan(0)
  })

  it('the banana: the squared coordinate pushes forward to the exact mean and variance of x1', () => {
    const def = defineModel('banana-ep', (m: Builder) => {
      const x0 = m.latent('x0', d.normal(0, 2))
      m.latent('x1', d.normal(ops.mul(0.5, ops.sub(x0.square(), 4)), 1))
    })
    const c = toFactorGraph(def.bind({}))
    const g = graphOf(c.graph)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-9 }), { maxSteps: 200 })
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    const sd = post.summary().find((s) => s.site === 'x1')?.sd as number
    expect(toFlat(post.mean('x1'))[0]).toBeCloseTo(0, 6)
    // The exact sd of x1 is sqrt(2 * 0.25 * 16 + 1) = 3.
    expect(sd).toBeCloseTo(3, 5)
  })

  it('the non-centred funnel: x = exp(v / 2) z has the log-normal scale mixture moments of v and z', () => {
    const def = defineModel('funnel-ep', (m: Builder) => {
      const v = m.latent('v', d.normal(0, 3))
      const z = m.latent('z', d.normal(0, 1))
      m.derived('x', ops.mul(ops.exp(ops.mul(v, 0.5)), z), { record: true } as never)
    })
    const c = toFactorGraph(def.bind({}))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-9 }), { maxSteps: 300 })
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    expect(toFlat(post.mean('v'))[0]).toBeCloseTo(0, 6)
    expect(post.summary().find((s) => s.site === 'v')?.sd).toBeCloseTo(3, 5)
  })
})
