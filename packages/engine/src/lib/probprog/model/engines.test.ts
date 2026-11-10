// Every existing engine that applies to a compiled spec runs it: the engines of the program layer need no change to run
// a model written in the builder. `chain` needs a `markov` loop and has no spec counterpart yet (recurrences are a
// reserved node kind), so it is the one engine not exercised here.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { analyse } from '../analysis.ts'
import { abcEngine } from '../engines/abc.ts'
import { conjugateEngine } from '../engines/conjugate.ts'
import { emEngine } from '../engines/em.ts'
import type { InferenceEngine } from '../engines/engine.ts'
import { enumerationEngine } from '../engines/enumeration.ts'
import { gaussianEngine } from '../engines/gaussian.ts'
import { gibbsEngine } from '../engines/gibbs.ts'
import { importanceEngine } from '../engines/importance.ts'
import { laplaceEngine } from '../engines/laplace.ts'
import { nutsEngine } from '../engines/nuts.ts'
import { smcEngine } from '../engines/smc.ts'
import { sviEngine } from '../engines/svi.ts'
import { explainPlan, infer, plan } from '../infer.ts'
import type { Posterior } from '../posterior.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toProgram } from './index.ts'

// mu ~ N(0, 2); y_i ~ N(mu, 1.5): conjugate and linear-Gaussian, posterior mean 0.2 * 4 ... computed in closed form below.
const Y = [0.3, -1.2, 2.1, 0.8]
const normalMean = defineModel('normal-mean', (b) => {
  const y = b.data('y', { dims: ['data'] })
  b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), y)
})
const exactMean = (() => {
  const precision = 1 / 4 + Y.length / 2.25
  return Y.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactSd = Math.sqrt(1 / (1 / 4 + Y.length / 2.25))

// A two-component mixture with learnable locations: a discrete latent per row and a param for the locations.
const mixture = defineModel('mixture', (b) => {
  const y = b.data('y', { dims: ['row'] })
  const locs = b.param('locs', [-1, 1], { dims: ['k'] })
  const z = b.latent('z', d.categorical({ probs: [0.4, 0.6] }), { dims: ['row'] })
  b.observe('y', d.normal(locs.at(z), 1), y)
})

const run = (e: InferenceEngine, program: () => unknown, k: number): Posterior => {
  const a = analyse(program, undefined)
  return e.run(program, undefined, a, { key: key(k) })
}

describe('engines run a compiled spec', () => {
  const compiled = toProgram(normalMean.bind({ y: Y }))

  it('conjugate, gaussian and laplace find the closed-form posterior', () => {
    for (const e of [conjugateEngine(), gaussianEngine(), laplaceEngine()]) {
      const post = run(e, compiled, 1)
      expect(post.status, e.name).not.toBe('failed')
      expect(Math.abs((post.mean('mu').data[0] as number) - exactMean), `${e.name} mean`).toBeLessThan(1e-4)
    }
  })

  it('nuts, gibbs, smc, importance and svi agree with it within sampling error', () => {
    const engines: [InferenceEngine, number][] = [
      [nutsEngine({ numChains: 2, numWarmup: 150, numDraws: 300 }), 0.25],
      [gibbsEngine({ numWarmup: 100, numDraws: 400 }), 0.3],
      [smcEngine({ numParticles: 400 }), 0.3],
      [importanceEngine({ numSamples: 1500 }), 0.3],
      [sviEngine({ numSteps: 400 }), 0.3],
    ]
    for (const [e, tol] of engines) {
      const post = run(e, compiled, 2)
      expect(post.status, e.name).not.toBe('failed')
      expect(Math.abs((post.mean('mu').data[0] as number) - exactMean), `${e.name} mean`).toBeLessThan(tol)
    }
    expect(exactSd).toBeGreaterThan(0.3)
  })

  it('abc runs on it too', () => {
    const post = run(abcEngine({ numParticles: 60, numGenerations: 2 }), compiled, 3)
    expect(post.status).not.toBe('failed')
    expect(post.draws?.['mu']).toBeDefined()
  })

  it('the planner ranks the engines for a spec and chooses the conjugate one', () => {
    const p = plan(compiled, undefined)
    expect(p.entries.find((e) => e.engine.name === 'conjugate')?.feasibility.applies).toBe(true)
    expect(explainPlan(p)).toContain('conjugate')
    expect(infer(compiled, undefined, { key: key(1) }).engine).toBe('conjugate')
  })

  it('enumeration and em run a discrete mixture with param locations', () => {
    const m = toProgram(mixture.bind({ y: [-1.2, 0.9, 1.1, -0.8] }, { sizes: { k: 2 } }))
    const e = run(enumerationEngine(), m, 4)
    expect(e.status).not.toBe('failed')
    expect(e.engine).toBe('enumeration')
    const em = run(emEngine({ maxIter: 30 }), m, 5)
    expect(em.status).not.toBe('failed')
    expect(em.engine).toBe('em')
  })

  it('nuts sums a finite discrete latent out of the density', () => {
    const noParam = defineModel('mixture-continuous', (b) => {
      const y = b.data('y', { dims: ['row'] })
      const mu = b.latent('mu', d.normal(0, 3), { dims: ['k'] })
      const z = b.latent('z', d.categorical({ probs: [0.4, 0.6] }), { dims: ['row'] })
      b.observe('y', d.normal(mu.at(z), 1), y)
    })
    const m = toProgram(noParam.bind({ y: [-2, 2.2, 1.9, -1.8] }, { sizes: { k: 2 } }))
    const post = run(nutsEngine({ numChains: 1, numWarmup: 60, numDraws: 40 }), m, 6)
    expect(post.status).not.toBe('failed')
    expect(post.draws?.['mu']?.shape).toEqual([40, 2])
    expect(post.draws?.['z']?.shape).toEqual([40, 4])
  })

  it('a model written with derived and operator nodes runs the same way', () => {
    const lin = defineModel('line', (b) => {
      const x = b.data('x', { dims: ['row'] })
      const y = b.data('y', { dims: ['row'] })
      const a = b.latent('a', d.normal(0, 3))
      const s = b.latent('s', d.halfNormal(2))
      b.observe('y', d.normal(ops.add(a, x.mul(2)), s), y)
    })
    const post = run(
      nutsEngine({ numChains: 1, numWarmup: 80, numDraws: 80 }),
      toProgram(lin.bind({ x: [0, 1, 2, 3], y: [0.5, 2.4, 4.4, 6.6] })),
      7,
    )
    expect(post.status).not.toBe('failed')
    expect(post.draws?.['s']?.shape).toEqual([80])
  })
})
