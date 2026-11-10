// fromProgram: the shared programs of the probprog tests lift to specs whose density and gradient are the program's (or are
// refused with a reason), the parameters that move with the random values become opaque custom nodes, and a program whose
// structure depends on random values is refused with the first difference.
import { describe, expect, it } from 'vitest'
import { DISCRETE, MODELS } from '../../../testing/probprog.ts'
import { bernoulli, normal, poisson } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { item, type Value } from '../../tensor/index.ts'
import { logJoint, potentialEnergy } from '../density.ts'
import { seed, traceOf } from '../handlers.ts'
import { observe, plate, sample } from '../index.ts'
import type { Model } from '../primitives.ts'
import { analyseSpec, fromProgram, structureReport, toProgram, toStructure } from './index.ts'

/** The latent values of one run of a program. */
function latentValues(program: Model): Record<string, Value> {
  const t = traceOf(seed(program, key(5)), undefined)
  const out: Record<string, Value> = {}
  for (const [n, s] of t.sites) if (s.type === 'sample' && !s.isObserved) out[n] = s.value
  return out
}

const programs: Record<string, Model> = { ...MODELS, ...DISCRETE }

/** What the lift does with each shared program: lifted, or refused with a reason that says why. */
const EXPECTED: Record<string, true | RegExp> = {
  normalMean: true,
  schools: true,
  logistic: true,
  gammaPoisson: true,
  dirichletCategorical: true,
  betaFactor: /factor site/,
  scaleMask: /scaled or masked/,
  subsampled: /subsampled/,
  nested: true,
  bounded: true,
  simplex: true,
  mvn: true,
  mixture: /not a family of the spec layer/,
  hmm: true,
}

describe('fromProgram on the shared programs', () => {
  for (const [name, program] of Object.entries(programs))
    it(`${name}: ${EXPECTED[name] === true ? "lifts, with the program's log joint" : 'is refused with a reason'}`, () => {
      const expected = EXPECTED[name]
      expect(expected, `no expectation written for ${name}`).toBeDefined()
      if (expected !== true) {
        expect(() => fromProgram(program, undefined)).toThrow(expected)
        return
      }
      const lifted = fromProgram(program, undefined)
      const values = latentValues(program)
      const original = item(logJoint(program, undefined, values).logJoint)
      const compiled = item(logJoint(toProgram(lifted.definition.bind({})), undefined, values).logJoint)
      expect(compiled).toBeCloseTo(original, 10)
      // The same sites, in order.
      const sites = [...traceOf(seed(program, key(0)), undefined).sites.values()]
        .filter((s) => s.type !== 'plate' && s.type !== 'markov')
        .map((s) => s.name)
      expect(lifted.spec.nodes.map((n) => n.name)).toEqual(sites)
      // It draws and is analysed.
      expect(toStructure(lifted.definition).nodes.length).toBeGreaterThan(0)
      expect(analyseSpec(lifted.spec).latent.length).toBe(lifted.spec.nodes.filter((n) => n.kind === 'latent').length)
    })

  it("the gradient through the opaque nodes is the program's: the custom nodes stay differentiable", () => {
    const program = programs['schools'] as Model
    const lifted = fromProgram(program, undefined)
    const a = potentialEnergy(program, undefined).logDensity
    const b = potentialEnergy(toProgram(lifted.definition.bind({})), undefined).logDensity
    const u = Float64Array.from({ length: a.dim }, (_, i) => 0.3 * Math.sin(i + 1))
    const ga = a.valueAndGrad(u)
    const gb = b.valueAndGrad(u)
    expect(gb.value).toBeCloseTo(ga.value, 10)
    for (const [i, g] of gb.grad.entries()) expect(g).toBeCloseTo(ga.grad[i] as number, 9)
  })
})

describe('what is constant and what is opaque', () => {
  it('a parameter that does not move with the random values is a constant; one that does is a custom node naming its sites', () => {
    const program = programs['normalMean'] as Model
    const lifted = fromProgram(program, undefined)
    expect(lifted.constants).toEqual(['mu.loc', 'mu.scale', 'y.scale'])
    expect(lifted.opaque).toEqual([{ node: 'y', parameter: 'loc', custom: 'y.loc', dependsOn: ['mu'] }])
    expect(lifted.spec.customs).toEqual(['y.loc'])
    expect(lifted.notes.join(' ')).toMatch(/opaque custom nodes \(y\.loc\)/)
    // Message passing and conjugacy through the mean do not see through it.
    const a = analyseSpec(lifted.spec)
    expect(a.opaque).toContain('y')
    expect(a.conjugate).toEqual([])
  })

  it('a chain of dependent sites is a chain of opaque nodes over the sites each depends on', () => {
    const lifted = fromProgram(DISCRETE['hmm'] as Model, undefined)
    expect(lifted.opaque.find((o) => o.node === 'z1')?.dependsOn).toEqual(['z0'])
    expect(lifted.opaque.find((o) => o.node === 'y1')?.dependsOn).toEqual(['z1'])
  })

  it('plates become named dimensions with their sizes, in the order the program nests them', () => {
    const lifted = fromProgram(programs['nested'] as Model, undefined)
    expect(lifted.spec.dims).toEqual([
      { name: 'b', size: 3 },
      { name: 'a', size: 2 },
    ])
    expect(lifted.spec.nodes.map((n) => n.dims)).toEqual([
      ['b', 'a'],
      ['b', 'a'],
    ])
  })
})

describe('a structure that depends on random values is refused', () => {
  it('a site that exists only on one branch', () => {
    const branching: Model = () => {
      const z = sample('z', bernoulli({ probs: 0.5 }))
      if (item(z) > 0.5) sample('extra', normal(0, 1))
      observe('y', normal(0, 1), 0.3)
    }
    expect(() => fromProgram(branching, undefined, { trials: 12 })).toThrow(/structure depends on random values/)
    expect(() => fromProgram(branching, undefined, { trials: 12 })).toThrow(/extra|y: observed/)
  })

  it('a plate whose size is drawn', () => {
    const sized: Model = () => {
      const n = Math.round(item(sample('n', poisson(3)))) + 1
      plate('p', n, () => observe('y', normal(0, 1), 0.3))
    }
    expect(() => fromProgram(sized, undefined, { trials: 12 })).toThrow(/depends on random values/)
  })

  it('a program with a fixed structure lifts whatever the number of runs', () => {
    expect(() => fromProgram(programs['schools'] as Model, undefined, { trials: 2 })).not.toThrow()
  })
})

describe("the lifted spec's structure report", () => {
  it('claims no conjugacy through an opaque parameter and draws', () => {
    const lifted = fromProgram(programs['gammaPoisson'] as Model, undefined)
    const r = structureReport(lifted.definition)
    // The Poisson rate is a function of the gamma site: opaque, so no conjugacy is claimed.
    expect(r.conjugateBlocks).toEqual([])
    expect(analyseSpec(lifted.spec).opaque).toContain('k')
  })
})
