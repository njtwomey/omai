// The model layer against the program layer: specs written again for the shared MODELS give the same log joint at
// random points, the JSON round trip is the identity, binding catches what it should, constraints use the existing
// bijectors, missing values and batches compile, and the spec is drawn as a structure.
import { describe, expect, it } from 'vitest'
import { MODELS } from '../../../testing/probprog.ts'
import { bijectors } from '../../dists/index.ts'
import { foldIn, key, normal as normalDraws } from '../../random/index.ts'
import { add, eye, full, item, mul, type Tensor, take, tensor, toFlat, type Value, zeros } from '../../tensor/index.ts'
import { logJoint, potentialEnergy } from '../density.ts'
import { seed, traceOf } from '../handlers.ts'
import type { Model } from '../primitives.ts'
import * as d from './dists/index.ts'
import {
  analyseSpec,
  boundFromJSON,
  boundToJSON,
  constant,
  defineModel,
  elementId,
  exprText,
  factorId,
  fromJSON,
  type ModelDefinition,
  nodeId,
  ops,
  plateId,
  texName,
  toProgram,
  toStructure,
  validateSpec,
} from './index.ts'

const Y4 = [0.3, -1.2, 2.1, 0.8]
const SIGMA = [15, 10, 16, 11]
const SCHOOLS = [28, 8, -3, 7]
const X = [
  [0.5, -1],
  [1.2, 0.3],
  [-0.7, 0.8],
  [0.1, -0.4],
  [1.5, 1.1],
]
const YB = [1, 0, 0, 1, 1]
const COUNTS = [3, 5, 4, 6]
const CATS = [0, 2, 2, 1, 2]
const FLIPS = [1, 0, 1, 1]
const SIMPLEX_DATA = [3, 0, 2, 3, 1, 3]

/** Random values of every latent of `names` from the unconstrained normal pushed through the support's bijector. */
function randomValues(spec: Model, i: number): Record<string, Tensor> {
  const proto = traceOf(seed(spec, key(100 + i)), undefined)
  const values: Record<string, Tensor> = {}
  let n = 0
  for (const [name, site] of proto.sites) {
    if (site.type !== 'sample' || site.isObserved) continue
    const draw = site.fn.sample(foldIn(key(200 + i), n++), [])
    values[name] = draw as Tensor
  }
  return values
}

/** Log joints of a spec and of a hand-written program at the same random points. */
function expectSameLogJoint(def: ModelDefinition, data: Record<string, unknown>, hand: Model, points = 8): void {
  const program = toProgram(def.bind(data as never))
  for (let i = 0; i < points; i++) {
    const values = randomValues(hand, i)
    const a = item(logJoint(program, undefined, values).logJoint)
    const b = item(logJoint(hand, undefined, values).logJoint)
    expect(Math.abs(a - b), `${def.name} at point ${i}: ${a} vs ${b}`).toBeLessThan(1e-12)
  }
}

describe('specs against the shared programs', () => {
  it('normalMean', () => {
    const m = defineModel('normalMean', (b) => {
      const y = b.data('y', { dims: ['data'] })
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), y)
    })
    expectSameLogJoint(m, { y: Y4 }, MODELS['normalMean'] as Model)
  })

  it('schools (a hierarchy over a plate, data as a scale)', () => {
    const m = defineModel('schools', (b) => {
      const J = b.range('J')
      const y = b.data('y', { dims: [J] })
      const sigma = b.data('sigma', { dims: [J], support: 'positive' })
      const mu = b.latent('mu', d.normal(0, 5))
      const tau = b.latent('tau', d.halfCauchy(5))
      const theta = b.latent('theta', d.normal(mu, tau), { dims: [J] })
      b.observe('y', d.normal(theta, sigma), y)
    })
    expectSameLogJoint(m, { y: SCHOOLS, sigma: SIGMA }, MODELS['schools'] as Model)
  })

  it('logistic (matmul and add over named dimensions)', () => {
    const m = defineModel('logistic', (b) => {
      const n = b.range('data')
      const f = b.range('features')
      const x = b.data('X', { dims: [n, f] })
      const y = b.data('y', { dims: [n], support: 'binary' })
      const alpha = b.latent('alpha', d.normal(0, 1))
      const beta = b.latent('beta', d.normal(0, 1), { dims: [f] })
      b.observe('y', d.bernoulli({ logits: x.matmul(beta).add(alpha) }), y)
    })
    expectSameLogJoint(m, { X, y: YB }, MODELS['logistic'] as Model)
  })

  it('gammaPoisson (a derived node is a deterministic site)', () => {
    const m = defineModel('gammaPoisson', (b) => {
      const k = b.data('k', { dims: ['data'], support: 'count' })
      const rate = b.latent('rate', d.gamma(2, 0.5))
      b.derived('double', rate.mul(2))
      b.observe('k', d.poisson(rate), k)
    })
    expectSameLogJoint(m, { k: COUNTS }, MODELS['gammaPoisson'] as Model)
    const t = traceOf(seed(toProgram(m.bind({ k: COUNTS })), key(0)), undefined)
    expect([...t.sites].filter(([, x]) => x.type !== 'plate').map(([n]) => n)).toEqual(['rate', 'double', 'k'])
  })

  it('dirichletCategorical and simplex (event axes)', () => {
    const m = defineModel('dirichletCategorical', (b) => {
      const z = b.data('z', { dims: ['data'], support: 'index' })
      const p = b.latent('p', d.dirichlet([1, 2, 3]))
      b.observe('z', d.categorical({ probs: p }), z)
    })
    expectSameLogJoint(m, { z: CATS }, MODELS['dirichletCategorical'] as Model)
    const s = defineModel('simplex', (b) => {
      const z = b.data('z', { dims: ['data'], support: 'index' })
      b.observe('z', d.categorical({ probs: b.latent('w', d.dirichlet([2, 1, 1.5, 3])) }), z)
    })
    expectSameLogJoint(s, { z: SIMPLEX_DATA }, MODELS['simplex'] as Model)
  })

  it('betaFactor (a factor node)', () => {
    const m = defineModel('betaFactor', (b) => {
      const y = b.data('y', { dims: ['data'], support: 'binary' })
      const p = b.latent('p', d.beta(2, 3))
      b.factor('penalty', p.square().mul(-1))
      b.observe('y', d.bernoulli({ probs: p }), y)
    })
    expectSameLogJoint(m, { y: FLIPS }, MODELS['betaFactor'] as Model)
  })

  it('bounded (a constant observation)', () => {
    const m = defineModel('bounded', (b) => {
      const u = b.latent('u', d.uniform(-1, 4))
      const s = b.latent('s', d.logNormal(0, 1))
      b.observe('y', d.normal(u, s), 1.3)
    })
    expectSameLogJoint(m, {}, MODELS['bounded'] as Model)
  })

  it('mvn, with a custom function for the coordinate lookup', () => {
    const m = defineModel('mvn', (b) => {
      const x = b.latent(
        'x',
        d.multivariateNormal([0, 1], {
          covariance: [
            [1, 0.3],
            [0.3, 2],
          ],
        }),
      )
      const pick = b.custom('x0plus2x1', ([v]) => add(take(v?.value as Value, 0), mul(take(v?.value as Value, 1), 2)), [
        x,
      ])
      b.observe('y', d.normal(pick, 1), 0.7)
    })
    expect(m.spec.customs).toEqual(['x0plus2x1'])
    expectSameLogJoint(m, {}, MODELS['mvn'] as Model)
  })
})

const coin = defineModel('coin', (b) => {
  const y = b.data('y', { dims: ['flip'], support: 'binary' })
  b.observe('y', d.bernoulli({ probs: b.latent('p', d.beta(2, 2)) }), y)
})

interface Raw {
  schema: string
  nodes: { dims: string[]; dist: { family: string; params: Record<string, unknown> } }[]
}

const at = (s: Raw, i: number): Raw['nodes'][number] => s.nodes[i] as Raw['nodes'][number]

describe('JSON round trip', () => {
  it('is the identity on specs and keeps the log joint', () => {
    const m = defineModel('schools', (b) => {
      const J = b.range('J', { labels: ['a', 'b', 'c', 'd'] })
      const y = b.data('y', { dims: [J] })
      const mu = b.latent('mu', d.normal(0, 5))
      const tau = b.latent('tau', d.halfCauchy(5), { constraint: { lower: 0 } })
      const theta = b.derived('theta', mu.add(tau.mul(b.latent('z', d.normal(0, 1), { dims: [J] }))))
      b.observe('y', d.normal(theta, ops.exp(0.1)), y)
      b.generated('yRep', d.normal(theta, 1))
    })
    const json = JSON.parse(JSON.stringify(m))
    expect(json).toEqual(m.spec)
    const back = fromJSON(json)
    expect(back.spec).toEqual(m.spec)
    const data = { y: SCHOOLS }
    const values = { mu: tensor(0.4), tau: tensor(1.3), z: tensor([0.1, -0.2, 0.3, 0.5]) }
    const a = item(logJoint(toProgram(m.bind(data)), undefined, values).logJoint)
    const b = item(logJoint(toProgram(back.bind(data)), undefined, values).logJoint)
    expect(a).toBe(b)
  })

  it('carries a bound model, missing values included', () => {
    const bound = coin.bind({ y: [1, Number.NaN, 0] })
    const json = JSON.parse(JSON.stringify(boundToJSON(bound)))
    expect(json.data.y.values).toEqual([1, null, 0])
    const back = boundFromJSON(json)
    expect(toFlat(back.data['y'] as Tensor).map((v) => (Number.isNaN(v) ? 'nan' : v))).toEqual([1, 'nan', 0])
    expect(back.missing['y']?.count).toBe(1)
  })

  it('rejects a spec with an unknown operator, family, reference or schema', () => {
    const spec = JSON.parse(JSON.stringify(coin.spec))
    const bad = (edit: (s: Raw) => void): unknown => {
      const copy = JSON.parse(JSON.stringify(spec))
      edit(copy)
      return () => fromJSON(copy)
    }
    expect(
      bad((s) => {
        s.schema = 'omai.model/9'
      }),
    ).toThrow(/schema/)
    expect(
      bad((s) => {
        at(s, 1).dist.family = 'nope'
      }),
    ).toThrow(/unknown distribution family/)
    expect(
      bad((s) => {
        at(s, 1).dist.params['a'] = { kind: 'ref', name: 'ghost' }
      }),
    ).toThrow(/ghost/)
    expect(
      bad((s) => {
        at(s, 2).dist.params['probs'] = { kind: 'op', op: 'cube', args: [{ kind: 'ref', name: 'p' }] }
      }),
    ).toThrow(/unknown operator/)
    expect(
      bad((s) => {
        at(s, 1).dims = ['ghostDim']
      }),
    ).toThrow(/not declared/)
  })

  it('needs the custom functions a spec names', () => {
    const m = defineModel('c', (b) => {
      const x = b.latent('x', d.normal(0, 1))
      b.observe(
        'y',
        d.normal(
          b.custom('twice', ([v]) => mul(v?.value as Value, 2), [x]),
          1,
        ),
        0.5,
      )
    })
    expect(() => fromJSON(JSON.parse(JSON.stringify(m)))).toThrow(/twice/)
    const back = fromJSON(JSON.parse(JSON.stringify(m)), { twice: ([v]) => mul(v?.value as Value, 2) })
    expect(item(logJoint(toProgram(back.bind({})), undefined, { x: 1 }).logJoint)).toBeCloseTo(
      item(logJoint(toProgram(m.bind({})), undefined, { x: 1 }).logJoint),
      14,
    )
  })
})

describe('binding', () => {
  const regression = defineModel('regression', (b) => {
    const n = b.range('row')
    const f = b.range('feature')
    const x = b.data('X', { dims: [n, f], role: 'covariate' })
    const y = b.data('y', { dims: [n], support: 'binary', role: 'response' })
    const w = b.latent('w', d.normal(0, 1), { dims: [f] })
    b.observe('y', d.bernoulli({ logits: x.matmul(w) }), y)
  })

  it('infers sizes from the data and keeps the data as tensors', () => {
    const bound = regression.bind({ X, y: YB })
    expect(bound.sizes).toEqual({ row: 5, feature: 2 })
    expect((bound.data['X'] as Tensor).shape).toEqual([5, 2])
  })

  it('names a missing data node', () => {
    expect(() => regression.bind({ X })).toThrow(/no data for 'y'/)
  })

  it('names an unknown data node and lists the known ones', () => {
    expect(() => regression.bind({ X, y: YB, z: [1] })).toThrow(
      /'z' is not a data node of regression \(its data nodes are X, y\)/,
    )
  })

  it('catches a size that disagrees between nodes, naming both', () => {
    expect(() => regression.bind({ X, y: [1, 0, 1] })).toThrow(
      /dimension 'row' has size 5 in data 'X' but 3 in data 'y'/,
    )
  })

  it('catches a rank that does not match the declared dimensions', () => {
    expect(() => regression.bind({ X: [1, 2, 3, 4, 5], y: YB })).toThrow(
      /data 'X' has shape \[5\], but the node has 2 named dimension/,
    )
  })

  it('checks supports and says where', () => {
    expect(() => regression.bind({ X, y: [1, 0, 2, 0, 1] })).toThrow(
      /data 'y'\[2\] is 2: a binary variable must be 0 or 1/,
    )
    const counts = defineModel('counts', (b) => {
      b.observe('k', d.poisson(b.latent('r', d.gamma(2, 1))), b.data('k', { dims: ['n'], support: 'count' }))
    })
    expect(() => counts.bind({ k: [1, 2.5] })).toThrow(/non-negative integer/)
    expect(() => counts.bind({ k: [1, Number.POSITIVE_INFINITY] })).toThrow(/finite/)
  })

  it('checks index codes against the dimension they index', () => {
    const grouped = defineModel('grouped', (b) => {
      const g = b.range('group')
      const idx = b.data('g', { dims: ['row'], support: 'index', indexes: g })
      const y = b.data('y', { dims: ['row'] })
      const mu = b.latent('mu', d.normal(0, 1), { dims: [g] })
      b.observe('y', d.normal(mu.at(idx), 1), y)
    })
    expect(() => grouped.bind({ g: [0, 1, 3], y: [1, 2, 3] }, { sizes: { group: 3 } })).toThrow(/index into 'group'/)
    expect(grouped.bind({ g: [0, 1, 2], y: [1, 2, 3] }, { sizes: { group: 3 } }).sizes).toEqual({ row: 3, group: 3 })
  })

  it('asks for the size of a dimension nothing pins down', () => {
    const m = defineModel('free', (b) => {
      b.latent('z', d.normal(0, 1), { dims: ['J'] })
    })
    expect(() => m.bind({})).toThrow(/size of 'J' is not known/)
    expect(m.bind({}, { sizes: { J: 3 } }).sizes).toEqual({ J: 3 })
    const fixed = defineModel('fixed', (b) => {
      b.latent('z', d.normal(0, 1), { dims: [b.range('J', { size: 2 })] })
    })
    expect(fixed.bind({}).sizes).toEqual({ J: 2 })
  })

  it('refuses NaN outside observed data', () => {
    expect(() => regression.bind({ X: [[1, Number.NaN]], y: [1] })).toThrow(
      /only the data of observed nodes may be missing/,
    )
  })

  it('refuses a missing value when the policy is error', () => {
    expect(() => coin.bind({ y: [1, Number.NaN] }, { missing: { y: 'error' } })).toThrow(/policy 'error'/)
  })

  it('rebinds covariates and unbinds a response for prediction', () => {
    const pred = defineModel('pred', (b) => {
      const n = b.range('row')
      const f = b.range('feature')
      const x = b.data('X', { dims: [n, f], role: 'covariate' })
      const y = b.data('y', { dims: [n], support: 'binary', role: 'response', optional: true })
      const w = b.latent('w', d.normal(0, 1), { dims: [f] })
      b.observe('y', d.bernoulli({ logits: x.matmul(w) }), y)
      b.generated('yRep', d.bernoulli({ logits: x.matmul(w) }))
    })
    const bound = pred.bind({ X, y: YB })
    expect(() => bound.withData({ X: [[1, 2]] })).toThrow(/dimension 'row' has size 1 in data 'X' but 5 in data 'y'/)
    const next = bound.withData({
      X: [
        [1, 2],
        [0, 1],
      ],
      y: null,
    })
    expect(next.sizes['row']).toBe(2)
    expect(next.unbound).toEqual(['y'])
    const t = traceOf(seed(toProgram(next, { generated: true }), key(1)), undefined)
    const sites = [...t.sites].filter(([, x]) => x.type !== 'plate')
    // The observation is left out (no response), the generated draw is there for the two new rows.
    expect(sites.map(([n]) => n)).toEqual(['w', 'yRep'])
    expect(sites[1]?.[1].type === 'sample' && sites[1][1].value.shape).toEqual([2])
  })

  it('binds a library dataset by field names', async () => {
    const { bindDataset } = await import('./index.ts')
    const ds = { x: tensor(X), y: tensor(YB) }
    expect(bindDataset(regression, ds, { X: 'x', y: 'y' }).sizes).toEqual({ row: 5, feature: 2 })
    expect(() => bindDataset(regression, ds, { X: 'x', y: 'nope' })).toThrow(/no field 'nope'/)
  })
})

describe('constraints', () => {
  const log = (x: number, m: number, s: number): number =>
    -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI)

  it('lower: 0 maps through exp with its Jacobian (the existing bijector), the density unchanged', () => {
    const m = defineModel('positive', (b) => {
      b.latent('s', d.normal(0, 1), { constraint: { lower: 0 } })
    })
    const pe = potentialEnergy(toProgram(m.bind({})), undefined)
    for (const u of [-1.2, 0, 0.7]) expect(pe.potential([u])).toBeCloseTo(-(log(Math.exp(u), 0, 1) + u), 12)
  })

  it('lower and upper map through a scaled sigmoid', () => {
    const m = defineModel('interval', (b) => {
      b.latent('p', d.normal(0, 1), { constraint: { lower: -1, upper: 3 } })
    })
    const pe = potentialEnergy(toProgram(m.bind({})), undefined)
    const u = 0.3
    const s = 1 / (1 + Math.exp(-u))
    const x = -1 + 4 * s
    expect(pe.potential([u])).toBeCloseTo(-(log(x, 0, 1) + Math.log(4 * s * (1 - s))), 12)
  })

  it('ordered gives sorted vectors from any unconstrained point', () => {
    const m = defineModel('ordered', (b) => {
      b.latent('c', d.normal(0, 3), { dims: [b.range('cut', { size: 4 })], constraint: { kind: 'ordered' } })
    })
    const pe = potentialEnergy(toProgram(m.bind({})), undefined)
    const c = toFlat(pe.constrain([0.5, -1, 2, 0.1])['c'] as Tensor)
    expect([...c]).toEqual([...c].sort((a, b) => a - b))
    expect(bijectors.bijectorForSupport({ kind: 'ordered' })).toBeDefined()
  })

  it('a latent takes the support of its distribution without a constraint (a simplex draw sums to one)', () => {
    const m = defineModel('simplex', (b) => {
      b.latent('w', d.dirichlet([1, 2, 3]))
    })
    const draws = pEnergyConstrain(m)
    expect(toFlat(draws).reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
  })
})

function pEnergyConstrain(m: ModelDefinition): Tensor {
  const pe = potentialEnergy(toProgram(m.bind({})), undefined)
  return pe.constrain(new Array(pe.dim).fill(0.3))['w'] as Tensor
}

describe('missing values', () => {
  it('ignore masks the entry out of the likelihood', () => {
    const m = defineModel('ignore', (b) => {
      const y = b.data('y', { dims: ['row'] })
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), y, { missing: 'ignore' })
    })
    const a = item(logJoint(toProgram(m.bind({ y: [1, Number.NaN, 3] })), undefined, { mu: 0.2 }).logJoint)
    const b = item(logJoint(toProgram(m.bind({ y: [1, 3] })), undefined, { mu: 0.2 }).logJoint)
    expect(a).toBeCloseTo(b, 13)
  })

  it('impute adds a latent site over the missing entries, whose prior is the likelihood of the entry', () => {
    const m = defineModel('impute', (b) => {
      const y = b.data('y', { dims: ['row'] })
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), y)
    })
    const program = toProgram(m.bind({ y: [1, Number.NaN, 3] }))
    const t = traceOf(seed(program, key(0)), undefined)
    const imputed = t.sites.get('y_missing')
    expect(imputed?.type === 'sample' && imputed.isObserved).toBe(false)
    const full = toProgram(m.bind({ y: [1, 2.5, 3] }))
    const a = item(logJoint(program, undefined, { mu: 0.2, y_missing: tensor([2.5]) }).logJoint)
    const b = item(logJoint(full, undefined, { mu: 0.2 }).logJoint)
    expect(a).toBeCloseTo(b, 13)
    expect(potentialEnergy(program, undefined).dim).toBe(2)
  })
})

describe('batches', () => {
  it('rescales the plate and uses one subset for every node over the dimension', () => {
    const y = [0.2, -0.4, 1.1, 0.7, -1.3, 0.5, 2.0, -0.1, 0.9, 0.3]
    const m = defineModel('batched', (b) => {
      const yy = b.data('y', { dims: ['data'] })
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), yy)
    })
    const program = toProgram(m.bind({ y }, { batch: { dim: 'data', size: 3 } }))
    const seeded = seed(program, key(4))
    const t = traceOf(seeded, undefined)
    const plateSite = t.sites.get('data')
    const idx = plateSite?.type === 'plate' ? toFlat(plateSite.value) : []
    expect(idx).toHaveLength(3)
    const lp = (x: number, mu: number): number => -0.5 * (x - mu) ** 2 - 0.5 * Math.log(2 * Math.PI)
    const mu = 0.3
    const expected = lp(mu, 0) + (10 / 3) * idx.reduce((a, i) => a + lp(y[i] as number, mu), 0)
    expect(item(logJoint(seeded, undefined, { mu }).logJoint)).toBeCloseTo(expected, 12)
  })

  it('refuses a batch dimension the model does not have, and a size beyond the dimension', () => {
    expect(() => coin.bind({ y: [1, 0] }, { batch: { dim: 'nope', size: 1 } })).toThrow(/not a dimension/)
    expect(() => coin.bind({ y: [1, 0] }, { batch: { dim: 'flip', size: 5 } })).toThrow(/batch: the size/)
  })
})

describe('structure', () => {
  const hier = defineModel('hier', (b) => {
    const g = b.range('group')
    const n = b.range('row')
    const idx = b.data('g', { dims: [n], support: 'index', indexes: g })
    const y = b.data('y', { dims: [n], role: 'response' })
    const mu = b.latent('mu', d.normal(0, 1))
    const a = b.latent('a', d.normal(mu, 1), { dims: [g] })
    const eta = b.derived('eta', a.at(idx))
    b.observe('y', d.normal(eta, 1), y)
  })
  const bound = hier.bind({ g: [0, 1, 1], y: [1, 2, 3] }, { sizes: { group: 2 } })

  it('draws latents, derived and observed nodes, the index data, and the dimensions as plates with sizes', () => {
    const s = toStructure(bound)
    expect(s.type).toBe('graphical-model')
    expect(s.nodes.map((x) => [x.id, x.role, x.group ?? ''])).toEqual([
      ['data:g', 'constant', 'plate:row'],
      ['mu', 'latent', ''],
      ['a', 'latent', 'plate:group'],
      ['eta', 'deterministic', 'plate:row'],
      ['y', 'observed', 'plate:row'],
    ])
    expect(s.edges.map((e) => `${e.from}>${e.to}`).sort()).toEqual(['a>eta', 'data:g>eta', 'eta>y', 'mu>a'])
    expect(s.groups.map((g) => [g.id, g.count])).toEqual([
      ['plate:row', 3],
      ['plate:group', 2],
    ])
  })

  it('writes the distribution or expression beside each node', () => {
    const s = toStructure(bound)
    expect(s.nodes.map((x) => x.detail)).toEqual([
      'index',
      '~ Normal(0, 1)',
      '~ Normal(mu, 1)',
      '= a[g]',
      '~ Normal(eta, 1)',
    ])
    expect(toStructure(bound, { details: false }).nodes.every((x) => x.detail === undefined)).toBe(true)
  })

  it('can show every data node, or none', () => {
    expect(toStructure(bound, { data: 'all' }).nodes.map((x) => x.id)).toContain('data:y')
    expect(toStructure(bound, { data: 'none' }).nodes.some((x) => x.id.startsWith('data:'))).toBe(false)
  })

  it('uses the ids of the sites, so a posterior or a tape names the node it belongs to', () => {
    const sites = [...traceOf(seed(toProgram(bound), key(0)), undefined).sites]
      .filter(([, x]) => x.type !== 'plate')
      .map(([n]) => n)
    const ids = toStructure(bound, { data: 'none' }).nodes.map((x) => x.id)
    expect(ids.sort()).toEqual(sites.sort())
    expect(
      toStructure(bound).nodes.every(
        (x) => (x.data as { site: string | null }).site === (x.id.startsWith('data:') ? null : x.id),
      ),
    ).toBe(true)
  })

  it('nests the plates of a node over two dimensions in the spec dimension order, whatever order it declares', () => {
    const m = defineModel('grid', (b) => {
      b.range('a')
      b.range('b')
      b.latent('x', d.normal(0, 1), { dims: ['b', 'a'] })
      b.latent('z', d.normal(0, 1), { dims: ['b'] })
    })
    const s = toStructure(m)
    expect(s.groups.map((g) => [g.id, g.parent ?? '', g.label])).toEqual([
      ['plate:a', '', 'a'],
      ['plate:a×b', 'plate:a', 'b'],
      ['plate:b', '', 'b'],
    ])
    expect(s.nodes.map((x) => [x.id, x.group])).toEqual([
      ['x', 'plate:a×b'],
      ['z', 'plate:b'],
    ])
  })

  it('draws the factor graph with a factor:<site> square per distribution, operator or term', () => {
    const s = toStructure(bound, { view: 'factor' })
    expect(s.type).toBe('factor-graph')
    expect(s.nodes.filter((x) => x.role === 'factor').map((x) => x.id)).toEqual([
      'factor:mu',
      'factor:a',
      'factor:eta',
      'factor:y',
    ])
    expect(s.edges.map((e) => `${e.from}>${e.to}`).sort()).toEqual(
      [
        'a>factor:eta',
        'data:g>factor:eta',
        'eta>factor:y',
        'factor:a>a',
        'factor:eta>eta',
        'factor:mu>mu',
        'factor:y>y',
        'mu>factor:a',
      ].sort(),
    )
    const withTerm = defineModel('term', (b) => b.factor('penalty', b.latent('p', d.beta(2, 2)).square()))
    expect(toStructure(withTerm, { view: 'factor' }).nodes.map((x) => x.id)).toEqual([
      'p',
      'factor:p',
      'factor:penalty',
    ])
  })

  it('typesets Greek names and subscripts as TeX on request', () => {
    expect(texName('tau')).toBe('$\\tau$')
    expect(texName('theta_j')).toBe('$\\theta_{j}$')
    expect(texName('beta1')).toBe('$\\beta_{1}$')
    expect(texName('y')).toBe('y')
    const m = defineModel('t', (b) => void b.latent('sigma', d.halfNormal(1)))
    expect(toStructure(m, { tex: true }).nodes[0]?.label).toBe('$\\sigma$')
  })

  it('writes expressions as text', () => {
    const e = ops.add(ops.mul(2, 3), ops.exp(1))
    expect(exprText(e.expr)).toBe('(2 * 3) + exp(1)')
    expect(exprText(ops.sum(constant([1, 2], ['i']), 'i').expr)).toBe('sum_i([1,2])')
  })

  it('validates what it builds', () => {
    expect(() => validateSpec(coin.spec)).not.toThrow()
  })
})

describe('the id convention', () => {
  it('names nodes, factors, plates and elements', () => {
    expect([nodeId({ kind: 'latent', name: 'mu' }), nodeId({ kind: 'data', name: 'y' })]).toEqual(['mu', 'data:y'])
    expect(factorId('y')).toBe('factor:y')
    expect(plateId(['row', 'feature'])).toBe('plate:row×feature')
    expect(plateId([])).toBeUndefined()
    expect(elementId('w', 1)).toBe('w[1]')
  })

  it('analysis and structure agree on ids, plates and factors', () => {
    const m = defineModel('p', (b) => {
      b.range('row')
      b.range('feature')
      b.data('X', { dims: ['feature', 'row'] })
      b.latent('w', d.normal(0, 1), { dims: ['feature'] })
    })
    const a = analyseSpec(m)
    expect(a.nodes.map((n) => [n.id, n.plate, n.factor])).toEqual([
      ['data:X', 'plate:row×feature', undefined],
      ['w', 'plate:feature', 'factor:w'],
    ])
    expect(a.plates.map((p) => [p.id, p.parent, p.nodes])).toEqual([
      ['plate:row', undefined, []],
      ['plate:feature', undefined, ['w']],
      ['plate:row×feature', 'plate:row', ['data:X']],
    ])
  })

  it('shows the model data of a node over named dimensions in the structure data field', () => {
    const m = defineModel('p', (b) => void b.latent('w', d.normal(0, 1), { dims: ['feature'] }))
    expect(toStructure(m).nodes[0]?.data).toEqual({
      spec: 'latent',
      site: 'w',
      dims: ['feature'],
      plate: 'plate:feature',
    })
  })
})

describe('include, condition and do', () => {
  const sub = defineModel('line', (b) => {
    const x = b.data('x', { dims: ['obs'] })
    const slope = b.latent('slope', d.normal(0, 1))
    const mean = b.derived('mean', slope.mul(x))
    b.observe('y', d.normal(mean, 1), b.data('y', { dims: ['obs'] }))
  })

  it('instantiates a sub-model under a prefix, with a data node bound to an outer expression', () => {
    const outer = defineModel('two lines', (b) => {
      const n = b.range('row')
      const x = b.data('x', { dims: [n] })
      b.include(sub, { prefix: 'a', bind: { x } })
      b.include(sub, { prefix: 'b', bind: { x: x.mul(2) } })
    })
    const names = outer.spec.nodes.map((n) => n.name)
    expect(names).toEqual(['x', 'a/slope', 'a/mean', 'a/y', 'a/y', 'b/slope', 'b/mean', 'b/y', 'b/y'])
    // The bound data node's dimension stands for the sub-model's, so both instances share 'row'.
    expect(outer.spec.dims.map((x) => x.name)).toEqual(['row'])
    const bound = outer.bind({ x: [1, 2, 3], 'a/y': [1, 2, 3], 'b/y': [2, 4, 6] })
    const t = traceOf(seed(toProgram(bound), key(0)), undefined)
    expect([...t.sites.keys()].filter((k) => !k.startsWith('row'))).toEqual([
      'a/slope',
      'a/mean',
      'a/y',
      'b/slope',
      'b/mean',
      'b/y',
    ])
  })

  it('conditions a latent on a value: it becomes an observed node and later nodes read the value', () => {
    const m = defineModel('chain', (b) => {
      const mu = b.latent('mu', d.normal(0, 1))
      b.observe('y', d.normal(mu, 1), 0.5)
    })
    const c = m.condition({ mu: 0.25 })
    expect(c.spec.nodes.map((n) => n.kind)).toEqual(['observed', 'observed'])
    const lj = item(logJoint(toProgram(c.bind({})), undefined, {}).logJoint)
    const lp = (x: number, mu: number): number => -0.5 * (x - mu) ** 2 - 0.5 * Math.log(2 * Math.PI)
    expect(lj).toBeCloseTo(lp(0.25, 0) + lp(0.5, 0.25), 13)
    expect(() => m.condition({ nope: 1 })).toThrow(/no site named 'nope'/)
    expect(() => m.condition({ mu: [1, 2] })).toThrow(/axes/)
  })

  it('intervenes on a latent: it becomes a recorded constant that adds nothing to the density', () => {
    const m = defineModel('chain', (b) => {
      const mu = b.latent('mu', d.normal(0, 1))
      b.observe('y', d.normal(mu, 1), 0.5)
    })
    const i = m.do({ mu: 0.25 })
    expect(i.spec.nodes.map((n) => n.kind)).toEqual(['derived', 'observed'])
    const lj = item(logJoint(toProgram(i.bind({})), undefined, {}).logJoint)
    expect(lj).toBeCloseTo(-0.5 * (0.5 - 0.25) ** 2 - 0.5 * Math.log(2 * Math.PI), 13)
    const t = traceOf(seed(toProgram(i.bind({})), key(0)), undefined)
    expect(t.sites.get('mu')?.type).toBe('deterministic')
  })

  it('leaves the original definition alone', () => {
    const before = JSON.stringify(sub)
    sub.condition({ slope: 1 })
    sub.do({ slope: 1 })
    expect(JSON.stringify(sub)).toBe(before)
  })
})

describe('tensors as operands', () => {
  it('a tensor is a constant exactly like the nested array of its values (a vector, a matrix, a scalar)', () => {
    expect(constant(zeros([3])).expr).toEqual(constant([0, 0, 0]).expr)
    expect(constant(eye(2)).expr).toEqual(
      constant([
        [1, 0],
        [0, 1],
      ]).expr,
    )
    expect(constant(full([], 2)).expr).toEqual(constant(2).expr)
  })

  it('a distribution takes tensor parameters: the same model as with arrays', () => {
    const build = (mean: Tensor, covariance: Tensor): ModelDefinition =>
      defineModel('tensor-operands', (m) => {
        const cell = m.range('cell', { size: 3 })
        m.latent('f', d.multivariateNormal(mean, { covariance }), { dims: [cell] })
      })
    const viaTensors = build(zeros([3]), eye(3))
    const viaArrays = defineModel('tensor-operands', (m) => {
      const cell = m.range('cell', { size: 3 })
      m.latent(
        'f',
        d.multivariateNormal([0, 0, 0], {
          covariance: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
        }),
        { dims: [cell] },
      )
    })
    expect(JSON.stringify(viaTensors)).toBe(JSON.stringify(viaArrays))
  })
})
