// The latent variable model with random Fourier features compiled as one `gplvm-row` per point (with `packScalars`): the graph it makes, the
// variables the posterior reads back, and the cases outside the factor's limits that compile by the scalar operators as before (three
// coordinates per point, too many weights, a missing observation, a noise that differs between entries, a weight matrix that something
// else reads, another form of the mean). `epRoute` takes the model either way.
import { describe, expect, it } from 'vitest'
import { GPLVM_MOST_WEIGHTS } from '../../../message-passing/index.ts'
import { key } from '../../../random/index.ts'
import { tensor, toArray, toFlat } from '../../../tensor/index.ts'
import { session } from '../../../trace/index.ts'
import { epRoute } from '../../inference/index.ts'
import { fourierFeatures, gplvmFourierModel, simulateGplvm } from '../../models/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, toFactorGraph } from '../index.ts'

const operatorsOf = (c: ReturnType<typeof toFactorGraph>): Record<string, number> => {
  const counts: Record<string, number> = {}
  for (const f of c.graph?.factors ?? []) counts[f.operator] = (counts[f.operator] ?? 0) + 1
  return counts
}

function bindOf(options: { features: number; coords: number; outputs: number }, points: number, seed = 1) {
  const { frequencies, phases } = fourierFeatures(key(seed), options)
  const data = simulateGplvm(key(seed + 100), { points, coords: options.coords, outputs: options.outputs })
  return {
    bound: gplvmFourierModel(options).bind({ omega: frequencies, phase: phases, y: data.y }),
    omega: frequencies,
    phase: phases,
    y: data.y,
  }
}

describe('gplvm-rows: the form of gplvmFourierModel compiles to one gplvm-row per point', () => {
  it('makes a factor per point on its coordinates and the weight vector, and no product, apply or sum', () => {
    const { bound, omega, phase, y } = bindOf({ features: 3, coords: 1, outputs: 2 }, 4)
    const c = toFactorGraph(bound, { packScalars: true })
    expect(c.issues).toEqual([])
    const counts = operatorsOf(c)
    expect(counts['gplvm-row']).toBe(4)
    expect(counts['product']).toBeUndefined()
    expect(counts['apply']).toBeUndefined()
    expect(counts['sum']).toBeUndefined()
    const rows = (c.graph?.factors ?? []).filter((f) => f.operator === 'gplvm-row')
    expect(rows[1]?.args).toEqual(['x[1]', 'W#8[0]', 'W#8[1]'])
    const params = rows[1]?.params as Record<string, unknown>
    expect(params['coords']).toBe(1)
    expect(params['features']).toBe(3)
    expect(params['outputs']).toBe(2)
    expect(params['omega']).toEqual(toFlat(omega))
    expect(params['phase']).toEqual(toFlat(phase))
    expect(params['noise']).toBe(0.1)
    expect(params['y']).toEqual((toArray(y) as number[][])[1])
    // The weights of each output are one Gaussian vector of the features, and the posterior reads each of them from its own variable.
    for (const vector of ['W#8[0]', 'W#8[1]'])
      expect(c.graph?.variables.find((v) => v.name === vector)?.family).toEqual({ kind: 'gaussian-mv', size: 3 })
    expect(c.graph?.variables.filter((v) => /^W\[/.test(v.name))).toHaveLength(6)
    expect(c.sites.find((s) => s.site === 'W')?.shape).toEqual([2, 3])
    expect(c.sites.find((s) => s.site === 'x')?.shape).toEqual([4, 1])
  })

  it('is the scalar route without packScalars: the same model, the old operators', () => {
    const { bound } = bindOf({ features: 3, coords: 1, outputs: 2 }, 4)
    const counts = operatorsOf(toFactorGraph(bound))
    expect(counts['gplvm-row']).toBeUndefined()
    expect(counts['product']).toBe(4 * 2 * 3)
    expect(counts['apply']).toBeGreaterThan(0)
    expect(counts['sum']).toBe(4 * 2)
  })

  it('is taken by epRoute, with the verdict applying', () => {
    const { bound } = bindOf({ features: 3, coords: 1, outputs: 2 }, 4)
    const route = epRoute(bound)
    expect(route.via).toBe('ep')
    expect(route.verdict.applies).toBe(true)
  })
})

/** A model written like `gplvmFourierModel`, with the pieces a test changes. */
function variant(options: {
  readonly points?: number
  readonly outputs?: number
  readonly features?: number
  readonly coords?: number
  readonly swapped?: boolean
  readonly phased?: boolean
  readonly notCosine?: boolean
  readonly noise?: 'one' | 'per-output'
  readonly weightsAlsoRead?: boolean
  readonly missing?: boolean
  readonly derived?: boolean
}) {
  const { points = 4, outputs = 2, features = 3, coords = 1 } = options
  const definition = defineModel('gplvm-variant', (m) => {
    const point = m.range('point')
    const feature = m.range('feature', { size: features })
    const output = m.range('output', { size: outputs })
    const coord = m.range('coord', { size: coords })
    const omega = m.data('omega', { dims: [coord, feature] })
    const phase = m.data('phase', { dims: [feature] })
    const y = m.data('y', { dims: [point, output], role: 'response' })
    const x = m.latent('x', d.normal(0, 1), { dims: [point, coord] })
    const w = m.latent('W', d.normal(0, 0.8), { dims: [output, feature] })
    const angle = options.phased === false ? x.matmul(omega) : x.matmul(omega).add(phase)
    const raw = options.notCosine === true ? angle.sigmoid() : angle.cos()
    const phi = options.derived === true ? m.derived('phi', raw) : raw
    const f = options.swapped === true ? w.mul(phi).sum(feature) : phi.mul(w).sum(feature)
    if (options.weightsAlsoRead === true) m.observe('z', d.normal(w.sum(feature), 1), m.data('z', { dims: [output] }))
    const noise = options.noise === 'per-output' ? m.data('noise', { dims: [output] }) : 0.1
    m.observe('y', d.normal(f, noise), y)
  })
  const { frequencies, phases } = fourierFeatures(key(3), { features, coords })
  const y = simulateGplvm(key(4), { points, coords, outputs }).y
  const values =
    options.missing === true
      ? tensor([[Number.NaN, ...toFlat(y).slice(1, outputs)], ...(toArray(y) as number[][]).slice(1)])
      : y
  const data = {
    omega: frequencies,
    phase: phases,
    y: values,
    ...(options.weightsAlsoRead === true ? { z: tensor(Array.from({ length: outputs }, () => 0.3)) } : {}),
    ...(options.noise === 'per-output'
      ? { noise: tensor(Array.from({ length: outputs }, (_, i) => 0.1 + 0.05 * i)) }
      : {}),
  }
  return definition.bind(data, options.missing === true ? { missing: { y: 'ignore' } } : {})
}

describe('gplvm-rows: the cases it does not take compile by the scalar operators, as before', () => {
  const rowsOf = (bound: ReturnType<typeof variant>): Record<string, number> =>
    operatorsOf(toFactorGraph(bound, { packScalars: true }))

  it('takes the product in either order, and the angle with no phase', () => {
    expect(rowsOf(variant({}))['gplvm-row']).toBe(4)
    expect(rowsOf(variant({ swapped: true }))['gplvm-row']).toBe(4)
    // with the weights first the mean's dimensions, and so the observed node's, run output then point: each row still has its own point's measurements
    const swapped = toFactorGraph(variant({ swapped: true }), { packScalars: true })
    const want = toArray(simulateGplvm(key(4), { points: 4, coords: 1, outputs: 2 }).y) as number[][]
    const rows = (swapped.graph?.factors ?? []).filter((f) => f.operator === 'gplvm-row')
    expect(rows.map((f) => (f.params as Record<string, unknown>)['y'])).toEqual(want)
    expect(rowsOf(variant({ phased: false }))['gplvm-row']).toBe(4)
  })

  it('leaves two coordinates per point to the scalar route (the factor takes two, but it is not better there)', () => {
    const counts = rowsOf(variant({ coords: 2 }))
    expect(counts['gplvm-row']).toBeUndefined()
    expect(counts['product']).toBeGreaterThan(0)
  })

  it('leaves more weights than a row takes', () => {
    const counts = rowsOf(variant({ outputs: 30, features: Math.ceil(GPLVM_MOST_WEIGHTS / 30) + 1 }))
    expect(counts['gplvm-row']).toBeUndefined()
    expect(counts['product']).toBeGreaterThan(0)
  })

  it('leaves a function that is not a cosine, a noise that differs between outputs, a weight matrix something else reads, missing values and the cosine as a derived node', () => {
    for (const options of [
      { notCosine: true },
      { noise: 'per-output' as const },
      { weightsAlsoRead: true },
      { missing: true },
      { derived: true },
    ]) {
      const counts = rowsOf(variant(options))
      expect(counts['gplvm-row'], JSON.stringify(options)).toBeUndefined()
    }
  })

  it('compiles every one of them without an issue, and the route stays plain EP where it was', () => {
    for (const options of [
      { coords: 3 },
      { notCosine: true },
      { noise: 'per-output' as const },
      { weightsAlsoRead: true },
    ]) {
      const bound = variant(options)
      expect(toFactorGraph(bound, { packScalars: true }).issues, JSON.stringify(options)).toEqual([])
    }
    // The model the story ran before this factor: the scalar route is still accepted without packing.
    for (const coords of [2, 3]) expect(epRoute(variant({ coords })).via, `${coords} coordinates`).toBe('ep')
  })

  it('posterior reads x and W back in the shapes of the model', () => {
    const { bound } = bindOf({ features: 3, coords: 2, outputs: 2 }, 3)
    const route = epRoute(bound)
    if (route.via !== 'ep') throw new Error(route.verdict.reason)
    const run = session(route.method, { steps: 3, mode: 'eager', config: { damping: 0.5, restarts: 0 } })
    const post = route.method.posterior?.(run.state, run.config)
    expect(post?.mean('x').shape).toEqual([3, 2])
    expect(post?.mean('W').shape).toEqual([2, 3])
    expect(post?.stddev('W').shape).toEqual([2, 3])
  })
})
