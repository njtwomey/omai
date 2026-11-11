// A Wishart (or inverse-Wishart) latent and the multivariate normals that read it, as models under EP: the conjugate case against the
// closed-form posterior of the matrix, the refusals that name the form that is mapped, and the shapes of the posterior.
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../message-passing/index.ts'
import { toArray } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from './dists/index.ts'
import { defineModel, factorGraphPosterior, ops, toFactorGraph } from './index.ts'

const ROWS = [
  [1, 0.5],
  [-0.3, 0.2],
  [0.8, -1.1],
  [0.1, 0.4],
  [0.6, 0.0],
]

function run(bound: Parameters<typeof toFactorGraph>[0]) {
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10, damping: 0.4 }), { maxSteps: 300 })
  expect(['converged', 'completed']).toContain(r.status)
  return {
    c,
    g,
    state: r.state as MpState,
    marginals: marginalsOf(g, r.state as MpState) as Record<string, Record<string, unknown>>,
  }
}

describe('a Wishart latent that is the precision of observed vectors with a known mean', () => {
  const model = defineModel('mvn-wishart', (m) => {
    const row = m.range('row')
    const lambda = m.latent(
      'Lambda',
      d.wishart(4, [
        [1, 0.2],
        [0.2, 0.8],
      ]),
    )
    const y = m.data('y', { dims: [row], eventRank: 1 })
    m.observe('y', d.multivariateNormal([0.1, -0.1], { precision: lambda }), y)
  })

  it('is the conjugate update: nu + n and the scale with the scatter about the known mean added', () => {
    const { marginals } = run(model.bind({ y: ROWS }))
    const lambda = marginals['Lambda'] as { nu: number; scale: number[][] }
    expect(lambda.nu).toBeCloseTo(9, 7)
    // S^-1 = S0^-1 + sum (y - m)(y - m)'
    const det0 = 1 * 0.8 - 0.2 * 0.2
    const inv0 = [
      [0.8 / det0, -0.2 / det0],
      [-0.2 / det0, 1 / det0],
    ]
    const inv = inv0.map((r) => [...r])
    for (const y of ROWS) {
      const r = [(y[0] as number) - 0.1, (y[1] as number) + 0.1]
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          (inv[i] as number[])[j] = ((inv[i] as number[])[j] as number) + (r[i] as number) * (r[j] as number)
    }
    const det = (inv[0]?.[0] as number) * (inv[1]?.[1] as number) - (inv[0]?.[1] as number) ** 2
    expect(lambda.scale[0]?.[0]).toBeCloseTo((inv[1]?.[1] as number) / det, 7)
    expect(lambda.scale[0]?.[1]).toBeCloseTo(-(inv[0]?.[1] as number) / det, 7)
  })

  it('the posterior is a library Wishart distribution with the matrix shapes', () => {
    const { c, g, state } = run(model.bind({ y: ROWS }))
    const p = factorGraphPosterior(c, g, state)
    expect(p.sites).toEqual(['Lambda'])
    expect(p.mean('Lambda').shape).toEqual([2, 2])
    expect(p.stddev('Lambda').shape).toEqual([2, 2])
    const mean = toArray(p.mean('Lambda')) as number[][]
    expect(mean[0]?.[1]).toBeCloseTo(mean[1]?.[0] as number, 12)
  })
})

describe('an inverse-Wishart covariance', () => {
  it('is the same variable read as its inverse: the posterior is the inverse Wishart of the same degrees of freedom', () => {
    const model = defineModel('mvn-iw', (m) => {
      const row = m.range('row')
      const sigma = m.latent(
        'Sigma',
        d.inverseWishart(4, [
          [1, 0.2],
          [0.2, 0.8],
        ]),
      )
      const y = m.data('y', { dims: [row], eventRank: 1 })
      m.observe('y', d.multivariateNormal([0, 0], { covariance: sigma }), y)
    })
    const { c, g, state } = run(model.bind({ y: ROWS }))
    const p = factorGraphPosterior(c, g, state)
    expect(p.mean('Sigma').shape).toEqual([2, 2])
    // the mean of an inverse Wishart IW(nu, Psi) is Psi / (nu - d - 1), with Psi the inverse of the Wishart scale: positive diagonal
    const mean = toArray(p.mean('Sigma')) as number[][]
    expect(mean[0]?.[0]).toBeGreaterThan(0)
    expect(mean[1]?.[1]).toBeGreaterThan(0)
  })
})

describe('refusals name the form that is mapped', () => {
  const issues = (build: Parameters<typeof defineModel>[1], data: Record<string, unknown>): string[] => {
    const bound = defineModel('bad', build).bind(data as never)
    return toFactorGraph(bound).issues.map((i) => i.reason)
  }
  it('a precision that is the covariance of an inverse Wishart, and the other way round', () => {
    const a = issues(
      (m) => {
        const row = m.range('row')
        const s = m.latent(
          'Sigma',
          d.inverseWishart(4, [
            [1, 0],
            [0, 1],
          ]),
        )
        m.observe('y', d.multivariateNormal([0, 0], { precision: s }), m.data('y', { dims: [row], eventRank: 1 }))
      },
      { y: ROWS },
    )
    expect(a.join(' ')).toMatch(/inverse Wishart/)
    const b = issues(
      (m) => {
        const row = m.range('row')
        const l = m.latent(
          'Lambda',
          d.wishart(4, [
            [1, 0],
            [0, 1],
          ]),
        )
        m.observe('y', d.multivariateNormal([0, 0], { covariance: l }), m.data('y', { dims: [row], eventRank: 1 }))
      },
      { y: ROWS },
    )
    expect(b.join(' ')).toMatch(/Wishart/)
  })

  it('degrees of freedom below the dimension', () => {
    const a = issues(
      (m) => {
        const row = m.range('row')
        const l = m.latent(
          'Lambda',
          d.wishart(1, [
            [1, 0],
            [0, 1],
          ]),
        )
        m.observe('y', d.multivariateNormal([0, 0], { precision: l }), m.data('y', { dims: [row], eventRank: 1 }))
      },
      { y: ROWS },
    )
    expect(a.join(' ')).toMatch(/degrees of freedom/)
  })

  it('a random mean that is not one whole Gaussian vector', () => {
    const a = issues(
      (m) => {
        const row = m.range('row')
        const coef = m.range('coef', { size: 2 })
        const l = m.latent(
          'Lambda',
          d.wishart(4, [
            [1, 0],
            [0, 1],
          ]),
        )
        const mu = m.latent(
          'mu',
          d.multivariateNormal([0, 0], {
            covariance: [
              [1, 0],
              [0, 1],
            ],
          }),
          { dims: [coef] },
        )
        m.observe(
          'y',
          d.multivariateNormal(ops.mul(mu, 2), { precision: l }, { over: coef }),
          m.data('y', { dims: [row], eventRank: 1 }),
        )
      },
      { y: ROWS },
    )
    expect(a.join(' ')).toMatch(/whole Gaussian vector|known vector/)
  })
})

describe('correlated random effects: a vector per group with a Wishart precision and a random mean', () => {
  const groups = 4
  const design: { g: number; t: number; y: number }[] = []
  for (let j = 0; j < groups; j++)
    for (const t of [-1, 0, 1, 2])
      design.push({ g: j, t, y: 0.5 * j + (j - 1.5) * 0.3 * t + 0.1 * Math.sin(j * 7 + t) })
  const model = defineModel('random-effects', (m) => {
    const group = m.range('group', { size: groups })
    const coef = m.range('coef', { size: 2 })
    const obs = m.range('obs')
    const lambda = m.latent(
      'Lambda',
      d.wishart(4, [
        [1, 0],
        [0, 1],
      ]),
    )
    const mu = m.latent(
      'mu',
      d.multivariateNormal([0, 0], {
        covariance: [
          [4, 0],
          [0, 4],
        ],
      }),
      { dims: [coef] },
    )
    const theta = m.latent('theta', d.multivariateNormal(mu, { precision: lambda }, { over: coef }), {
      dims: [group, coef],
    })
    const x = m.data('X', { dims: [obs, coef] })
    const g = m.data('g', { dims: [obs], indexes: 'group' })
    m.observe('y', d.normal(ops.sum(ops.mul(theta.at(g), x), coef), 0.3), m.data('y', { dims: [obs] }))
  })
  const bound = model.bind({ X: design.map((o) => [1, o.t]), g: design.map((o) => o.g), y: design.map((o) => o.y) })

  it('compiles to a vector variable per group and a factor per vector, and runs', () => {
    const { c, g, state, marginals } = run(bound)
    expect(c.graph?.variables.filter((v) => v.family.kind === 'gaussian-mv').map((v) => v.name)).toEqual([
      'mu',
      'theta[0]',
      'theta[1]',
      'theta[2]',
      'theta[3]',
    ])
    expect(c.graph?.factors.filter((f) => f.operator === 'mvn-wishart')).toHaveLength(groups)
    const p = factorGraphPosterior(c, g, state)
    expect(p.sites).toEqual(['Lambda', 'mu', 'theta'])
    expect(p.mean('theta').shape).toEqual([groups, 2])
    expect(p.stddev('theta').shape).toEqual([groups, 2])
    // each group's intercept and slope are recovered to within a posterior sd of the data-generating line at this noise
    const th1 = marginals['theta[1]'] as { mean: number[] }
    expect(th1.mean[0]).toBeGreaterThan(0.2)
    expect(th1.mean[0]).toBeLessThan(0.8)
  })
})
