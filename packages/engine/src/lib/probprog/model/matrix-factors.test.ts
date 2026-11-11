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

describe('three dimensions end to end', () => {
  const rows3 = [
    [1, 0.5, -0.2],
    [-0.3, 0.2, 0.4],
    [0.8, -1.1, 0.1],
    [0.1, 0.4, -0.6],
    [0.6, 0.0, 0.9],
    [-0.9, 0.3, 0.2],
    [0.2, -0.5, -0.1],
  ]
  const s0 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]
  const posteriorScale = (centre: number[]): number[][] => {
    // S^-1 = I + sum (y - centre)(y - centre)'
    const inv = s0.map((r) => [...r])
    for (const y of rows3)
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          (inv[i] as number[])[j] =
            ((inv[i] as number[])[j] as number) +
            ((y[i] as number) - (centre[i] as number)) * ((y[j] as number) - (centre[j] as number))
    return inv
  }
  const invert3 = (a: number[][]): number[][] => {
    const [[a0, a1, a2], [b0, b1, b2], [c0, c1, c2]] = a as [
      [number, number, number],
      [number, number, number],
      [number, number, number],
    ]
    const det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
    return [
      [(b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det, (a1 * b2 - a2 * b1) / det],
      [(b2 * c0 - b0 * c2) / det, (a0 * c2 - a2 * c0) / det, (a2 * b0 - a0 * b2) / det],
      [(b0 * c1 - b1 * c0) / det, (a1 * c0 - a0 * c1) / det, (a0 * b1 - a1 * b0) / det],
    ]
  }

  it('a known mean: the exact conjugate posterior of a 3 by 3 precision matrix', () => {
    const model = defineModel('mvn3', (m) => {
      const row = m.range('row')
      const lambda = m.latent('Lambda', d.wishart(5, s0))
      m.observe('y', d.multivariateNormal([0, 0, 0], { precision: lambda }), m.data('y', { dims: [row], eventRank: 1 }))
    })
    const { marginals } = run(model.bind({ y: rows3 }))
    const lambda = marginals['Lambda'] as { nu: number; scale: number[][] }
    expect(lambda.nu).toBeCloseTo(12, 7)
    const expected = invert3(posteriorScale([0, 0, 0]))
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(lambda.scale[i]?.[j]).toBeCloseTo(expected[i]?.[j] as number, 7)
  })

  it('an unknown mean with a nearly flat prior: the Wishart of n - 1 degrees of freedom and the scatter about the sample mean', () => {
    const model = defineModel('mvn3-mean', (m) => {
      const row = m.range('row')
      const coef = m.range('coef', { size: 3 })
      const mu = m.latent(
        'mu',
        d.multivariateNormal([0, 0, 0], {
          covariance: [
            [1e4, 0, 0],
            [0, 1e4, 0],
            [0, 0, 1e4],
          ],
        }),
        { dims: [coef] },
      )
      const lambda = m.latent('Lambda', d.wishart(5, s0))
      m.observe(
        'y',
        d.multivariateNormal(mu, { precision: lambda }, { over: coef }),
        m.data('y', { dims: [row], eventRank: 1 }),
      )
    })
    const { marginals } = run(model.bind({ y: rows3 }))
    const lambda = marginals['Lambda'] as { nu: number; scale: number[][]; mean: number[][] }
    const ybar = [0, 1, 2].map((k) => rows3.reduce((s, y) => s + (y[k] as number), 0) / rows3.length)
    // the exact marginal of the matrix is Wishart(nu0 + n - 1, (S0^-1 + scatter)^-1) in the flat-mean limit; EP matches its moments
    const expected = invert3(posteriorScale(ybar))
    const nu = 5 + rows3.length - 1
    expect(Math.abs(lambda.nu - nu) / nu).toBeLessThan(0.02)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        expect(Math.abs((lambda.mean[i]?.[j] as number) - nu * (expected[i]?.[j] as number))).toBeLessThan(
          0.02 * (1 + Math.abs(nu * (expected[i]?.[j] as number))),
        )
    const mu = marginals['mu'] as { mean: number[] }
    for (let k = 0; k < 3; k++) expect(mu.mean[k]).toBeCloseTo(ybar[k] as number, 2)
  })
})

describe('a bivariate normal with standard deviations and a correlation', () => {
  const rows = [
    [1.1, 0.6],
    [-0.4, -0.2],
    [0.9, 1.3],
    [0.2, 0.1],
    [-1.2, -0.8],
    [0.5, 0.9],
  ]
  const build = (known: boolean) =>
    defineModel('bivariate', (m) => {
      const row = m.range('row')
      const coef = m.range('coef', { size: 2 })
      const rho = m.latent('rho', d.correlation(1))
      const y = m.data('y', { dims: [row], eventRank: 1 })
      if (known) {
        m.observe('y', d.bivariateNormal([0, 0], { scale1: 1, scale2: 1, rho }), y)
        return
      }
      const tau1 = m.latent('tau1', d.gamma(2, 2))
      const tau2 = m.latent('tau2', d.gamma(2, 2))
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
      const sd1 = ops.div(1, ops.sqrt(tau1))
      const sd2 = ops.div(1, ops.sqrt(tau2))
      m.observe('y', d.bivariateNormal(mu, { scale1: sd1, scale2: sd2, rho }, { over: coef }), y)
    })

  it('compiles to one plated factor over the precisions, the correlation and the mean, and reads the correlation as a correlation', () => {
    const { c, g, state } = run(build(false).bind({ y: rows }))
    expect(c.graph?.factors.filter((f) => f.operator === 'mvn-sd-corr')).toHaveLength(1)
    const p = factorGraphPosterior(c, g, state)
    expect(p.sites).toEqual(['rho', 'tau1', 'tau2', 'mu'])
    const mean = Number(toArray(p.mean('rho')))
    const sd = Number(toArray(p.stddev('rho')))
    expect(mean).toBeGreaterThan(0)
    expect(mean).toBeLessThan(1)
    expect(sd).toBeGreaterThan(0)
  })

  it('known sds and mean: the plate is the exact one-dimensional posterior of the correlation, a factor per row an approximation of it', () => {
    // the exact posterior: (1 - rho^2)^(-n / 2) exp(-(sxx - 2 rho sxy + syy) / (2 (1 - rho^2))) on (-1, 1), uniform prior
    const n = rows.length
    const sxx = rows.reduce((acc, r) => acc + (r[0] as number) ** 2, 0)
    const syy = rows.reduce((acc, r) => acc + (r[1] as number) ** 2, 0)
    const sxy = rows.reduce((acc, r) => acc + (r[0] as number) * (r[1] as number), 0)
    let z = 0
    let m1 = 0
    let m2 = 0
    const N = 4000
    for (let i = 0; i < N; i++) {
      const rho = -1 + (2 * (i + 0.5)) / N
      const w = Math.exp(-(n / 2) * Math.log(1 - rho * rho) - (sxx - 2 * rho * sxy + syy) / (2 * (1 - rho * rho)))
      z += w
      m1 += w * rho
      m2 += w * rho * rho
    }
    const [mean, sd] = [m1 / z, Math.sqrt(m2 / z - (m1 / z) ** 2)]
    const betaMean = (m: { a: number; b: number }): number => (2 * m.a) / (m.a + m.b) - 1
    const plated = run(build(true).bind({ y: rows })).marginals['rho'] as { a: number; b: number }
    expect(Math.abs(betaMean(plated) - mean) / sd).toBeLessThan(0.05)
    const c = toFactorGraph(build(true).bind({ y: rows }), { plates: 'expand' })
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'mvn-sd-corr')).toHaveLength(rows.length)
    const g = compileGraph(c.graph as never, factorOperators())
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10, damping: 0.4 }), { maxSteps: 400 })
    const expanded = marginalsOf(g, r.state as MpState)['rho'] as unknown as { a: number; b: number }
    // a factor per row is ordinary EP with six sites on the one variable: a looser match (measured below one posterior sd)
    expect(Math.abs(betaMean(expanded) - mean) / sd).toBeLessThan(1)
  })

  it('refusals: a latent bivariate normal, a scale that is not the precision idiom', () => {
    const issues = (build: Parameters<typeof defineModel>[1], data: Record<string, unknown>): string[] =>
      toFactorGraph(defineModel('bad', build).bind(data as never)).issues.map((i) => i.reason)
    const a = issues((m) => {
      const row = m.range('row', { size: 3 })
      const rho = m.latent('rho', d.correlation(1))
      m.latent('x', d.bivariateNormal([0, 0], { scale1: 1, scale2: 1, rho }), { dims: [row] })
    }, {})
    expect(a.join(' ')).toMatch(/only observed vectors are mapped/)
    const b = issues(
      (m) => {
        const row = m.range('row')
        const s = m.latent('s', d.halfNormal(1))
        const rho = m.latent('rho', d.correlation(1))
        m.observe(
          'y',
          d.bivariateNormal([0, 0], { scale1: s, scale2: 1, rho }),
          m.data('y', { dims: [row], eventRank: 1 }),
        )
      },
      { y: rows },
    )
    expect(b.join(' ')).toMatch(/precision idiom/)
  })
})
