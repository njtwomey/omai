// gaussian-mv-stack: a Gaussian vector assembled from scalar Gaussian variables. Its EP messages are those of the unit `dot-constant` ties
// it replaces, its evidence the integral of the cavities, and a graph of it under EP is exact where the model is linear Gaussian.
import { describe, expect, it } from 'vitest'
import { integrateBox } from '../../../../testing/message-passing.ts'
import { ctx, msg, near } from '../../../../testing/operator-checks.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as M from '../../index.ts'

const vec = M.gaussianMv(3)
const cavity = M.gaussianMvEta([0.4, -0.6, 1.1], [1.5, 0.3, -0.2, 0.3, 1.1, 0.4, -0.2, 0.4, 0.9])
const scalars = [M.gaussianEta(0.2, 0.7), M.gaussianEta(-0.5, 1.3), M.gaussianEta(0.9, 0.4)]
const stackArgs = (): M.Arg[] => [msg(vec, cavity), ...scalars.map((e) => msg(M.gaussian, e))]

describe('gaussian-mv-stack under EP is the unit dot-constants in one factor', () => {
  it('sends the vector the diagonal of the scalars cavities, the sum of what the ties send', () => {
    const m = M.gaussianMvStack.ep?.(ctx(0, stackArgs(), { n: 3 })).message as number[]
    const ties = scalars.map(
      (e, j) =>
        M.dotConstant.ep?.(
          ctx(1, [msg(M.gaussian, e), msg(vec, cavity)], { a: [0, 1, 2].map((i) => (i === j ? 1 : 0)) }),
        ).message,
    ) as number[][]
    const sum = (ties[0] as number[]).map((_, i) => ties.reduce((t, row) => t + (row[i] as number), 0))
    near(m, sum, 1e-12)
  })

  it('sends a scalar the marginal at its coordinate of the vector cavity with the other scalars cavities, what its tie sends', () => {
    for (let j = 0; j < 3; j++) {
      const m = M.gaussianMvStack.ep?.(ctx(j + 1, stackArgs(), { n: 3 })).message as number[]
      // The vector as the tie of coordinate j sees it: its cavity and the diagonal messages of the other ties.
      const h = [...(cavity.slice(0, 3) as number[])]
      const lam = [...(cavity.slice(3) as number[])]
      for (const [i, e] of scalars.entries())
        if (i !== j) {
          h[i] = (h[i] as number) + (e[0] as number)
          lam[i * 3 + i] = (lam[i * 3 + i] as number) + (e[1] as number)
        }
      const seen = [...h, ...lam]
      const tie = M.dotConstant.ep?.(
        ctx(0, [msg(M.gaussian, scalars[j] as M.Eta), msg(vec, seen)], { a: [0, 1, 2].map((i) => (i === j ? 1 : 0)) }),
      ).message as number[]
      near(m, tie, 1e-12)
      // The marginal's moments, from the covariance of that vector.
      const { mean, covariance } = M.gaussianMvMoments(seen, 3)
      near(m, M.gaussianEta(mean[j] as number, covariance[j * 3 + j] as number), 1e-10)
    }
  })

  it('has the evidence of the integral of the vector cavity times the scalars cavities at its coordinates', () => {
    const d = 2
    const v = M.gaussianMv(d)
    const eta = M.gaussianMvEta([0.3, -0.2], [1.4, 0.35, 0.35, 0.9])
    const ys = [M.gaussianEta(0.5, 0.8), M.gaussianEta(-0.1, 1.2)]
    const args = [msg(v, eta), ...ys.map((e) => msg(M.gaussian, e))]
    const logf = ([a, b]: readonly number[]): number => {
      const [x0, x1] = [a as number, b as number]
      const [h0, h1, j00, j01, , j11] = eta as [number, number, number, number, number, number]
      // Unnormalised, as the evidence convention is for a latent: exp(h . x - x' J x / 2) for each cavity.
      let t = h0 * x0 + h1 * x1 - 0.5 * (j00 * x0 * x0 + 2 * j01 * x0 * x1 + j11 * x1 * x1)
      for (const [j, e] of ys.entries())
        t += (e[0] as number) * (j === 0 ? x0 : x1) - 0.5 * (e[1] as number) * (j === 0 ? x0 : x1) ** 2
      return t
    }
    const brute = Math.log(
      integrateBox(
        (p) => Math.exp(logf(p)),
        [
          [-9, 9],
          [-9, 9],
        ],
        16,
      ),
    )
    expect(M.gaussianMvStack.logEvidence?.(ctx(0, args, { n: d }))).toBeCloseTo(brute, 6)
  })

  it('has no evidence while nothing has a density', () => {
    const flat = [msg(vec, vec.flat), ...scalars.map(() => msg(M.gaussian, M.gaussian.flat))]
    expect(M.gaussianMvStack.logEvidence?.(ctx(0, flat, { n: 3 }))).toBeNaN()
  })

  it('refuses a vector of another length than n, and a cavity with no density for the message to a scalar', () => {
    expect(M.gaussianMvStack.check?.({ n: 2 }, [vec, M.gaussian, M.gaussian])).toMatch(/length of the vector/)
    expect(M.gaussianMvStack.check?.({ n: 3 }, [vec, M.gaussian, M.gaussian, M.gaussian])).toBeUndefined()
    expect(() => M.gaussianMvStack.ep?.(ctx(1, [msg(vec, vec.flat), ...stackArgs().slice(1)], { n: 3 }))).toThrow(
      M.ImproperMessageError,
    )
  })

  it('is exact on a linear Gaussian model: the posterior of the vector is the closed form', () => {
    // y_j ~ N(m_j, v_j) independent, observations o_r ~ N(a_r . y, 1 / tau): the posterior of the vector is a Gaussian we can write down.
    const prior = [
      [0.2, 0.9],
      [-0.3, 0.7],
    ] as const
    const rows = [
      { a: [1, 0.6], o: 0.9 },
      { a: [0.3, -1], o: -0.4 },
      { a: [0.8, 0.9], o: 0.5 },
    ]
    const tau = 2.5
    const g = M.compileGraph(
      {
        variables: [
          { name: 'y0', family: { kind: 'gaussian' } },
          { name: 'y1', family: { kind: 'gaussian' } },
          { name: 'd', family: { kind: 'gaussian-mv', size: 2 } },
          ...rows.map((r, i) => ({ name: `o${i}`, family: { kind: 'gaussian' } as M.FamilySpec, observed: r.o })),
        ],
        factors: [
          M.gaussianPrior('y0', prior[0][0], prior[0][1]),
          M.gaussianPrior('y1', prior[1][0], prior[1][1]),
          { operator: 'gaussian-mv-stack', args: ['d', 'y0', 'y1'], params: { n: 2 } },
          ...rows.map((r, i) => ({
            operator: 'linear-regression-row',
            args: [`o${i}`, 'd'],
            params: { a: r.a, precision: tau },
          })),
        ],
      },
      M.defaultOperators(),
    )
    const run = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 500 })
    expect(run.status).toBe('converged')
    // The closed form: precision = diag(1 / v) + tau A'A, h = m / v + tau A'o.
    const lam = [1 / prior[0][1], 0, 0, 1 / prior[1][1]]
    const h = [prior[0][0] / prior[0][1], prior[1][0] / prior[1][1]]
    for (const r of rows) {
      for (let i = 0; i < 2; i++) {
        h[i] = (h[i] as number) + tau * r.o * (r.a[i] as number)
        for (let k = 0; k < 2; k++)
          lam[i * 2 + k] = (lam[i * 2 + k] as number) + tau * (r.a[i] as number) * (r.a[k] as number)
      }
    }
    const det = (lam[0] as number) * (lam[3] as number) - (lam[1] as number) * (lam[2] as number)
    const mean = [
      ((lam[3] as number) * (h[0] as number) - (lam[1] as number) * (h[1] as number)) / det,
      ((lam[0] as number) * (h[1] as number) - (lam[2] as number) * (h[0] as number)) / det,
    ]
    const marginals = M.marginalsOf(g, run.state) as Record<string, Record<string, number | number[]>>
    near(marginals['d']?.['mean'] as number[], mean, 1e-8)
    const variance = [(lam[3] as number) / det, (lam[0] as number) / det]
    near([marginals['y0']?.['variance'] as number, marginals['y1']?.['variance'] as number], variance, 1e-8)
  })
})

describe('gaussian-mv-stack under variational message passing defines the vector', () => {
  it('sends the vector the diagonal Gaussian of the scalars beliefs, and the scalars the expected log of the vector cavity', () => {
    expect(M.gaussianMvStack.defines).toBe(0)
    const args = stackArgs()
    const toVector = M.gaussianMvStack.vmp?.(ctx(0, args, { n: 3 }, 'vmp')).message as number[]
    near(
      toVector,
      [
        ...scalars.map((e) => e[0] as number),
        ...[0, 1, 2].flatMap((i) => [0, 1, 2].map((k) => (i === k ? (scalars[i] as M.Eta)[1] : 0))),
      ] as number[],
      1e-12,
    )
    for (let j = 0; j < 3; j++) {
      const m = M.gaussianMvStack.vmp?.(ctx(j + 1, args, { n: 3 }, 'vmp')).message as number[]
      // h_j minus the couplings to the other coordinates at their means, and the diagonal of the cavity precision.
      let h = cavity[j] as number
      for (let i = 0; i < 3; i++)
        if (i !== j) {
          const e = scalars[i] as M.Eta
          h -= (cavity[3 + j * 3 + i] as number) * ((e[0] as number) / (e[1] as number))
        }
      near(m, [h, cavity[3 + j * 3 + j] as number], 1e-12)
    }
  })
})
