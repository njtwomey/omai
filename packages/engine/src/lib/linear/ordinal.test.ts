import { describe, expect, it } from 'vitest'
import { matrixView } from '../learn/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { linearModel } from './model.ts'
import {
  type OrdinalLink,
  type OrdinalMode,
  ordinal,
  ordinalAlgorithm,
  ordinalLogLikelihood,
  ordinalProbabilities,
} from './ordinal.ts'

const n = 60
const x = Array.from({ length: n }, (_, i) => [Math.sin(i * 1.7) * 2, Math.cos(i * 0.9)])
const latent = x.map((r, i) => (r[0] ?? 0) - 0.5 * (r[1] ?? 0) + 0.9 * Math.sin(i * 12.9898))
const y = latent.map((v) => (v < -1 ? 0 : v < 0 ? 1 : v < 1.2 ? 2 : 3))

const problem = (xs: number[][], codes: number[], K: number, w?: number[]) => ({
  x: matrixView(xs, 'test'),
  codes,
  w: w ?? codes.map(() => 1),
  nClasses: K,
})

describe('ordinal likelihood', () => {
  const links: OrdinalLink[] = ['logit', 'probit', 'cloglog', 'loglog']
  const modes: OrdinalMode[] = ['cumulative', 'sequential']
  for (const link of links)
    for (const mode of modes)
      it(`has the gradient of its finite differences (${mode}, ${link})`, () => {
        const p = problem(x, y, 4)
        const beta = [0.7, -0.3]
        const theta = [-1.1, 0.2, 1.4]
        const e = ordinalLogLikelihood(p, beta, theta, link, mode)
        const ll = (b: number[], t: number[]): number => ordinalLogLikelihood(p, b, t, link, mode).logLikelihood
        const h = 1e-6
        for (let j = 0; j < 2; j++) {
          const up = [...beta]
          const dn = [...beta]
          up[j] = (up[j] ?? 0) + h
          dn[j] = (dn[j] ?? 0) - h
          expect(e.gradCoef[j]).toBeCloseTo((ll(up, theta) - ll(dn, theta)) / (2 * h), 5)
        }
        for (let k = 0; k < 3; k++) {
          const up = [...theta]
          const dn = [...theta]
          up[k] = (up[k] ?? 0) + h
          dn[k] = (dn[k] ?? 0) - h
          expect(e.gradThresholds[k]).toBeCloseTo((ll(beta, up) - ll(beta, dn)) / (2 * h), 5)
        }
      })

  it('has category probabilities that sum to one and whose log is the likelihood', () => {
    const eta = [0.3, -1.2]
    for (const mode of modes) {
      const pr = ordinalProbabilities(eta, [-0.5, 0.4, 1], 'probit', mode)
      for (let i = 0; i < 2; i++) expect(pr.slice(i * 4, i * 4 + 4).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      const p = problem([[0.3], [-1.2]], [2, 0], 4)
      const ll = ordinalLogLikelihood(p, [1], [-0.5, 0.4, 1], 'probit', mode).logLikelihood
      expect(ll).toBeCloseTo(Math.log(pr[2] ?? 0) + Math.log(pr[4] ?? 0), 12)
    }
  })
})

describe('ordinal regression', () => {
  it('reaches a stationary point with increasing thresholds', () => {
    const f = ordinal({ tol: 1e-10 }).fit({ x, y })
    expect(f.params.converged).toBe(true)
    const th = f.params.thresholds
    expect(th[0]).toBeLessThan(th[1] ?? 0)
    expect(th[1]).toBeLessThan(th[2] ?? 0)
    const e = ordinalLogLikelihood(problem(x, y, 4), f.params.coef, th)
    for (const g of [...e.gradCoef, ...e.gradThresholds]) expect(Math.abs(g)).toBeLessThan(1e-6)
    expect(f.params.coef[0]).toBeGreaterThan(0)
    expect(f.score({ x, y })).toBeGreaterThan(0.5)
    expect(f.classes).toEqual([0, 1, 2, 3])
  })

  it('is logistic regression for two categories, in either mode', () => {
    const y2 = y.map((v) => (v >= 2 ? 1 : 0))
    const glm = linearModel({ family: 'binomial', alpha: 0 }).fit({ x, y: y2 })
    for (const mode of ['cumulative', 'sequential'] as const) {
      const f = ordinal({ mode, tol: 1e-11 }).fit({ x, y: y2 })
      f.params.coef.forEach((v, j) => {
        expect(v).toBeCloseTo(glm.params.coef[j] ?? 0, 6)
      })
      expect(f.params.thresholds[0]).toBeCloseTo(-glm.params.intercept, 6)
    }
  })

  it('treats integer weights as repeated rows', () => {
    const w = x.map((_, i) => (i % 4 === 0 ? 2 : 1))
    const a = ordinal({ tol: 1e-11 }).fit({ x, y, weights: w })
    const b = ordinal({ tol: 1e-11 }).fit({
      x: x.flatMap((r, i) => (w[i] === 2 ? [r, r] : [r])),
      y: y.flatMap((v, i) => (w[i] === 2 ? [v, v] : [v])),
    })
    a.params.coef.forEach((v, j) => {
      expect(v).toBeCloseTo(b.params.coef[j] ?? 0, 7)
    })
    expect(a.params.logLikelihood).toBeCloseTo(b.params.logLikelihood, 8)
  })

  it('shrinks the slopes with a ridge penalty and predicts probabilities that favour the right end', () => {
    const plain = ordinal().fit({ x, y }).params.coef
    const ridge = ordinal({ alpha: 0.5 }).fit({ x, y }).params.coef
    expect(Math.abs(ridge[0] ?? 0)).toBeLessThan(Math.abs(plain[0] ?? 0))
    const f = ordinal().fit({ x, y })
    const pr = toFlat(
      f.predictProba([
        [-3, 0],
        [3, 0],
      ]),
    )
    expect(pr[0]).toBeGreaterThan(pr[3] ?? 1)
    expect(pr[4 + 3]).toBeGreaterThan(pr[4] ?? 1)
    expect(toFlat(f.decisionFunction([[1, 0]]))[0]).toBeCloseTo(f.params.coef[0] ?? 0, 12)
  })

  it('can be stopped and resumed from a serialised state', () => {
    const alg = ordinalAlgorithm(problem(x, y, 4))
    const full = runAlgorithm(alg, { maxSteps: 3, key: key(0) })
    const first = runAlgorithm(alg, { maxSteps: 1, key: key(0) })
    const rest = runAlgorithm(alg, {
      maxSteps: 2,
      key: key(0),
      start: { state: alg.serialise?.(first.state), step: first.steps, serialised: true },
    })
    expect(rest.state.u).toEqual(full.state.u)
  })

  it('reports a Wald table with a row per slope and threshold', () => {
    const f = ordinal().fit({ x, y })
    const table = f.exposition[0]
    expect(table?.kind).toBe('coefficientTable')
    if (table?.kind === 'coefficientTable') {
      expect(table.names).toEqual(['x0', 'x1', '0|1', '1|2', '2|3'])
      expect(table.standardErrors?.every((v) => v > 0)).toBe(true)
    }
  })
})
