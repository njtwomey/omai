import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { gaussianGraph } from '../datasets/index.ts'
import { linalg } from '../numerics/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { adaptiveGraphicalLasso, adaptiveGraphicalLassoAlgorithm } from './adaptive-graphical-lasso.ts'
import { edgeRecovery, precisionGraph } from './evaluation.ts'
import { graphicalLasso } from './graphical-lasso.ts'

const planted = (n: number, p = 8, seed = 1) => {
  const { data, truth } = generate(gaussianGraph, {
    key: key(seed),
    knobs: { n, p, structure: 'band', strength: 0.4 },
  })
  return { s: toArray(empiricalCovariance(data.x)) as number[][], adjacency: truth.extra?.['adjacency'] as boolean[][] }
}
const tight = { tol: 1e-9, enetTol: 1e-11, maxIter: 400 }

describe('the adaptive graphical lasso', () => {
  it('with no weighting (gamma 0) is the plain graphical lasso, and with no rounds it is the initial estimate', () => {
    const { s } = planted(150)
    const plain = graphicalLasso(s, 0.1, tight)
    const flat = adaptiveGraphicalLasso(s, 0.1, { ...tight, gamma: 0 })
    for (let a = 0; a < 8; a++)
      for (let b = 0; b < 8; b++) expect(flat.precision[a]?.[b]).toBeCloseTo(plain.precision[a]?.[b] ?? 0, 6)
    const none = adaptiveGraphicalLasso(s, 0.1, { ...tight, rounds: 0 })
    expect(none.rounds).toBe(0)
    for (let a = 0; a < 8; a++)
      for (let b = 0; b < 8; b++) expect(none.precision[a]?.[b]).toBeCloseTo(plain.precision[a]?.[b] ?? 0, 6)
  })

  it('keeps the true edges and removes the false ones that the plain estimate at the same penalty keeps', () => {
    const { s, adjacency } = planted(600)
    const plain = edgeRecovery(precisionGraph(graphicalLasso(s, 0.05, tight).precision), adjacency)
    const adaptive = edgeRecovery(precisionGraph(adaptiveGraphicalLasso(s, 0.05, tight).precision), adjacency)
    expect(plain.falsePositives).toBeGreaterThan(0)
    expect(adaptive.falsePositives).toBeLessThan(plain.falsePositives)
    expect(adaptive.recall).toBe(1)
  })

  it('solves the weighted problem its last round states: the optimality conditions hold with the recorded weights', () => {
    const { s } = planted(300)
    const out = adaptiveGraphicalLasso(s, 0.05, { ...tight, rounds: 2 })
    expect(out.rounds).toBe(2)
    const sigma = toArray(linalg.inv(out.precision)) as number[][]
    let active = 0
    for (let a = 0; a < 8; a++)
      for (let b = 0; b < 8; b++) {
        if (a === b) continue
        const penalty = 0.05 * (out.weights[a]?.[b] ?? 0)
        const gradient = (s[a]?.[b] ?? 0) - (sigma[a]?.[b] ?? 0)
        expect(Math.abs(gradient)).toBeLessThan(penalty + 1e-4)
        if (Math.abs(out.precision[a]?.[b] ?? 0) > 1e-8) {
          active++
          expect(Math.abs(gradient)).toBeCloseTo(penalty, 3)
        }
      }
    expect(active).toBeGreaterThan(0)
  })

  it('weights strong entries less than weak ones, and the weights fall as the precision grows', () => {
    const { s } = planted(300)
    const out = adaptiveGraphicalLasso(s, 0.05, tight)
    let strong = [0, 0, 0]
    let weak = [0, 0, 0]
    for (let a = 0; a < 8; a++)
      for (let b = a + 1; b < 8; b++) {
        const size = Math.abs(out.precision[a]?.[b] ?? 0)
        if (size > (strong[0] ?? 0)) strong = [size, a, b]
        if (size === 0) weak = [size, a, b]
      }
    expect(out.weights[strong[1] as number]?.[strong[2] as number]).toBeLessThan(
      out.weights[weak[1] as number]?.[weak[2] as number] ?? 0,
    )
  })

  it('starts from the inverse covariance or from a given precision, and traces a step per round', () => {
    const { s } = planted(400)
    const inverse = adaptiveGraphicalLasso(s, 0.05, { ...tight, initial: 'inverse' })
    const given = adaptiveGraphicalLasso(s, 0.05, { ...tight, initial: toArray(linalg.inv(s)) as number[][] })
    expect(precisionGraph(inverse.precision)).toEqual(precisionGraph(given.precision))
    const alg = adaptiveGraphicalLassoAlgorithm(s, 0.05, { rounds: 3 })
    const run = runAlgorithm(alg, { maxSteps: 10 })
    expect(run.steps).toBe(3)
    expect(run.state.round).toBe(3)
    expect(Number.isFinite(run.state.change)).toBe(true)
    expect(alg.deserialise?.(alg.serialise?.(run.state) ?? null)).toEqual(run.state)
  })

  it('rejects bad options', () => {
    const { s } = planted(100, 4)
    expect(() => adaptiveGraphicalLasso(s, -1)).toThrow(/non-negative/)
    expect(() => adaptiveGraphicalLasso(s, 0.1, { epsilon: 0 })).toThrow(/epsilon/)
    expect(() => adaptiveGraphicalLasso(s, 0.1, { rounds: 1.5 })).toThrow(/non-negative/)
  })
})
