import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import {
  asymmetricReplicatorDynamics,
  bimatrixGame,
  fictitiousPlay,
  normalFormGame,
  projectedReplicatorDynamics,
  regretMatching,
  replicatorDynamics,
} from './index.ts'

const matrices = (inputs: readonly unknown[]): number[][][] =>
  inputs.map((x) => toArray(decode(x) as Tensor) as number[][])

describe('fictitious play against nashpy', () => {
  for (const c of loadFixture('games/learning').cases) {
    it(c.name, () => {
      const [a, b] = matrices(c.inputs)
      const r = fictitiousPlay(bimatrixGame(a ?? [], b), {
        iterations: (c.params['plays'] as number[][]).length,
        initialCounts: c.params['initial'] as number[][],
      })
      expect(r.plays).toEqual(c.params['plays'])
      for (const [p, counts] of (c.params['counts'] as number[][]).entries())
        for (const [k, v] of counts.entries()) expect(r.counts[p]?.[k]).toBeCloseTo(v, 12)
    })
  }
})

describe('regret matching against OpenSpiel', () => {
  for (const c of loadFixture('games/no_regret').cases) {
    it(c.name, () => {
      const game = normalFormGame(c.inputs.map((x) => decode(x) as Tensor))
      const r = regretMatching(game, { iterations: c.params['iterations'] as number, gamma: 1e-6, initialRegret: 1e-6 })
      for (const [p, x] of (c.params['average'] as number[][]).entries())
        for (const [a, v] of x.entries())
          expect(Math.abs((r.average[p]?.[a] ?? 0) - v)).toBeLessThanOrEqual(c.atol + c.rtol * Math.abs(v))
    })
  }
})

describe('evolutionary dynamics against nashpy and OpenSpiel', () => {
  for (const c of loadFixture('games/evolutionary').cases) {
    it(c.name, () => {
      const p = c.params
      const timepoints = p['timepoints'] as number[] | undefined
      if (c.fn === 'replicatorDynamics') {
        const xs = replicatorDynamics(decode(c.inputs[0]) as Tensor, {
          x0: p['x0'] as number[],
          timepoints: timepoints ?? [],
          ...(p['mutation'] === undefined ? {} : { mutation: p['mutation'] as number[][] }),
        })
        expectMatches(tensor(xs), c.output, c.rtol, c.atol, c.name)
      } else if (c.fn === 'asymmetricReplicatorDynamics') {
        const [a, b] = matrices(c.inputs)
        const r = asymmetricReplicatorDynamics(bimatrixGame(a ?? [], b), {
          x0: p['x0'] as number[],
          y0: p['y0'] as number[],
          timepoints: timepoints ?? [],
        })
        expectMatches(tensor(r.row.map((x, t) => [...x, ...(r.column[t] ?? [])])), c.output, c.rtol, c.atol, c.name)
      } else {
        const game = normalFormGame(c.inputs.map((x) => decode(x) as Tensor))
        const r = projectedReplicatorDynamics(game, {
          iterations: p['iterations'] as number,
          dt: p['dt'] as number,
          gamma: p['gamma'] as number,
        })
        for (const [k, x] of (p['average'] as number[][]).entries())
          for (const [a, v] of x.entries()) expect(Math.abs((r.average[k]?.[a] ?? 0) - v)).toBeLessThan(1e-10)
      }
    })
  }
})
