import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import {
  aggregateMean,
  aggregateMedian,
  alignCurves,
  interquartileMean,
  normalizeScores,
  optimalityGap,
  percentileInterval,
  performanceProfile,
  probabilityOfImprovement,
} from './evaluation.ts'

const matrix = (x: unknown): number[][] => toArray(decode(x) as Tensor) as number[][]
const vector = (x: unknown): number[] => toFlat(decode(x) as Tensor)

describe('aggregate metrics against scipy.stats.trim_mean, scipy.stats.mannwhitneyu and numpy', () => {
  for (const c of loadFixture('rl/evaluation').cases) {
    it(c.name, () => {
      const [a, b, d] = c.inputs
      let got: number | number[] | number[][]
      switch (c.fn) {
        case 'iqm':
          got = interquartileMean(matrix(a))
          break
        case 'mean':
          got = aggregateMean(matrix(a))
          break
        case 'median':
          got = aggregateMedian(matrix(a))
          break
        case 'optimalityGap':
          got = optimalityGap(matrix(a), c.params['gamma'] as number)
          break
        case 'runProfile':
        case 'averageProfile':
          got = [
            ...performanceProfile(matrix(a), vector(b), {
              reps: 0,
              kind: c.fn === 'runProfile' ? 'run' : 'average',
            }).value,
          ]
          break
        case 'poi':
          got = probabilityOfImprovement(matrix(a), matrix(b))
          break
        case 'normalize':
          got = normalizeScores(matrix(a), vector(b), vector(d))
          break
        case 'percentileInterval':
          got = percentileInterval(vector(a), c.params['level'] as number)
          break
        default: {
          const r = alignCurves([{ steps: vector(a), values: vector(b) }], { grid: vector(d) })
          got = r.values[0] ?? []
        }
      }
      expectMatches(typeof got === 'number' ? got : tensor(got), c.output, c.rtol, c.atol, c.name)
    })
  }
})
