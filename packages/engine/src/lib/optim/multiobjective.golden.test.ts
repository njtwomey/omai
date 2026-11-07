import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { VECTOR_PROBLEMS } from '../../testing/optim-problems.ts'
import { type Tensor, tensor, type Value } from '../tensor/index.ts'
import { crowdingDistance, hypervolume, hypervolumeContributions, nonDominatedSort } from './multiobjective.ts'
import { nsga2 } from './nsga2.ts'

describe('multiobjective against pymoo and moocore', () => {
  for (const c of loadFixture('optim/multiobjective').cases) {
    it(c.name, () => {
      const points = c.inputs[0] === undefined ? undefined : (decode(c.inputs[0]) as Tensor)
      const reference = c.params['reference'] as number[] | undefined
      switch (c.fn) {
        case 'nonDominatedSort': {
          const fronts = nonDominatedSort(points as Tensor)
          const ranks = new Int32Array(points?.shape[0] ?? 0)
          for (const [r, front] of fronts.entries()) for (const i of front) ranks[i] = r
          expect(Array.from(ranks)).toEqual(Array.from((decode(c.output) as Tensor).data))
          break
        }
        case 'crowdingDistance':
          expectMatches(crowdingDistance(points as Tensor), c.output, c.rtol, c.atol, c.name)
          break
        case 'hypervolume':
          expectMatches(tensor(hypervolume(points as Tensor, reference ?? [])), c.output, c.rtol, c.atol, c.name)
          break
        case 'hypervolumeContributions':
          expectMatches(hypervolumeContributions(points as Tensor, reference ?? []), c.output, c.rtol, c.atol, c.name)
          break
        default: {
          const f = VECTOR_PROBLEMS[c.params['problem'] as string] as (x: Value) => Value
          const n = c.params['nVar'] as number
          const r = nsga2((x) => f(x as Value), new Array(n).fill(0.5), {
            bounds: new Array(n).fill([0, 1]),
            popSize: c.params['popSize'] as number,
            maxIterations: c.params['generations'] as number,
            seed: 1,
          })
          expectMatches(tensor(hypervolume(r.paretoFront, reference ?? [])), c.output, c.rtol, c.atol, c.name)
        }
      }
    })
  }
})
