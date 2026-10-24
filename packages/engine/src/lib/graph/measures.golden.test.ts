import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { fromStorage, type Tensor, tensor } from '../tensor/index.ts'
import {
  algebraicConnectivity,
  attributeAssortativity,
  averageClustering,
  center,
  clustering,
  conductance,
  degreeAssortativity,
  density,
  diameter,
  eccentricity,
  edgeExpansion,
  numericAssortativity,
  periphery,
  radius,
  spectralGap,
  transitivity,
  triangles,
} from './index.ts'

const vec = (x: ArrayLike<number>) => fromStorage(Float64Array.from(x), [x.length], 'float64')

describe('graph measures against networkx', () => {
  for (const c of loadFixture('graph/measures').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      const match = (y: Parameters<typeof expectMatches>[0]): void => expectMatches(y, c.output, c.rtol, c.atol, c.name)
      switch (c.fn) {
        case 'density':
          return match(tensor(density(g)))
        case 'distances':
          match(vec(eccentricity(g)))
          expect(diameter(g)).toBe(p['diameter'])
          expect(radius(g)).toBe(p['radius'])
          expect(center(g)).toEqual(p['center'])
          expect(periphery(g)).toEqual(p['periphery'])
          expect(diameter(g, { weighted: true })).toBe(p['weightedDiameter'])
          return
        case 'clustering':
          return match(vec(clustering(g)))
        case 'clusteringWeighted':
          return match(vec(clustering(g, { weighted: true })))
        case 'averageClustering':
          return match(tensor(averageClustering(g)))
        case 'assortativity':
          return match(tensor(degreeAssortativity(g)))
        case 'attributeAssortativity':
          return match(
            vec([
              numericAssortativity(g, p['values'] as number[]),
              attributeAssortativity(g, p['categories'] as number[]),
            ]),
          )
        case 'expansion': {
          const s = p['subset'] as number[]
          return match(vec([conductance(g, s), edgeExpansion(g, s)]))
        }
        case 'triangles':
          match(vec(triangles(g)))
          expect(transitivity(g)).toBeCloseTo(p['transitivity'] as number, 12)
          return
        case 'spectrum': {
          const want = Array.from((decode(c.output) as Tensor).data)
          match(
            vec([
              algebraicConnectivity(g),
              algebraicConnectivity(g, { normalised: true }),
              spectralGap(g),
              spectralGap(g, { absolute: true }),
            ]),
          )
          expect(want.length).toBe(4)
          return
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
