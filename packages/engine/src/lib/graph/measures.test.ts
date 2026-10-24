import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import {
  algebraicConnectivity,
  barbellGraph,
  cheegerConstant,
  clustering,
  completeGraph,
  conductance,
  connectedComponents,
  cycleGraph,
  density,
  digraph,
  eccentricity,
  edgeExpansion,
  graph,
  homophily,
  isConnected,
  radius,
  spectralGap,
  triangles,
} from './index.ts'

/** Brute-force minimum over all proper non-empty subsets. */
function bruteCheeger(g: ReturnType<typeof graph>, kind: 'conductance' | 'expansion'): number {
  let best = Number.POSITIVE_INFINITY
  for (let mask = 1; mask < 2 ** g.n - 1; mask++) {
    const s = Array.from({ length: g.n }, (_, v) => v).filter((v) => ((mask >> v) & 1) === 1)
    const value = kind === 'conductance' ? conductance(g, s) : edgeExpansion(g, s)
    if (value < best) best = value
  }
  return best
}

describe('measures', () => {
  it('finds the Cheeger constant by exhaustive Gray-code search, within the spectral bounds', () => {
    fc.assert(
      fc.property(arbitraryGraph({ minNodes: 3, maxNodes: 8, density: 0.5 }), (g) => {
        if (!isConnected(g)) return
        const h = cheegerConstant(g)
        expect(h.value).toBeCloseTo(bruteCheeger(g, 'conductance'), 12)
        expect(conductance(g, [...h.set])).toBeCloseTo(h.value, 12)
        expect(cheegerConstant(g, { kind: 'expansion' }).value).toBeCloseTo(bruteCheeger(g, 'expansion'), 12)
        // Cheeger's inequality for the normalised Laplacian: lambda_2 / 2 <= h <= sqrt(2 lambda_2).
        const mu = spectralGap(g)
        expect(mu / 2).toBeLessThanOrEqual(h.value + 1e-12)
        expect(h.value).toBeLessThanOrEqual(Math.sqrt(2 * mu) + 1e-12)
      }),
    )
    expect(() => cheegerConstant(digraph([[0, 1]]))).toThrow()
    expect(() => cheegerConstant(completeGraph(30))).toThrow()
  })

  it('measures homophily three ways', () => {
    const g = barbellGraph(3, 0)
    const labels = [0, 0, 0, 1, 1, 1]
    expect(homophily(g, labels, { kind: 'node' })).toBeCloseTo((4 * 1 + 2 * (2 / 3)) / 6, 12)
    expect(homophily(g, labels, { kind: 'class' })).toBeCloseTo(2 * (6 / 7 - 1 / 2), 12)
    expect(homophily(g, [0, 1, 0, 1, 0, 1])).toBeCloseTo(2 / 7, 12)
    expect(homophily(graph([], { nodes: 2 }), [0, 1])).toBe(0)
    expect(homophily(graph([[0, 1]]), [0, 0], { kind: 'class' })).toBe(1)
  })

  it('computes distances, clustering and spectral quantities on small cases', () => {
    expect(Array.from(eccentricity(cycleGraph(6)))).toEqual([3, 3, 3, 3, 3, 3])
    expect(() => eccentricity(graph([[0, 1]], { nodes: 3 }))).toThrow()
    expect(radius(graph([], { nodes: 0 }))).toBe(0)
    expect(density(graph([], { nodes: 1 }))).toBe(0)
    expect(density(digraph([[0, 1]]))).toBe(0.5)
    expect(() => triangles(digraph([[0, 1]]))).toThrow()
    // A directed triangle with one reciprocal pair (Fagiolo's formula).
    const d = digraph([
      [0, 1],
      [1, 2],
      [2, 0],
      [1, 0],
    ])
    expect(Array.from(clustering(d)).every((c) => c > 0 && c <= 1)).toBe(true)
    expect(algebraicConnectivity(graph([[0, 1]], { nodes: 3 }))).toBeCloseTo(0, 12)
    expect(() => algebraicConnectivity(graph([], { nodes: 1 }))).toThrow()
    expect(() => spectralGap(graph([], { nodes: 1 }))).toThrow()
    expect(connectedComponents(barbellGraph(3, 0)).count).toBe(1)
  })
})
