// Wilson's and the Aldous-Broder algorithms sample uniform spanning trees: on tiny grids, each of the N spanning trees
// (N from the matrix-tree theorem, by networkx) appears about equally often. A chi-square statistic with N - 1 degrees
// of freedom has mean N - 1 and standard deviation sqrt(2 (N - 1)); the test allows six standard deviations, and checks
// that every tree appears. The other algorithms are not uniform, and the test shows it for the binary tree.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { key } from '../random/index.ts'
import { generateMaze, isPerfectMaze, type MazeAlgorithmName } from './index.ts'

function counts(
  algorithm: MazeAlgorithmName,
  rows: number,
  cols: number,
  topology: 'square' | 'hex',
  samples: number,
): Map<string, number> {
  const seen = new Map<string, number>()
  for (let i = 0; i < samples; i++) {
    const w = generateMaze(key(i), { rows, cols, topology, algorithm })
    expect(isPerfectMaze(w)).toBe(true)
    const k = Array.from(w.walls).join(',')
    seen.set(k, (seen.get(k) ?? 0) + 1)
  }
  return seen
}

const chiSquare = (seen: Map<string, number>, n: number, samples: number): number => {
  const expected = samples / n
  let x = 0
  for (const c of seen.values()) x += (c - expected) ** 2 / expected
  return x + (n - seen.size) * expected
}

describe('uniform spanning trees against the matrix-tree theorem', () => {
  for (const c of loadFixture('worlds/generators').cases) {
    const n = (c.output as { data: number[] }).data[0] ?? 0
    const { rows, cols, topology, determinant } = c.params as {
      rows: number
      cols: number
      topology: 'square' | 'hex'
      determinant: number
    }
    it(`${c.name}: ${n} trees`, () => {
      expect(determinant).toBe(n)
      const samples = Math.min(40 * n, 4000)
      for (const algorithm of ['wilson', 'aldousBroder'] as const) {
        const seen = counts(algorithm, rows, cols, topology, samples)
        expect(seen.size, algorithm).toBe(n)
        expect(chiSquare(seen, n, samples), algorithm).toBeLessThan(n - 1 + 6 * Math.sqrt(2 * (n - 1)))
      }
    })
  }

  it('the test has power: the binary tree never makes some trees, and the backtracker and Prim fail the chi-square', () => {
    expect(counts('binaryTree', 3, 3, 'square', 600).size).toBeLessThan(192)
    for (const algorithm of ['backtracker', 'prim'] as const) {
      const seen = counts(algorithm, 3, 3, 'square', 4000)
      expect(chiSquare(seen, 192, 4000), algorithm).toBeGreaterThan(191 + 6 * Math.sqrt(2 * 191))
    }
  })
})
