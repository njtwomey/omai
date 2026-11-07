import { describe, expect, it } from 'vitest'
import { solveKernel } from './_kernels.ts'
import { eigenvectorKernel, schurEigenvalues, schurEigenvectors, schurKernel } from './_schur.ts'

// The O(n^3) Schur-form eigenvectors against the dense 2n x 2n inverse iteration they replaced (the reference).
function lcg(seed: number): () => number {
  let s = seed
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296 - 0.5
  }
}

const random = (n: number, seed: number): number[][] => {
  const r = lcg(seed)
  return Array.from({ length: n }, () => Array.from({ length: n }, r))
}

const cases: readonly (readonly [string, number[][]])[] = [
  ['1 x 1', [[2]]],
  ['random 4 (complex pairs)', random(4, 1)],
  ['random 7', random(7, 2)],
  ['random 12', random(12, 3)],
  [
    'rotations',
    [
      [0, -1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 0, -3],
      [0, 0, 3, 0],
    ],
  ],
  [
    'companion',
    [
      [0, 0, 0, -24],
      [1, 0, 0, 50],
      [0, 1, 0, -35],
      [0, 0, 1, 10],
    ],
  ],
  [
    'Jordan block',
    [
      [2, 1, 0],
      [0, 2, 1],
      [0, 0, 2],
    ],
  ],
  [
    'upper triangular, repeated',
    [
      [1, 2, 3],
      [0, 1, 4],
      [0, 0, 1],
    ],
  ],
]

describe('schurEigenvectors', () => {
  for (const [name, rows] of cases) {
    it(`agrees with the dense inverse iteration: ${name}`, () => {
      const n = rows.length
      const a = Float64Array.from(rows.flat())
      const { t, z } = schurKernel(a, n)
      const { re, im } = schurEigenvalues(t, n)
      const fast = schurEigenvectors(t, z, n, re, im)
      for (let k = 0; k < n; k++) {
        const slow = eigenvectorKernel(a, n, re[k] ?? 0, im[k] ?? 0, (m, b, size) => solveKernel(m, b, size, 1))
        for (let i = 0; i < n; i++) {
          expect(fast.re[i * n + k] ?? 0).toBeCloseTo(slow.re[i] ?? 0, 9)
          expect(fast.im[i * n + k] ?? 0).toBeCloseTo(slow.im[i] ?? 0, 9)
        }
      }
    })
  }
})
