import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { fromStorage, type Tensor, tensor } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as L from './index.ts'

function projector(q: readonly number[], n: number, k: number): number[] {
  const p = new Array<number>(n * n).fill(0)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let c = 0; c < k; c++) p[i * n + j] = (p[i * n + j] ?? 0) + (q[i * k + c] ?? 0) * (q[j * k + c] ?? 0)
  return p
}

describe('subspace iteration and Jacobi SVD against numpy and scipy.linalg', () => {
  for (const c of loadFixture('numerics/linalg/iterations').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const a = t[0] as Tensor
      if (c.fn === 'projector' || c.fn === 'ritz') {
        const n = a.shape[0] as number
        const k = (t[1] as Tensor).shape[1] as number
        const alg = L.subspaceIterationAlgorithm(a, k, { x0: t[1] as Tensor, tol: 0 })
        const state = runAlgorithm(alg, { maxSteps: c.params['steps'] as number, seed: 0 }).state
        const actual =
          c.fn === 'projector'
            ? fromStorage(Float64Array.from(projector(state.q, n, k)), [n, n], 'float64')
            : tensor([...state.ritzValues])
        expectMatches(actual, c.output, c.rtol, c.atol, c.name)
      } else if (c.fn === 'converged') {
        const r = L.subspaceIteration(a, (t[1] as Tensor).shape[1] as number, {
          x0: t[1] as Tensor,
          tol: 1e-12,
          maxIterations: 500,
        })
        expect(r.converged).toBe(true)
        expectMatches(tensor([...r.values]), c.output, c.rtol, c.atol, c.name)
      } else {
        const r = L.jacobiSvd(a)
        expect(r.converged).toBe(true)
        if (c.fn === 'values') expectMatches(tensor([...r.s]), c.output, c.rtol, c.atol, c.name)
        else {
          const [m, n] = a.shape as [number, number]
          const us = Array.from(r.u.data, (v, i) => v * (r.s[i % n] ?? 0))
          const rec = new Array<number>(m * n).fill(0)
          for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++)
              for (let l = 0; l < n; l++)
                rec[i * n + j] = (rec[i * n + j] ?? 0) + (us[i * n + l] ?? 0) * (r.vt.data[l * n + j] ?? 0)
          expectMatches(fromStorage(Float64Array.from(rec), [m, n], 'float64'), c.output, c.rtol, c.atol, c.name)
        }
      }
    })
  }
})
