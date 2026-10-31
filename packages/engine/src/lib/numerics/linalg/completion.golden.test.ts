import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { fromStorage, type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as L from './index.ts'

describe('matrix completion against dense numpy implementations of the same updates', () => {
  for (const c of loadFixture('numerics/linalg/completion').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const m = p['m'] as number
      const n = p['n'] as number
      const obs: L.ObservedEntries = {
        shape: [m, n],
        rows: toFlat(t[0] as Tensor),
        cols: toFlat(t[1] as Tensor),
        values: toFlat(t[2] as Tensor),
      }
      const init = { u: t[3] as Tensor, v: t[4] as Tensor }
      const rank = (t[3] as Tensor).shape[1] as number
      const matrix = (x: readonly number[], rows: number, cols: number): Tensor =>
        fromStorage(Float64Array.from(x), [rows, cols], 'float64')
      if (c.fn === 'alsU' || c.fn === 'alsV') {
        const alg = L.alsCompletionAlgorithm(obs, { rank, lambda: p['lam'] as number, init })
        const s = runAlgorithm(alg, { maxSteps: p['half'] as number, seed: 0 }).state
        expectMatches(c.fn === 'alsU' ? matrix(s.u, m, rank) : matrix(s.v, n, rank), c.output, c.rtol, c.atol, c.name)
      } else if (c.fn === 'implicitU' || c.fn === 'implicitV') {
        const alg = L.implicitAlsAlgorithm(obs, { rank, lambda: p['lam'] as number, alpha: p['alpha'] as number, init })
        const s = runAlgorithm(alg, { maxSteps: p['half'] as number, seed: 0 }).state
        expectMatches(
          c.fn === 'implicitU' ? matrix(s.u, m, rank) : matrix(s.v, n, rank),
          c.output,
          c.rtol,
          c.atol,
          c.name,
        )
      } else if (c.fn === 'sgd') {
        const alg = L.sgdCompletionAlgorithm(obs, {
          rank,
          lambda: p['lam'] as number,
          learningRate: p['lr'] as number,
          order: 'sequential',
          init,
        })
        const s = runAlgorithm(alg, { maxSteps: p['epochs'] as number, seed: 0 }).state
        expectMatches(L.factorsToMatrix(s, rank), c.output, c.rtol, c.atol, c.name)
      } else {
        const alg = L.softImputeAlgorithm(obs, { lambda: p['lam'] as number, tol: 0 })
        const s = runAlgorithm(alg, { maxSteps: p['steps'] as number, seed: 0 }).state
        expectMatches(c.fn === 'softImpute' ? matrix(s.z, m, n) : tensor(s.objective), c.output, c.rtol, c.atol, c.name)
      }
    })
  }
})
