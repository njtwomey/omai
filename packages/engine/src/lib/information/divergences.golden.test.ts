import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
import * as I from './index.ts'

type Params = Readonly<Record<string, unknown>>
const at = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor
const num = (v: unknown): number => (v === 'inf' ? Number.POSITIVE_INFINITY : (v as number))

function grid(xs: readonly Tensor[], p: Params): I.GridOptions {
  return {
    ...(p['dx'] === undefined ? {} : { dx: p['dx'] as number }),
    ...(xs[2] === undefined ? {} : { x: xs[2] }),
    ...(p['base'] === undefined ? {} : { base: p['base'] as number }),
  }
}

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => Value>> = {
  fDivergence: (x, p) => I.fDivergence(at(x, 0), at(x, 1), p['f'] as I.FGeneratorName, grid(x, p)),
  renyiDivergence: (x, p) => I.renyiDivergence(at(x, 0), at(x, 1), num(p['alpha']), grid(x, p)),
  renyiEntropy: (x, p) => I.renyiEntropy(at(x, 0), num(p['alpha']), grid(x, p)),
  alphaDivergence: (x, p) => I.alphaDivergence(at(x, 0), at(x, 1), num(p['alpha']), grid(x, p)),
  betaDivergence: (x, p) => I.betaDivergence(at(x, 0), at(x, 1), num(p['beta']), grid(x, p)),
  bregmanDivergence: (x, p) => I.bregmanDivergence(at(x, 0), at(x, 1), p['phi'] as I.BregmanName, grid(x, p)),
}

describe('information/divergences against scipy and sklearn', () => {
  for (const c of loadFixture('information/divergences').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(
        call(
          c.inputs.map((x) => decode(x) as Tensor),
          c.params,
        ),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})
