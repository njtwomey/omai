import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, type Value } from '../tensor/index.ts'
import { multinomialResample, nextTemperature, stratifiedResample, systematicResample } from './smc.ts'

type Params = Readonly<Record<string, unknown>>

const indices = (x: Int32Array): Value => tensor(Array.from(x))

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => Value | number>> = {
  systematic: ([w, u], p) =>
    indices(systematicResample(w as Tensor, (u as Tensor).data[0] as number, p['n'] as number)),
  stratified: ([w, u], p) =>
    indices(stratifiedResample(w as Tensor, (u as Tensor).data as Float64Array, p['n'] as number)),
  multinomial: ([w, u], p) =>
    indices(multinomialResample(w as Tensor, (u as Tensor).data as Float64Array, p['n'] as number)),
  nextTemperature: ([l], p) =>
    nextTemperature(l as Tensor, p['temperature'] as number, { targetEss: p['targetEss'] as number }),
}

describe('sampling/smc against BlackJAX', () => {
  for (const c of loadFixture('sampling/smc').cases) {
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
