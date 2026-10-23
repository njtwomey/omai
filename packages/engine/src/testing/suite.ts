// A golden-test suite over a fixture: one `it` per case, dispatching on the case's `fn`. Node only; not published.
import { describe, it } from 'vitest'
import type { Tensor, Value } from '../lib/tensor/index.ts'
import { decode, expectMatches, type GoldenCase, loadFixture } from './golden.ts'

/** Case parameters as plain JSON (tensors decoded). */
export type Params = Readonly<Record<string, unknown>>

/** A case's decoded inputs: at least two by type (absent ones are undefined at run time and fail loudly). */
export type Inputs = readonly [Tensor, Tensor, ...(Tensor | undefined)[]]

/** Compute a case's output from its decoded inputs and parameters. */
export type CaseFn = (inputs: Inputs, params: Params, c: GoldenCase) => Value | number

/** Decode a case's parameters: encoded tensors become tensors, everything else passes through. */
export function decodeParams(params: Params): Params {
  return Object.fromEntries(Object.entries(params).map(([k, v]) => [k, decode(v)]))
}

/**
 * Register one test per case of a fixture: `fns[c.fn]` computes the output, which must match within the case's
 * tolerances. Unknown function names fail their test.
 */
export function goldenSuite(title: string, fixture: string, fns: Readonly<Record<string, CaseFn>>): void {
  describe(title, () => {
    for (const c of loadFixture(fixture).cases) {
      it(c.name, () => {
        const f = fns[c.fn]
        if (f === undefined) throw new Error(`no function ${c.fn}`)
        const inputs = c.inputs.map((x) => decode(x) as Tensor) as unknown as Inputs
        expectMatches(f(inputs, decodeParams(c.params), c), c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}
