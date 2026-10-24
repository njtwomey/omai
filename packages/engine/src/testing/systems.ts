// Test support for the systems golden tests: decoding complex parts and canonical orderings. Node only; not published.

import type { ComplexArray } from '../lib/numerics/polynomial/index.ts'
import { type Tensor, tensor, toFlat, type Value } from '../lib/tensor/index.ts'
import { decode, expectMatches, type GoldenCase, loadFixture } from './golden.ts'

/** A `[2, n]` tensor of parts as a complex array. */
export function complexOf(t: Tensor): ComplexArray {
  const flat = toFlat(t)
  const n = t.shape[1] ?? 0
  return { re: flat.slice(0, n), im: flat.slice(n, 2 * n) }
}

/** Complex values as `[2, n]` parts sorted by real then imaginary part (as the oracle's `sorted_parts`). */
export function sortedParts(z: ComplexArray): Tensor {
  const pairs = z.re.map((re, i) => [re, z.im[i] ?? 0] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (pairs.length === 0) return tensor([[], []]) as Tensor
  return tensor([pairs.map((p) => p[0]), pairs.map((p) => p[1])]) as Tensor
}

/** What a golden case evaluates to. */
export type GoldenFn = (inputs: Tensor[], params: Readonly<Record<string, unknown>>, c: GoldenCase) => Value | number

/** Run every case of a fixture through a table of functions. */
export function runFixture(
  path: string,
  fns: Readonly<Record<string, GoldenFn>>,
  it: (name: string, f: () => void) => void,
): void {
  for (const c of loadFixture(path).cases) {
    it(c.name, () => {
      const f = fns[c.fn]
      if (f === undefined) throw new Error(`no function ${c.fn}`)
      const inputs = c.inputs.map((x) => decode(x) as Tensor)
      expectMatches(f(inputs, c.params, c), c.output, c.rtol, c.atol, c.name)
    })
  }
}
