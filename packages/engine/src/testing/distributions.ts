// Golden tests of distributions against scipy.stats / torch.distributions fixtures (oracle/whitebox_oracle/distributions.py).
// Node only; not published.
import { describe, it } from 'vitest'
import type { Distribution } from '../lib/dists/index.ts'
import type { Tensor, Value } from '../lib/tensor/index.ts'
import { decode, expectMatches, type GoldenCase, loadFixture } from './golden.ts'

/** Build a distribution from the parameter tensors of a case. */
export type Family = (params: readonly Tensor[]) => Distribution

/** A case the standard dispatch does not cover: compute the output from the decoded inputs. */
export type Custom = (inputs: readonly Tensor[], c: GoldenCase) => Value

const NO_ARGUMENT = new Set(['mean', 'variance', 'stddev', 'entropy', 'mode'])

/**
 * One test per fixture case. A case named `<family>.<method>` has the family's parameters as its first `arity` inputs
 * (`params.arity`) and the method's argument (if any) after them; other functions are looked up in `custom`.
 */
export function describeDistributions(
  fixture: string,
  families: Readonly<Record<string, Family>>,
  custom: Readonly<Record<string, Custom>> = {},
): void {
  describe(`${fixture} against the oracle`, () => {
    for (const c of loadFixture(fixture).cases) {
      it(c.name, () => {
        const inputs = c.inputs.map((x) => decode(x) as Tensor)
        const special = custom[c.fn]
        let got: Value
        if (special) got = special(inputs, c)
        else {
          const [family, method] = c.fn.split('.') as [string, string]
          const make = families[family]
          if (make === undefined) throw new Error(`no family ${family} for case ${c.name}`)
          const arity = typeof c.params['arity'] === 'number' ? c.params['arity'] : inputs.length
          const dist = make(inputs.slice(0, arity))
          const call = dist[method as keyof Distribution] as unknown as (x?: Tensor) => Value
          got = NO_ARGUMENT.has(method) ? call() : call(inputs[arity])
        }
        expectMatches(got, c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}
