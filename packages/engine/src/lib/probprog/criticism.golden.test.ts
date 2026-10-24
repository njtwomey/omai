import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { compareModels, psisLoo, waic } from './index.ts'

describe('probprog/criticism against ArviZ', () => {
  for (const c of loadFixture('probprog/criticism').cases) {
    it(c.name, () => {
      const inputs = c.inputs.map((x) => decode(x) as Tensor)
      const ll = inputs[0] as Tensor
      switch (c.fn) {
        case 'waic': {
          const r = waic(ll)
          expectMatches(tensor([r.elpd, r.se, r.p]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'waicPointwise':
          expectMatches(tensor(Array.from(waic(ll).pointwise)), c.output, c.rtol, c.atol, c.name)
          return
        case 'loo': {
          const r = psisLoo(ll)
          expectMatches(tensor([r.elpd, r.se, r.p]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'looK':
          expectMatches(tensor(Array.from(psisLoo(ll).paretoK ?? [])), c.output, c.rtol, c.atol, c.name)
          return
        case 'compare': {
          const [a, b, m] = inputs as [Tensor, Tensor, Tensor]
          const rows = compareModels({ a: psisLoo(a), b: psisLoo(b), c: psisLoo(m) })
          const byName = (n: string) => rows.find((r) => r.name === n)
          const out = c.output as Record<string, unknown>
          const pick = (f: (r: NonNullable<ReturnType<typeof byName>>) => number): Tensor =>
            tensor(
              ['a', 'b', 'c'].map((n) => {
                const r = byName(n)
                if (r === undefined) throw new Error(`no row ${n}`)
                return f(r)
              }),
            )
          expectMatches(
            pick((r) => r.weight),
            out['weight'],
            c.rtol,
            c.atol,
            `${c.name}.weight`,
          )
          expectMatches(
            pick((r) => r.elpdDiff),
            out['elpdDiff'],
            c.rtol,
            c.atol,
            `${c.name}.elpdDiff`,
          )
          expectMatches(
            pick((r) => r.dse),
            out['dse'],
            c.rtol,
            c.atol,
            `${c.name}.dse`,
          )
          return
        }
        default:
          throw new Error(`no function ${c.fn}`)
      }
    })
  }
})
