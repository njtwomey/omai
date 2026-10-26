import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { modelRows } from '../../testing/rl.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { blackjackStep, blackjackSum, makeTaxi } from './toy-text.ts'

describe('toy text against gymnasium', () => {
  for (const c of loadFixture('methods/rl/environments/toy_text').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'blackjack') {
        const cards = [...(p['cards'] as number[])]
        const r = blackjackStep(
          { player: p['player'] as number[], dealer: p['dealer'] as number[] },
          p['action'] as number,
          () => cards.shift() ?? 0,
          { natural: p['natural'] === true, sab: p['sab'] === true },
        )
        const usable = r.state.player.includes(1) && r.state.player.reduce((a, b) => a + b, 0) + 10 <= 21
        const got = [
          blackjackSum(r.state.player),
          r.state.dealer[0] ?? 0,
          usable ? 1 : 0,
          r.reward,
          r.terminated ? 1 : 0,
        ]
        expectMatches(tensor(got), c.output, 0, 0, c.name)
        return
      }
      const env = makeTaxi({
        isRainy: p['is_rainy'] === true,
        ...(typeof p['rainy_probability'] === 'number' ? { rainyProbability: p['rainy_probability'] } : {}),
      })
      if (c.fn === 'taxiTable') {
        const reference = toArray(decode(c.output) as Tensor) as number[][]
        const rows = modelRows(env, reference)
        expect(rows.length).toBe(reference.length)
        expectMatches(tensor(rows), c.output, 1e-15, 1e-15, c.name)
      } else {
        const initial = new Array<number>(500).fill(0)
        for (const { probability, state } of env.model?.initial ?? []) initial[state] = probability
        expectMatches(tensor(initial), c.output, 1e-15, 1e-15, c.name)
      }
    })
  }
})
