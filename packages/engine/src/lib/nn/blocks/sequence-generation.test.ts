import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { sliceAxis, toFlat, type Value } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { mambaModel } from './mamba.ts'
import { sequenceMixers } from './mixers.ts'
import { generationAlgorithm, sequenceLanguageModel } from './sequence-generation.ts'
import { SequenceModel } from './sequence-model.ts'

const logSoftmax = (row: number[]): number[] => {
  const m = Math.max(...row)
  const z = Math.log(row.reduce((s, v) => s + Math.exp(v - m), 0)) + m
  return row.map((v) => v - z)
}

describe('generation as an algorithm', () => {
  for (const name of ['mamba', 'mamba2', 's4d', 'retnet', 'rwkv', 'attention'] as const) {
    it(`${name}: one token per step reproduces the parallel pass (tokens, log-probabilities)`, () => {
      const model = new SequenceModel(8, 2, (d, k) => sequenceMixers.make({ kind: name, stateSize: 4 })(d, k), {
        inputSize: 9,
        key: key(1),
      })
      const prompt = [3, 1, 4]
      const result = runAlgorithm(generationAlgorithm(model, undefined, prompt, { maxNewTokens: 6, eosId: null }), {
        maxSteps: 6,
        key: key(2),
      })
      const tokens = [...result.state.tokens]
      expect(tokens).toHaveLength(9)
      // The parallel pass over everything generated: its greedy choice at each position is the token generated next,
      // and the log-probabilities add up to the algorithm's.
      const logits = model.forward([tokens.slice(0, -1)]) // [1, 8, 9]
      let total = 0
      for (let t = prompt.length - 1; t < tokens.length - 1; t++) {
        const row = toFlat(sliceAxis(logits, 1, t, t + 1) as Value)
        const lp = logSoftmax(row)
        const best = lp.indexOf(Math.max(...lp))
        expect(best).toBe(tokens[t + 1])
        total += lp[best] ?? 0
      }
      expect(result.state.logprob).toBeCloseTo(total, 9)
    })
  }

  it('the logits of the incremental interface equal the parallel logits, and the state serialises', () => {
    const model = mambaModel(7, 8, 1, { stateSize: 4, key: key(3) })
    const lm = sequenceLanguageModel(model)
    const whole = toFlat(lm.logits([[1, 2, 3, 4]]))
    const head = lm.step?.([[1, 2]], lm.initCache?.(1) as never)
    const tail = lm.step?.([[3, 4]], head?.cache as never)
    const joined = [...toFlat(head?.logits as Value), ...toFlat(tail?.logits as Value)]
    for (const [i, v] of whole.entries()) expect(joined[i]).toBeCloseTo(v, 9)
    const alg = generationAlgorithm(model, undefined, [1, 2], { maxNewTokens: 3, eosId: null })
    const run = runAlgorithm(alg, { maxSteps: 2, key: key(4) })
    const back = alg.deserialise?.(alg.serialise?.(run.state) as never)
    const resumed = runAlgorithm(alg, { maxSteps: 1, key: key(4), start: { state: back, step: 2 } })
    const straight = runAlgorithm(alg, { maxSteps: 3, key: key(4) })
    expect(resumed.state.generated).toEqual(straight.state.generated)
  })
})
