import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { concat, sliceAxis, toFlat, type Value } from '../../tensor/index.ts'
import { causalPad, StackedRecurrent, TCN, TemporalBlock, WaveNet, WaveNetBlock } from './sequence.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

/** The output up to time t depends only on inputs up to t: changing the future leaves the past alone. */
const causal = (f: (x: Value) => Value, x: Value, t: number): void => {
  const future = concat([sliceAxis(x, 2, 0, t), n(99, [x.shape[0] ?? 1, x.shape[1] ?? 1, (x.shape[2] ?? 0) - t])], 2)
  const a = toFlat(sliceAxis(f(x), 2, 0, t))
  const b = toFlat(sliceAxis(f(future), 2, 0, t))
  for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? Number.NaN, 12)
}

describe('sequence blocks', () => {
  it('temporal blocks and TCNs are causal and keep the length', () => {
    const tb = new TemporalBlock(2, 3, { dilation: 2, key: key(0) }).eval()
    const x = n(1, [1, 2, 9])
    expect(tb.forward(x).shape).toEqual([1, 3, 9])
    causal((v) => tb.forward(v), x, 5)
    const tcn = new TCN(2, [3, 3], { kernelSize: 2, weightNorm: false, key: key(2) }).eval()
    expect(tcn.receptiveField).toBe(7)
    causal((v) => tcn.forward(v), x, 4)
    expect(Object.keys(tb.stateDict())).toContain('conv1.weight_g')
  })

  it('WaveNet blocks and stacks are causal, with conditioning', () => {
    const b = new WaveNetBlock(3, 4, { dilation: 2, conditionChannels: 2, key: key(3) })
    const r = b.forward({ x: n(4, [1, 3, 6]), condition: n(5, [1, 2, 6]) })
    expect(r.output.shape).toEqual([1, 3, 6])
    expect(r.skip.shape).toEqual([1, 4, 6])
    const w = new WaveNet(2, 5, { residualChannels: 4, skipChannels: 6, layersPerStack: 3, stacks: 1, key: key(6) })
    causal((v) => w.forward(v), n(7, [1, 2, 10]), 6)
    expect(toFlat(causalPad([[[1, 2, 3]]], 3))).toEqual([0, 0, 1, 2, 3])
  })

  it('stacked recurrent layers with residuals, norms and dropout', () => {
    const s = new StackedRecurrent(3, 3, 2, { key: key(8) })
    expect(() => s.train().forward(n(9, [2, 4, 3]))).not.toThrow()
    const g = new StackedRecurrent(3, 2, 2, { cell: 'gru', bidirectional: true, dropout: 0.5, key: key(10) })
    expect(() => g.train().forward(n(11, [1, 4, 3]))).toThrow()
    expect(g.eval().forward(n(11, [1, 4, 3])).shape).toEqual([1, 4, 4])
  })

  it('pass the gradient conformance check', () => {
    const tb = new TemporalBlock(1, 2, { kernelSize: 2, key: key(12) })
    expect(
      checkPrimitive((x) => tb.apply(tb.params, x, { train: false }), [n(13, [1, 1, 4])], { rtol: 1e-5 }).failures,
    ).toEqual([])
    const b = new WaveNetBlock(2, 2, { key: key(14) })
    expect(checkPrimitive((x) => b.apply(b.params, x).output, [n(15, [1, 2, 3])], { rtol: 1e-5 }).failures).toEqual([])
  })
})
