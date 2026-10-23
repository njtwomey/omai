import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { sliceAxis, toArray, toFlat, type Value } from '../../tensor/index.ts'
import * as NN from '../index.ts'
import {
  gruCell,
  lengthsFromMask,
  lstmCell,
  packedLengths,
  packPaddedSequence,
  padPackedSequence,
  recurrent,
  rnnCell,
  scan,
} from './recurrent.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('recurrent gradients', () => {
  it('cells pass the conformance check', () => {
    ok(
      (x, h, wi, wh, b) => rnnCell(x, h, { weightIh: wi, weightHh: wh, biasIh: b }),
      [n(0, [2, 3]), n(1, [2, 2]), n(2, [2, 3]), n(3, [2, 2]), n(4, [2])],
    )
    ok(
      (x, h, c, wi, wh) => lstmCell(x, [h, c], { weightIh: wi, weightHh: wh })[0],
      [n(5, [1, 2]), n(6, [1, 2]), n(7, [1, 2]), n(8, [8, 2]), n(9, [8, 2])],
    )
    ok(
      (x, h, wi, wh, bh) => gruCell(x, h, { weightIh: wi, weightHh: wh, biasHh: bh }),
      [n(10, [1, 2]), n(11, [1, 2]), n(12, [6, 2]), n(13, [6, 2]), n(14, [6])],
    )
  })

  it('a bidirectional GRU over padded sequences passes the conformance check', () => {
    const p = { weightIh: n(15, [6, 2]), weightHh: n(16, [6, 2]) }
    ok((x) => recurrent(x, [[p, p]], { mode: 'GRU', bidirectional: true, lengths: [3, 1] }).output, [n(17, [3, 2, 2])])
  })
})

describe('recurrent behaviour', () => {
  it('scans forwards and backwards', () => {
    const xs = [[1], [2], [3]]
    const back = scan((c: Value, x: Value) => [x, c] as const, n(0, [1]), xs, { reverse: true })
    expect(toArray(back.carry)).toEqual([1])
    expect(toFlat(sliceAxis(back.ys, 0, 0, 2))).toEqual([2, 3])
  })

  it('steps past a sequence length leave the state alone and output zeros', () => {
    const p = { weightIh: n(1, [3, 2]), weightHh: n(2, [3, 3]) }
    const x = n(3, [4, 1, 2])
    const short = recurrent(x, [[p]], { mode: 'RNN_TANH', lengths: [2] })
    const cut = recurrent(sliceAxis(x, 0, 0, 2), [[p]], { mode: 'RNN_TANH' })
    expect(toFlat(short.h)).toEqual(toFlat(cut.h))
    expect(toFlat(sliceAxis(short.output, 0, 2, 4)).every((v) => v === 0)).toBe(true)
    expect(() => recurrent(x, [[p]], { mode: 'RNN_TANH', lengths: [9] })).toThrow()
    expect(() => recurrent(x, [[p, p]], { mode: 'RNN_TANH' })).toThrow()
  })

  it('packs and unpacks variable-length batches', () => {
    const x = n(4, [4, 3, 2])
    const packed = packPaddedSequence(x, [2, 4, 1])
    expect(packed.batchSizes).toEqual([3, 2, 1, 1])
    expect(packed.sortedIndices).toEqual([1, 0, 2])
    expect(packedLengths(packed)).toEqual([2, 4, 1])
    const { padded, lengths } = padPackedSequence(packed, { batchFirst: true, totalLength: 5 })
    expect(padded.shape).toEqual([3, 5, 2])
    expect(lengths).toEqual([2, 4, 1])
    expect(() => packPaddedSequence(x, [2, 4, 1], { enforceSorted: true })).toThrow()
    expect(
      lengthsFromMask([
        [true, true, false],
        [true, false, false],
      ]),
    ).toEqual([2, 1])
  })

  it('modules: dropout between layers needs a key, unbatched inputs work', () => {
    const lstm = new NN.LSTM(2, 3, { numLayers: 2, dropout: 0.5, key: key(0) })
    expect(() => lstm.forward(n(5, [4, 1, 2]))).toThrow()
    expect(lstm.forward(n(5, [4, 1, 2]), { key: key(1) }).output).toBeDefined()
    const gru = new NN.GRU(2, 3, { key: key(2) }).eval()
    const r = gru.forward(n(6, [4, 2]))
    expect((r.output as Value).shape).toEqual([4, 3])
    expect(r.h.shape).toEqual([1, 3])
    expect(new NN.RNN(2, 3, { bidirectional: true }).outputShape([4, 1, 2])).toEqual([4, 1, 6])
    expect(() => new NN.RNNBase('GRU', 2, 3, { projSize: 1 })).toThrow()
    expect(new NN.LSTMCell(2, 3).forward(n(7, [2]))[0].shape).toEqual([3])
  })
})
