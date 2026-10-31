import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import { blocks, Conv1d, Conv2d, ConvTranspose1d, Linear, MaxPool1d, ReLU, Sequential } from './index.ts'
import {
  effectiveReceptiveField,
  layerSpecs,
  receptiveField,
  receptiveFieldOf,
  receptiveFieldSize,
} from './receptive-field.ts'

describe('theoretical receptive field', () => {
  it('reproduces the textbook arithmetic: a VGG-like stack and strided layers', () => {
    // 3x3, 3x3, 2x2 pool (stride 2), 3x3, 3x3: sizes 3, 5, 6, 10, 14; the jump doubles after the pool.
    const levels = receptiveField([
      { kernel: 3 },
      { kernel: 3 },
      { kernel: 2, stride: 2 },
      { kernel: 3 },
      { kernel: 3 },
    ])
    expect(levels.map((l) => l.size[0])).toEqual([1, 3, 5, 6, 10, 14])
    expect(levels.map((l) => l.jump[0])).toEqual([1, 1, 1, 2, 2, 2])
    // Without padding the first output's field is centred (size - 1) / 2 in from the left edge of the input.
    expect(levels.at(-1)?.start[0]).toBe(6.5)
  })

  it('padding moves the start to the left and "same" centres the first field on pixel zero', () => {
    const same = receptiveField([
      { kernel: 3, padding: 'same' },
      { kernel: 5, padding: 'same' },
    ])
    expect(same.at(-1)?.start[0]).toBe(0)
    expect(receptiveField([{ kernel: 3, padding: 1 }]).at(-1)?.start[0]).toBe(0)
  })

  it('dilation grows the field exponentially: a TCN of L temporal blocks has 1 + 2 (k - 1)(2^L - 1)', () => {
    const k = 3
    const layers = [0, 1, 2].flatMap((l) => [
      { kernel: k, dilation: 2 ** l },
      { kernel: k, dilation: 2 ** l },
    ])
    expect(receptiveFieldSize(layers)).toBe(new blocks.TCN(2, [4, 4, 4], { kernelSize: k, key: key(0) }).receptiveField)
    expect(receptiveFieldSize([1, 2, 4, 8].map((dilation) => ({ kernel: 2, dilation })))).toBe(16)
  })

  it('handles each axis separately', () => {
    const last = receptiveField([{ kernel: [3, 5], stride: [1, 2] }, { kernel: 3 }]).at(-1)
    expect(last?.size).toEqual([5, 9])
    expect(last?.jump).toEqual([1, 2])
  })

  it('reads the geometry from modules and refuses layers without a finite field', () => {
    const net = new Sequential(
      new Conv2d(1, 4, 3, { padding: 1, key: key(1) }),
      new ReLU(),
      new Conv2d(4, 4, 3, { dilation: 2, key: key(2) }),
    )
    expect(layerSpecs(net)).toHaveLength(2)
    expect(receptiveFieldOf(net).at(-1)?.size).toEqual([7, 7])
    expect(receptiveFieldOf(new Sequential(new Conv1d(1, 1, 3), new MaxPool1d(2))).at(-1)?.size).toEqual([4])
    expect(() => layerSpecs(new Linear(2, 2))).toThrow(/no finite receptive field/)
    expect(() => layerSpecs(new ConvTranspose1d(1, 1, 3))).toThrow(/not supported/)
  })

  it('matches the extent of the gradient of a random linear convolution stack', () => {
    const net = new Sequential(
      new Conv1d(1, 1, 3, { bias: false, key: key(3) }),
      new Conv1d(1, 1, 3, { bias: false, dilation: 2, key: key(4) }),
      new Conv1d(1, 1, 2, { bias: false, stride: 2, key: key(5) }),
      new Conv1d(1, 1, 3, { bias: false, key: key(6) }),
    )
    const length = 40
    const x = tensor([[Array.from({ length }, (_, i) => Math.sin(i))]])
    const spec = receptiveField(layerSpecs(net)).at(-1)
    for (const unit of [0, 2]) {
      const erf = toFlat(effectiveReceptiveField((v: Value) => net.apply(net.params, v), x, unit))
      const support = erf.map((v, i) => (v > 1e-14 ? i : -1)).filter((i) => i >= 0)
      expect(support.length).toBe(spec?.size[0])
      // The first input of the field is the unit's jump times its index.
      expect(support[0]).toBe(unit * (spec?.jump[0] as number))
    }
  })
})

describe('effective receptive field', () => {
  it('sums to one, and for a stack of averaging convolutions its mass sits in the centre of the theoretical field', () => {
    const layers = Array.from({ length: 10 }, (_, i) =>
      new Conv1d(1, 1, 3, { bias: false, padding: 1, key: key(i) }).loadStateDict({
        weight: tensor([[[1 / 3, 1 / 3, 1 / 3]]]),
      }),
    )
    const net = new Sequential(...layers)
    const length = 41
    const erf = toFlat(
      effectiveReceptiveField((v: Value) => net.apply(net.params, v), tensor([[Array.from({ length }, () => 0)]]), 20),
    )
    expect(erf.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(receptiveFieldSize(layerSpecs(net))).toBe(21)
    const central = erf.slice(15, 26).reduce((a, b) => a + b, 0)
    expect(central).toBeGreaterThan(0.85)
    expect(erf[20] as number).toBeGreaterThan(erf[12] as number)
  })
})
