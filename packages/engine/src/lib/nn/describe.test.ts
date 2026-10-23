import { expect, describe as group, it } from 'vitest'
import { key } from '../random/index.ts'
import { add, mul, tensor, type Value } from '../tensor/index.ts'
import * as NN from './index.ts'

group('describe', () => {
  it('residual blocks get a skip edge into an add node, parallel branches a join', () => {
    const net = new NN.Sequential(
      new NN.Linear(3, 3, { key: key(0) }),
      new NN.Residual(new NN.Sequential(new NN.Linear(3, 3, { key: key(1) }), new NN.ReLU())),
      new NN.Parallel([new NN.Linear(3, 2, { key: key(2) }), new NN.Linear(3, 4, { key: key(3) })], 'concat'),
    )
    const s = NN.describe(net, [5, 3])
    const skip = s.edges.filter((e) => e.kind === 'skip')
    expect(skip).toEqual([{ from: 'layer:0', to: 'op:1', label: '[5, 3]', kind: 'skip' }])
    expect(s.nodes.find((n) => n.id === 'op:2')?.label).toBe('concat')
    expect(s.edges.filter((e) => e.to === 'op:2').map((e) => e.from)).toEqual(['layer:2.0', 'layer:2.1'])
    expect(s.nodes.at(-1)?.badges).toEqual({ shape: '[5, 6]' })
    expect(s.groups?.map((g) => g.id)).toEqual(['group:1', 'group:1.body', 'group:2'])
  })

  it('composite blocks list their parts and repeats', () => {
    const stack = new NN.blocks.TransformerStack(3, (i) => new NN.blocks.TransformerBlock(8, 2, { key: key(10 + i) }))
    const row = NN.layerTable(stack, [1, 4, 8])[0]
    expect(row?.repeat).toBe(3)
    expect(row?.parts.map((p) => p.name)).toEqual(['blocks'])
    expect(row?.flops).toBeGreaterThan(0)
    const s = NN.describe(stack, [1, 4, 8], { expandParts: true })
    expect(s.groups?.find((g) => g.kind === 'repeat')?.count).toBe(3)
    // Structured inputs cannot be built from a shape: no FLOP estimate, but a row.
    expect(NN.layerTable(new NN.Bilinear(2, 2, 1, { key: key(4) }), [3, 2])[0]?.flops).toBeUndefined()
  })

  it('counts FLOPs as multiply-adds times two and estimates memory', () => {
    // A 3x3 convolution with bias to 2 channels on 1 x 1 x 4 x 4: 2 * 9 * 2 * 4 (matmul) + 8 (bias).
    expect(NN.flops(new NN.Conv2d(1, 2, 3, { key: key(5) }), [1, 1, 4, 4])).toBe(2 * 9 * 2 * 4 + 8)
    expect(NN.functionFlops((x: Value) => add(mul(x, 2), 1), tensor([1, 2, 3]))).toBe(6)
    const m = NN.memoryEstimate(new NN.Sequential(new NN.Linear(4, 3), new NN.BatchNorm1d(3)), [2, 4])
    expect(m).toEqual({ params: 21 * 8, buffers: 7 * 8, activations: 12 * 8, input: 64, total: 168 + 56 + 96 + 64 })
  })

  it('summaries report totals, frozen parameters and sizes', () => {
    const net = NN.mlp([3, 8, 2], { key: key(0) })
    ;(net.layers[0] as NN.Module).requiresGrad(false)
    const text = NN.summary(net, [16, 3])
    expect(text).toContain('Total params: 50')
    expect(text).toContain('Trainable params: 18')
    expect(text).toContain('Non-trainable params: 32')
    expect(text).toContain('FLOPs (forward)')
    expect(text.split('\n')[0]).toBe('Layer       Output shape  Params  FLOPs')
  })

  it('functional models describe their computation graphs', () => {
    const g = NN.describeFunction(
      (p: { w: Value; b: Value }, x: Value) => NN.linear(x, p.w, p.b),
      { w: NN.zerosInit([2, 3]), b: NN.zerosInit([2]) },
      NN.zerosInit([1, 3]),
    )
    expect(g.type).toBe('computation-graph')
    expect(g.nodes.map((n) => n.label)).toContain('matmul')
  })
})
