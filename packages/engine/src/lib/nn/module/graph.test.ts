import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { take, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { tapContext } from '../../trace/index.ts'
import { gatConv, gcnConv, ginConv, sageConv } from '../functional/graph.ts'
import { GATConv, GCNConv, GINConv, SAGEConv } from './graph.ts'
import { Linear } from './layers.ts'
import { InducedSetAttentionBlock, PoolingByMultiheadAttention, SetAttentionBlock } from './sets.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const EI = [
  [0, 1, 2, 3, 0],
  [1, 2, 0, 0, 3],
]
const x = n(0, [4, 3])

describe('graph modules', () => {
  it('apply their functional forms with PyTorch Geometric parameter names', () => {
    const gcn = new GCNConv(3, 2, { key: key(1) })
    expect(Object.keys(gcn.stateDict())).toEqual(['bias', 'lin.weight'])
    expect(toFlat(gcn.forward({ x, edgeIndex: EI }))).toEqual(
      toFlat(
        gcnConv(x, EI, (gcn.params['lin'] as Record<string, Value>)['weight'] as Value, {
          bias: gcn.params['bias'] as Value,
        }),
      ),
    )
    const gat = new GATConv(3, 2, { heads: 2, key: key(2) })
    expect(Object.keys(gat.stateDict())).toEqual(['att_dst', 'att_src', 'bias', 'lin.weight'])
    const taps = tapContext()
    const y = gat.forward({ x, edgeIndex: EI }, { taps, path: 'gat' })
    expect(y.shape).toEqual(gat.outputShape(x.shape))
    expect(taps.entries.get('gat.attention')?.value.shape).toEqual([9, 2])
    const p = gat.params as Record<string, Value>
    expect(toFlat(y)).toEqual(
      toFlat(
        gatConv(
          x,
          EI,
          {
            weight: (p['lin'] as unknown as Record<string, Value>)['weight'] as Value,
            attSrc: p['att_src'] as Value,
            attDst: p['att_dst'] as Value,
            bias: p['bias'] as Value,
          },
          { heads: 2 },
        ).output,
      ),
    )
    expect(new GATConv(3, 2, { heads: 3, concat: false }).forward({ x, edgeIndex: EI }).shape).toEqual([4, 2])
    const sage = new SAGEConv(3, 2, { aggregate: 'max', key: key(3) })
    expect(sage.forward({ x, edgeIndex: EI }).shape).toEqual([4, 2])
    expect(Object.keys(new SAGEConv(3, 2, { rootWeight: false, bias: false }).stateDict())).toEqual(['lin_l.weight'])
    const lin = new Linear(3, 2, { key: key(4) })
    const gin = new GINConv(lin, { eps: 0.5, trainEps: true })
    expect(toFlat(gin.forward({ x, edgeIndex: EI }))).toEqual(toFlat(ginConv(x, EI, (h) => lin.forward(h), 0.5)))
    expect(gin.outputShape([4, 3])).toEqual([4, 2])
  })

  it('pass the gradient conformance check in their parameters', () => {
    const gcn = new GCNConv(3, 2, { key: key(5) })
    const w = (gcn.params['lin'] as Record<string, Value>)['weight'] as Value
    expect(
      checkPrimitive(
        (ww) =>
          gcn.apply(
            { lin: { weight: ww }, bias: gcn.params['bias'] },
            { x, edgeIndex: EI, edgeWeight: [1, 2, 1, 0.5, 1] },
          ),
        [w],
      ).failures,
    ).toEqual([])
    const sage = new SAGEConv(3, 2, { key: key(6) })
    const l = sage.params['lin_l'] as Record<string, Value>
    expect(
      checkPrimitive(
        (ww) => sageConv(x, EI, { weightNeighbour: ww, biasNeighbour: l['bias'] as Value }),
        [l['weight'] as Value],
      ).failures,
    ).toEqual([])
  })
})

describe('set modules', () => {
  const s = n(7, [2, 5, 4])
  it('keep their shapes, mask padding and are equivariant or invariant', () => {
    const sabm = new SetAttentionBlock(4, 2, { key: key(8) })
    const order = [2, 4, 0, 1, 3]
    const permuted = take(s, order, 1)
    expect(toFlat(take(sabm.forward(s), order, 1)).map((v) => Math.round(v * 1e9))).toEqual(
      toFlat(sabm.forward(permuted)).map((v) => Math.round(v * 1e9)),
    )
    const isabm = new InducedSetAttentionBlock(4, 2, 3, { key: key(9) })
    expect(isabm.forward(s).shape).toEqual([2, 5, 4])
    const pmam = new PoolingByMultiheadAttention(4, 1, 2, { key: key(10), layerNorm: false })
    expect(pmam.forward(s).shape).toEqual(pmam.outputShape(s.shape))
    expect(toFlat(pmam.forward(permuted)).map((v) => Math.round(v * 1e9))).toEqual(
      toFlat(pmam.forward(s)).map((v) => Math.round(v * 1e9)),
    )
    const mask = tensor(
      [
        [true, true, false, false, false],
        [true, true, true, true, true],
      ],
      { dtype: 'bool' },
    )
    expect(sabm.forward({ x: s, mask }).shape).toEqual([2, 5, 4])
    expect(Object.keys(sabm.stateDict())).toContain('mab.attn.in_proj_weight')
    expect(() => new SetAttentionBlock(5, 2)).toThrow()
  })
})
