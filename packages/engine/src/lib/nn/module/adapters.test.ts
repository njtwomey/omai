import { describe, expect, it } from 'vitest'
import { xy } from '../../data/index.ts'
import { mse } from '../../losses/index.ts'
import { sgd } from '../../optim/index.ts'
import { key, normal } from '../../random/index.ts'
import { add, matmul, mul, type Tensor, toFlat, transpose, type Value } from '../../tensor/index.ts'
import * as NN from '../index.ts'

const n = (seed: number, shape: number[], std = 1): Tensor => mul(normal(key(seed), shape), std) as Tensor
const flat = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : toFlat(v as Value))
const maxDiff = (a: unknown, b: unknown): number =>
  Math.max(...flat(a).map((v, i) => Math.abs(v - (flat(b)[i] as number))))

/** LoRA params with a nonzero B for the named target. */
function withRandomB(lora: NN.LoRA<Value, Value>, seed: number): NN.Params {
  const l = lora.params['lora'] as Record<string, { A: Value; B: Value }>
  const next: Record<string, unknown> = {}
  Object.entries(l).forEach(([name, f], i) => {
    next[name] = { A: f.A, B: n(seed + i, [...f.B.shape], 0.3) }
  })
  return { ...lora.params, lora: next }
}

describe('LoRA on a Linear layer', () => {
  const base = new NN.Linear(6, 4, { key: key(0) })
  const lora = new NN.LoRA(base, { rank: 2, alpha: 4, key: key(1) })
  const x = n(2, [5, 6])

  it('equals the base while B is zero', () => {
    expect(maxDiff(lora.forward(x), base.apply(base.params, x))).toBe(0)
  })

  it('computes x W^T + b + (alpha / r) x A^T B^T, and merge reproduces it to 1e-12', () => {
    const params = withRandomB(lora, 10)
    const f = (params['lora'] as Record<string, { A: Value; B: Value }>)['weight'] as { A: Value; B: Value }
    const w = base.params['weight'] as Value
    const b = base.params['bias'] as Value
    const explicit = add(add(matmul(x, transpose(w)), b), mul(matmul(matmul(x, transpose(f.A)), transpose(f.B)), 2))
    const adapted = lora.apply(params, x)
    expect(maxDiff(adapted, explicit)).toBeLessThan(1e-12)
    expect(maxDiff(base.apply(lora.merged(params), x), adapted)).toBeLessThan(1e-12)
    // The update changed the output, so the comparison is not vacuous.
    expect(maxDiff(adapted, base.apply(base.params, x))).toBeGreaterThan(1e-3)
  })

  it('unmerge returns the base weights', () => {
    const params = withRandomB(lora, 20)
    const back = lora.unmerge(lora.merged(params), params)
    expect(maxDiff(back['weight'], base.params['weight'])).toBeLessThan(1e-14)
    expect(maxDiff(back['bias'], base.params['bias'])).toBe(0)
  })

  it('counts r (d_in + d_out) trainable parameters and freezes the base', () => {
    expect(lora.loraParameterCount()).toBe(2 * (6 + 4))
    const c = NN.parameterCounts(lora)
    expect(c.trainable).toBe(20)
    expect(c.total).toBe(20 + 6 * 4 + 4)
    expect(c.frozen).toBe(28)
    expect(c.trainableShare).toBeCloseTo(20 / 48, 15)
    expect(NN.loraParameterCount(768, 768, 8)).toBe(12288)
  })

  it('trains only the adapter: the base weights do not move and the loss falls', () => {
    const model = new NN.LoRA(new NN.Linear(3, 2, { key: key(3) }), { rank: 2, key: key(4) })
    const before = flat((model.params['base'] as NN.Params)['weight'])
    const xs = n(5, [30, 3])
    const target = matmul(xs, n(6, [3, 2]))
    const r = NN.fit(model, xy(xs, target), {
      loss: (p, t) => mse(p, t),
      optimizer: sgd({ learningRate: 0.05 }),
      epochs: 200,
    })
    expect(flat((r.params['base'] as NN.Params)['weight'])).toEqual(before)
    expect(r.losses.at(-1) as number).toBeLessThan((r.losses[0] as number) / 2)
    const l = r.params['lora'] as Record<string, { A: Value; B: Value }>
    expect(Math.max(...flat((l['weight'] as { B: Value }).B).map(Math.abs))).toBeGreaterThan(1e-3)
  })

  it('rank-stabilised scaling uses alpha / sqrt(r)', () => {
    expect(new NN.LoRA(base, { rank: 4, alpha: 8, rankStabilised: true }).scaling).toBe(4)
    expect(new NN.LoRA(base, { rank: 4, alpha: 8 }).scaling).toBe(2)
  })
})

describe('LoRA on attention', () => {
  const attn = new NN.MultiheadAttention(8, 2, { batchFirst: true, key: key(0) })
  const x = n(1, [2, 5, 8])

  it('adapts the query and value rows of the packed projection only', () => {
    const lora = new NN.LoRA(attn, { rank: 2, key: key(2) })
    expect(lora.targets.map((t) => t.name)).toEqual(['q', 'v'])
    expect(lora.loraParameterCount()).toBe(2 * 2 * (8 + 8))
    expect(maxDiff(lora.forward(x).output, attn.apply(attn.params, x).output)).toBe(0)
    const params = withRandomB(lora as unknown as NN.LoRA<Value, Value>, 30)
    const merged = lora.merged(params)
    const w0 = flat(attn.params['in_proj_weight'])
    const w1 = flat(merged['in_proj_weight'])
    const e = 8 * 8
    // Rows [0, 8) (q) and [16, 24) (v) changed; the key rows [8, 16) did not.
    expect(maxDiff(w0.slice(0, e), w1.slice(0, e))).toBeGreaterThan(1e-3)
    expect(w0.slice(e, 2 * e)).toEqual(w1.slice(e, 2 * e))
    expect(maxDiff(w0.slice(2 * e), w1.slice(2 * e))).toBeGreaterThan(1e-3)
    expect(maxDiff(lora.apply(params, x).output, attn.apply(merged, x).output)).toBe(0)
    const back = lora.unmerge(merged, params)
    expect(maxDiff(back['in_proj_weight'], attn.params['in_proj_weight'])).toBeLessThan(1e-14)
  })

  it('can adapt the output projection too', () => {
    const lora = new NN.LoRA(attn, { rank: 1, targets: NN.loraTargets(attn, ['q', 'k', 'v', 'out']) })
    expect(lora.targets).toHaveLength(4)
    expect(NN.parameterCounts(lora).trainable).toBe(4 * 16)
  })
})

describe('bottleneck adapter, prompt tuning, prefix tuning and BitFit', () => {
  it('a bottleneck adapter starts as the identity and counts its parameters', () => {
    const ad = new NN.BottleneckAdapter(8, 2, { key: key(0) })
    const x = n(1, [3, 8])
    expect(maxDiff(ad.forward(x), x)).toBe(0)
    expect(ad.parameterCount()).toBe(8 * 2 + 2 + 2 * 8 + 8)
    const wrapped = NN.withAdapter(new NN.Linear(4, 8, { key: key(2) }), ad)
    expect(NN.parameterCounts(wrapped).trainable).toBe(ad.parameterCount())
    expect(wrapped.forward(n(3, [5, 4])).shape).toEqual([5, 8])
  })

  it('an adapter trains while its base stays put', () => {
    const base = new NN.Linear(2, 4, { key: key(0) })
    const model = NN.withAdapter(base, new NN.BottleneckAdapter(4, 3, { key: key(1) }))
    const before = flat(base.params['weight'])
    const xs = n(2, [40, 2])
    const target = n(3, [40, 4])
    const r = NN.fit(model, xy(xs, target), {
      loss: (p, t) => mse(p, t),
      optimizer: sgd({ learningRate: 0.05 }),
      epochs: 100,
    })
    expect(flat((r.params['base'] as NN.Params)['weight'])).toEqual(before)
    expect(r.losses.at(-1) as number).toBeLessThan(r.losses[0] as number)
  })

  it('prompt tuning prepends the prompt, and only the prompt trains', () => {
    const pt = new NN.PromptTuning(new NN.Sequential(), 3, 4, { key: key(0) })
    const x = n(1, [2, 5, 4])
    const y = pt.forward(x)
    expect(y.shape).toEqual([2, 8, 4])
    expect(NN.parameterCounts(pt)).toEqual({ total: 12, trainable: 12, frozen: 0, trainableShare: 1 })
    expect(flat(y).slice(0, 12)).toEqual(flat(pt.params['prompt']))
    // The original tokens follow the prompt unchanged.
    expect(flat(y).slice(12, 12 + 20)).toEqual(flat(x).slice(0, 20))
    expect(pt.outputShape([2, 5, 4])).toEqual([2, 8, 4])
  })

  it('prefix tuning lets every query attend to the prefix and trains only it', () => {
    const attn = new NN.MultiheadAttention(8, 2, { batchFirst: true, key: key(0) })
    const pa = new NN.PrefixAttention(attn, 3, { key: key(1) })
    const x = n(2, [2, 5, 8])
    const out = pa.forward(x)
    expect(out.output.shape).toEqual([2, 5, 8])
    expect(out.weights?.shape).toEqual([2, 5, 8])
    // Attention weights over the 3 prefix + 5 sequence positions sum to one.
    const w = flat(out.weights)
    for (let r = 0; r < 10; r++) expect(w.slice(r * 8, r * 8 + 8).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(NN.parameterCounts(pa).trainable).toBe(2 * 3 * 8)
    // A causal mask still shows the prefix: the first query attends to prefix positions and itself only.
    const causal = pa.forward({ query: x, isCausal: true })
    const first = flat(causal.weights).slice(0, 8)
    expect(first.slice(0, 4).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(first.slice(4).every((v) => v === 0)).toBe(true)
  })

  it('BitFit trains the biases only', () => {
    const net = NN.mlp([4, 8, 2], { key: key(0) })
    NN.bitFit(net)
    const c = NN.parameterCounts(net)
    expect(c.trainable).toBe(8 + 2)
    expect(c.total).toBe(4 * 8 + 8 + 8 * 2 + 2)
    const xs = n(1, [30, 4])
    const target = n(2, [30, 2])
    const w0 = flat((net.params['0'] as NN.Params)['weight'])
    const b0 = flat((net.params['0'] as NN.Params)['bias'])
    const r = NN.fit(net, xy(xs, target), {
      loss: (p, t) => mse(p, t),
      optimizer: sgd({ learningRate: 0.05 }),
      epochs: 20,
    })
    expect(flat((r.params['0'] as NN.Params)['weight'])).toEqual(w0)
    expect(flat((r.params['0'] as NN.Params)['bias'])).not.toEqual(b0)
  })

  it('trainOnly selects by path', () => {
    const net = NN.mlp([2, 3, 1], { key: key(0) })
    expect(NN.parameterCounts(NN.trainOnly(net, (p) => p.startsWith('2.'))).trainable).toBe(3 + 1)
  })
})
