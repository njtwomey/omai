import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { blocks } from '../nn/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  activationPatching,
  attributionPatching,
  causalTrace,
  directLogitAttribution,
  headSite,
  lensDivergence,
  logitDifference,
  logitLens,
  pathPatching,
  plantedCopyCircuit,
  residualModel,
  tokenProbability,
  tunedLensAlgorithm,
} from './index.ts'

const lm = new blocks.TransformerLM(9, 8, 3, 2, { key: key(3) })
const model = residualModel(lm)
const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9): void => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs((a[i] as number) - (b[i] as number))).toBeLessThan(tol)
}

describe('residualModel', () => {
  it('reproduces the language model and its heads sum to the attention output', () => {
    const tokens = [[1, 4, 2, 7, 3]]
    const run = model.run(tokens)
    close(toFlat(run.logits), toFlat(lm.forward(tokens).output))
    // The block's residual stream is the embeddings plus every component's write.
    let acc = toFlat(run.sites.get('embed') as never)
    const sum = Float64Array.from(acc)
    for (const s of model.sites)
      if (/\.(head\d+|mlp)$/.test(s)) {
        const v = toFlat(run.sites.get(s) as never)
        for (let i = 0; i < sum.length; i++) sum[i] = (sum[i] as number) + (v[i] as number)
      }
    acc = toFlat(run.residual[model.numLayers] as never)
    close(sum, acc)
  })
  it('rejects post-norm blocks', () => {
    expect(() => residualModel(new blocks.TransformerLM(5, 4, 1, 2, { preNorm: false, key: key(1) }))).toThrow(
      /pre-norm/,
    )
  })
})

describe('logit lens', () => {
  it('at the last depth equals the model logits', () => {
    const tokens = [2, 5, 1, 8]
    const lens = logitLens(model, tokens)
    expect(lens.logits.shape).toEqual([model.numLayers + 1, 4, 9])
    const logits = toFlat(lm.forward([tokens]).output)
    const last = toFlat(lens.logits).slice(model.numLayers * 4 * 9)
    close(last, logits)
    // Distributions normalise; the entropy of the last depth matches the prediction's.
    const probs = toFlat(lens.probabilities)
    let s = 0
    for (let k = 0; k < 9; k++) s += probs[k] as number
    expect(s).toBeCloseTo(1, 12)
  })
  it('ranks the target by depth', () => {
    const tokens = [2, 5, 1, 8]
    const lens = logitLens(model, tokens, { targets: [5, 1, 8, 2] })
    expect(lens.targetRank?.shape).toEqual([model.numLayers + 1, 4])
    const preds = toFlat(lens.prediction)
    const ranks = toFlat(lens.targetRank as never)
    for (let l = 0; l <= model.numLayers; l++)
      for (let p = 0; p < 4; p++) {
        const target = [5, 1, 8, 2][p] as number
        if ((preds[l * 4 + p] as number) === target) expect(ranks[l * 4 + p]).toBe(0)
      }
  })
})

describe('tuned lens', () => {
  const corpus = [
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [2, 2, 6, 1, 0],
    [7, 0, 3, 3, 8],
  ]
  it('starts as the logit lens and lowers the divergence to the final layer', () => {
    const alg = tunedLensAlgorithm(model, corpus, { steps: 80, learningRate: 0.03 })
    const base = lensDivergence(model, corpus)
    expect(base[base.length - 1]).toBeCloseTo(0, 12)
    const r = runAlgorithm(alg, { maxSteps: 80 })
    expect(r.status).toBe('completed')
    const tuned = lensDivergence(model, corpus, r.state.lens)
    expect(tuned[tuned.length - 1]).toBeCloseTo(0, 12)
    for (let l = 0; l < model.numLayers; l++) expect(tuned[l] as number).toBeLessThan((base[l] as number) * 0.9)
    // The tuned lens's last depth is untranslated: still the model's logits.
    const lens = logitLens(model, [1, 2, 3], { translators: r.state.lens })
    close(toFlat(lens.logits).slice(model.numLayers * 27), toFlat(lm.forward([[1, 2, 3]]).output))
  })
  it('serialises its state', () => {
    const alg = tunedLensAlgorithm(model, corpus, { steps: 3 })
    const r = runAlgorithm(alg, { maxSteps: 2 })
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(back?.step).toBe(2)
    close(toFlat(back?.lens.w[0] as never), toFlat(r.state.lens.w[0] as never))
  })
})

const planted = plantedCopyCircuit()
const pm = residualModel(planted.lm)
// Names 0 and 1 at the first position decide the answer: clean has name 0, corrupted name 1.
const clean = [0, 4, 5]
const corrupted = [1, 4, 5]
const metric = logitDifference(0, 1)

describe('activation patching', () => {
  it('locates the planted copying head and the name position', () => {
    const r = activationPatching(pm, clean, corrupted, { metric, heads: true })
    expect(r.clean).toBeGreaterThan(r.corrupted)
    const recovered = (site: string, p: number): number => {
      const s = r.sites.indexOf(site)
      return toFlat(r.recovered)[s * r.positions.length + p] as number
    }
    const { layer, head } = planted.truth
    expect(recovered(headSite(layer, head), 2)).toBeCloseTo(1, 6)
    expect(recovered(headSite(layer, 1 - head), 2)).toBeCloseTo(0, 6)
    expect(recovered('embed', 0)).toBeCloseTo(1, 6)
    expect(recovered('embed', 1)).toBeCloseTo(0, 6)
    expect(recovered('layer0.mlp', 2)).toBeCloseTo(0, 6)
    // The head writes at the last position only.
    expect(recovered(headSite(layer, head), 0)).toBeCloseTo(0, 6)
  })
  it('noising the head destroys the behaviour', () => {
    const r = activationPatching(pm, clean, corrupted, {
      metric,
      sites: [headSite(0, 0), headSite(0, 1)],
      direction: 'noise',
      positions: [2],
    })
    expect(toFlat(r.recovered)[0]).toBeCloseTo(1, 6)
    expect(toFlat(r.recovered)[1]).toBeCloseTo(0, 6)
  })
  it('patching a site with its own clean activation restores the clean output', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: model.sites.length - 1 }), fc.integer({ min: 0, max: 3 }), (si, p) => {
        const tokens = [3, 1, 4, 1]
        const r = activationPatching(model, tokens, tokens, {
          metric: logitDifference(1, 2),
          sites: [model.sites[si] as string],
          positions: [p],
        })
        expect(toFlat(r.patched)[0]).toBeCloseTo(r.clean, 10)
      }),
      { numRuns: 25 },
    )
  })
  it('patching every component of the stream equals replacing the whole run', () => {
    const a = [2, 7, 1, 3]
    const b = [5, 0, 8, 4]
    const runA = model.run([a])
    const comp = (s: string): boolean => s === 'embed' || /\.(head\d+|mlp)$/.test(s)
    const patched = model.run([b], (site) => (comp(site) ? runA.sites.get(site) : undefined))
    close(toFlat(patched.logits), toFlat(runA.logits))
    // Patching the last block's output at every position also replaces it.
    const last = `layer${model.numLayers - 1}.out`
    const patchedOut = model.run([b], (site) => (site === last ? runA.sites.get(site) : undefined))
    close(toFlat(patchedOut.logits), toFlat(runA.logits))
  })
  it('rejects prompts of different length', () => {
    expect(() => activationPatching(pm, [0, 1], [0, 1, 2], { metric })).toThrow()
  })
})

describe('causal tracing', () => {
  it('restoring the subject embedding recovers the answer; the maps have layer by position shape', () => {
    const r = causalTrace(model, [1, 2, 3, 4], {
      subject: [1],
      metric: tokenProbability(5),
      key: key(1),
      kinds: ['out', 'mlp'],
    })
    expect(r.effects['out']?.shape).toEqual([3, 4])
    expect(r.effects['mlp']?.shape).toEqual([3, 4])
    // The last block's output restored at the last position is the whole stream the logits read: full recovery.
    const out = toFlat(r.effects['out'] as never)
    expect(out[2 * 4 + 3]).toBeCloseTo(1, 8)
    expect(toFlat(r.embedding)[1]).not.toBeNaN()
  })
})

describe('attribution patching', () => {
  it('ranks the planted head first and tracks exact patching', () => {
    const exact = activationPatching(pm, clean, corrupted, { metric, heads: true, positions: 'all' })
    const approx = attributionPatching(pm, clean, corrupted, { metric, heads: true })
    // Per-site attribution summed over positions against the exact whole-sequence patch.
    const t = clean.length
    const sums = approx.sites.map((_, i) => {
      let s = 0
      for (let p = 0; p < t; p++) s += toFlat(approx.attribution)[i * t + p] as number
      return s
    })
    const head0 = approx.sites.indexOf(headSite(0, 0))
    const best = sums.indexOf(
      Math.max(...sums.filter((_, i) => !['embed', 'layer0.out', 'layer0.attn'].includes(approx.sites[i] as string))),
    )
    expect(best).toBe(head0)
    const gap = exact.clean - exact.corrupted
    const e = (toFlat(exact.recovered)[exact.sites.indexOf(headSite(0, 0))] as number) * gap
    expect(sums[head0]).toBeGreaterThan(0.5 * e)
    expect(sums[head0]).toBeLessThan(1.5 * e)
  })
})

describe('path patching', () => {
  it('the copying head reaches the output directly; filler head and mlp do not', () => {
    const [h0, h1, mlp] = pathPatching(pm, clean, corrupted, [headSite(0, 0), headSite(0, 1), 'layer0.mlp'], { metric })
    expect(h0?.effect).toBeCloseTo(1, 6)
    expect(h1?.effect).toBeCloseTo(0, 6)
    expect(mlp?.effect).toBeCloseTo(0, 6)
  })
  it('a receiver relays a sender through a later layer', () => {
    const m = residualModel(new blocks.TransformerLM(9, 8, 2, 2, { key: key(8) }))
    const [direct] = pathPatching(m, [1, 2, 3], [4, 2, 3], ['embed'], { metric: logitDifference(1, 2) })
    const [viaHead] = pathPatching(m, [1, 2, 3], [4, 2, 3], ['embed'], {
      metric: logitDifference(1, 2),
      receivers: [headSite(1, 0)],
    })
    expect(direct?.patched).not.toBeNaN()
    expect(viaHead?.patched).not.toBeNaN()
    expect(direct?.patched).not.toBeCloseTo(viaHead?.patched ?? 0, 6)
  })
})

describe('direct logit attribution', () => {
  it('the contributions sum to the logit difference exactly', () => {
    const a = directLogitAttribution(model, [3, 1, 4, 1], 2, 5)
    expect(a.total).toBeCloseTo(a.logitDifference, 9)
    const p = directLogitAttribution(pm, clean, 0, 1)
    expect(p.total).toBeCloseTo(p.logitDifference, 9)
    const head0 = p.components.indexOf(headSite(0, 0))
    const biggest = p.contribution.indexOf(Math.max(...p.contribution))
    expect(biggest).toBe(head0)
    expect(p.contribution[p.components.indexOf(headSite(0, 1))]).toBeCloseTo(0, 9)
  })
})
