import { describe, expect, it } from 'vitest'
import { type GraphRecord, generate, load, memoryCache, rowCount, sha256Hex } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  barsAndStripes,
  contextualSbm,
  digitGlyphs,
  familyGraph,
  fashionMnistTest,
  type GRAPH_FAMILIES,
  geometricScenes,
  graphFamilies,
  karateClub,
  kinshipTriples,
  lesMiserables,
  shapes,
  smallMolecules,
} from './index.ts'

function degrees(g: GraphRecord): number[] {
  const d = new Array<number>(g.nNodes).fill(0)
  const e = toFlat(g.edges)
  for (const v of e) d[v] = (d[v] ?? 0) + 1
  return d
}

function connected(g: GraphRecord): boolean {
  const adj = Array.from({ length: g.nNodes }, () => [] as number[])
  const e = toFlat(g.edges)
  for (let i = 0; i < e.length; i += 2) {
    adj[e[i] ?? 0]?.push(e[i + 1] ?? 0)
    adj[e[i + 1] ?? 0]?.push(e[i] ?? 0)
  }
  const seen = new Set([0])
  const stack = [0]
  while (stack.length > 0) for (const v of adj[stack.pop() ?? 0] ?? []) if (!seen.has(v)) seen.add(v) && stack.push(v)
  return seen.size === g.nNodes
}

describe('graphs', () => {
  it('karate club and Les Miserables', () => {
    const k = karateClub()
    const g = k.graphs[0] as GraphRecord
    expect(g.nNodes).toBe(34)
    expect(toFlat(g.nodeLabels as Tensor).filter((c) => c === 0).length).toBe(17)
    expect(degrees(g)[0]).toBe(16)
    expect(degrees(g)[33]).toBe(17)
    const l = lesMiserables().graphs[0] as GraphRecord
    expect(l.edges.shape).toEqual([254, 2])
    expect(l.nodeNames).toContain('Valjean')
  })

  it('graph families have their defining properties', () => {
    const n = 12
    const f = (family: (typeof GRAPH_FAMILIES)[number]) => familyGraph(key(1), family, n)
    expect(f('complete').edges.shape[0]).toBe(66)
    expect(f('path').edges.shape[0]).toBe(11)
    expect(degrees(f('cycle')).every((d) => d === 2)).toBe(true)
    expect(degrees(f('star'))[0]).toBe(11)
    const tree = f('tree')
    expect(tree.edges.shape[0]).toBe(11)
    expect(connected(tree)).toBe(true)
    expect(f('grid').edges.shape[0]).toBe(17)
    expect(connected(familyGraph(key(2), 'barabasiAlbert', 50, { m: 2 }))).toBe(true)
    const er = familyGraph(key(3), 'erdosRenyi', 200, { p: 0.1 })
    expect(Math.abs((er.edges.shape[0] ?? 0) / 19900 - 0.1)).toBeLessThan(0.01)
    const ws = familyGraph(key(4), 'wattsStrogatz', 40, { m: 2, p: 0 })
    expect(degrees(ws).every((d) => d === 4)).toBe(true)
    const { data } = generate(graphFamilies, { key: key(5), knobs: { graphs: 30 } })
    expect(rowCount(data)).toBe(30)
  })

  it('contextual SBMs have denser blocks and informative features', () => {
    const { data, truth } = generate(contextualSbm, { key: key(6), knobs: { nodes: 400, degree: 20, lambda: 3 } })
    const g = data.graphs[0] as GraphRecord
    const blocks = toFlat(truth.clusters as Tensor)
    const e = toFlat(g.edges)
    let within = 0
    for (let i = 0; i < e.length; i += 2) if (blocks[e[i] ?? 0] === blocks[e[i + 1] ?? 0]) within++
    expect(within / (e.length / 2)).toBeGreaterThan(0.6)
    const avgDegree = e.length / 400
    expect(Math.abs(avgDegree - 20)).toBeLessThan(2)
    const masks = data.masks
    const total = ['train', 'val', 'test'].reduce(
      (s, m) => s + toFlat(masks?.[m as 'train'] as Tensor).filter((v) => v === 1).length,
      0,
    )
    expect(total).toBe(400)
  })

  it('small molecules have the right formulas and rings', () => {
    const { data, truth } = smallMolecules()
    const names = truth.extra?.['names'] as string[]
    const formulas = truth.extra?.['formulas'] as string[]
    const rings = truth.extra?.['ringCounts'] as number[]
    const weight = truth.extra?.['molecularWeights'] as number[]
    const of = (name: string) => names.indexOf(name)
    expect(formulas[of('caffeine')]).toBe('C8H10N4O2')
    expect(formulas[of('aspirin')]).toBe('C9H8O4')
    expect(formulas[of('ibuprofen')]).toBe('C13H18O2')
    expect(formulas[of('paracetamol')]).toBe('C8H9NO2')
    expect(formulas[of('nicotine')]).toBe('C10H14N2')
    expect(formulas[of('glucose (open chain)')]).toBe('C6H12O6')
    expect(formulas[of('water')]).toBe('H2O')
    expect(formulas[of('thiophene')]).toBe('C4H4S')
    expect(formulas[of('chloroform')]).toBe('CHCl3')
    expect(rings[of('naphthalene')]).toBe(2)
    expect(rings[of('caffeine')]).toBe(2)
    expect(rings[of('ethanol')]).toBe(0)
    expect(weight[of('caffeine')]).toBeCloseTo(194.194, 2)
    expect(toFlat(data.graphLabels as Tensor)[of('benzene')]).toBe(1)
  })

  it('kinship triples follow their rules', () => {
    const { data, truth } = generate(kinshipTriples, { key: key(7), knobs: { observed: 1 } })
    const all = truth.extra?.['allTriples'] as [number, number, number][]
    const has = new Set(all.map((t) => t.join(',')))
    for (const [h, r, t] of all) {
      if (r === 0) expect(has.has(`${t},1,${h}`)).toBe(true)
      if (r === 2) expect(has.has(`${t},2,${h}`)).toBe(true)
      if (r === 3) expect(has.has(`${t},3,${h}`)).toBe(true)
    }
    expect(rowCount(data)).toBe(all.length)
    const partial = generate(kinshipTriples, { key: key(7), knobs: { observed: 0.5 } })
    const held = partial.truth.extra?.['heldOut'] as unknown[]
    expect(rowCount(partial.data) + held.length).toBe(all.length)
  })
})

describe('images', () => {
  it('shapes: masks lie inside boxes and match the labels', () => {
    const { data } = generate(shapes, { key: key(8), knobs: { n: 20, size: 24, noise: 0 } })
    const m = toFlat(data.masks as Tensor)
    const labels = toFlat(data.labels as Tensor)
    for (let i = 0; i < 20; i++) {
      const box = toFlat((data.boxes ?? [])[i] as Tensor)
      for (let y = 0; y < 24; y++)
        for (let x = 0; x < 24; x++) {
          const v = m[i * 576 + y * 24 + x] ?? 0
          if (v > 0) {
            expect(v - 1).toBe(labels[i])
            expect(x + 1).toBeGreaterThanOrEqual(box[0] ?? 0)
            expect(x).toBeLessThanOrEqual(box[2] ?? 0)
          }
        }
    }
  })
  it('digit glyphs, bars and stripes', () => {
    const g = generate(digitGlyphs, { key: key(9), knobs: { n: 30, noise: 0, jitter: 0 } })
    const clean = toFlat(g.truth.latents?.['clean'] as Tensor)
    expect(clean.slice(0, 256).some((v) => v > 0)).toBe(true)
    const b = generate(barsAndStripes, { knobs: { side: 4 } })
    expect(b.data.images.shape[0]).toBe(30)
    const px = toFlat(b.data.images)
    const patterns = new Set(Array.from({ length: 30 }, (_, i) => px.slice(i * 16, (i + 1) * 16).join('')))
    expect(patterns.size).toBe(30)
  })
  it('geometric scenes mark edges between regions and visible corners', () => {
    const { data, truth } = generate(geometricScenes, { key: key(10), knobs: { n: 3, size: 40, noise: 0 } })
    const edges = toFlat(truth.latents?.['edges'] as Tensor)
    const regions = toFlat(data.masks as Tensor)
    for (let i = 0; i < 3 * 1600; i++) {
      const x = i % 40
      if (edges[i] === 0 && x + 1 < 40 && (i + 1) % 1600 !== 0) expect(regions[i + 1]).toBe(regions[i])
    }
    const corners = truth.latents?.['corners'] as number[][][]
    expect(corners.every((c) => c.length > 0)).toBe(true)
  })
  it('Fashion-MNIST parses gzipped IDX files', async () => {
    const gz = async (b: Uint8Array) =>
      new Uint8Array(
        await new Response(
          new Blob([b.slice().buffer]).stream().pipeThrough(new CompressionStream('gzip')),
        ).arrayBuffer(),
      )
    const img = new Uint8Array(16 + 2 * 4)
    new DataView(img.buffer).setUint32(0, 0x00000803, false)
    new DataView(img.buffer).setUint32(4, 2, false)
    new DataView(img.buffer).setUint32(8, 2, false)
    new DataView(img.buffer).setUint32(12, 2, false)
    img.set([0, 255, 51, 102, 0, 0, 0, 255], 16)
    const lab = new Uint8Array(8 + 2)
    new DataView(lab.buffer).setUint32(0, 0x00000801, false)
    new DataView(lab.buffer).setUint32(4, 2, false)
    lab.set([9, 0], 8)
    const files = [await gz(img), await gz(lab)]
    const ref = {
      ...fashionMnistTest,
      files: await Promise.all(files.map(async (f, i) => ({ url: `file${i}`, sha256: await sha256Hex(f) }))),
    }
    const fetcher = async (url: string) => ({
      ok: true,
      status: 200,
      arrayBuffer: async () => (files[Number(url.slice(4))] as Uint8Array).slice().buffer,
    })
    const d = await load(ref, { fetch: fetcher, cache: memoryCache() })
    expect(d.images.shape).toEqual([2, 2, 2, 1])
    expect(toFlat(d.images).slice(0, 4)).toEqual([0, 1, 0.2, 0.4])
    expect(d.classes?.[toFlat(d.labels as Tensor)[0] ?? 0]).toBe('Ankle boot')
  })
})
