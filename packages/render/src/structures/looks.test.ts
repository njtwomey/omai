import { type SceneNode, type Structure, type StructureNode, structure } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { texExtent } from '../maths/tex.ts'
import { diagram } from './diagram.ts'

const flatten = (nodes: readonly SceneNode[]): SceneNode[] =>
  nodes.flatMap((n) => (n.kind === 'group' ? [n, ...flatten(n.children)] : [n]))
const all = (s: Structure, options = {}): SceneNode[] => flatten(diagram(s, options).children)
const node = (id: string, extra: Partial<StructureNode> = {}): StructureNode => ({
  id,
  label: id,
  role: 'operation',
  position: [0, 0],
  ...extra,
  look: { shape: 'rect', ...extra.look },
})
const texts = (nodes: readonly SceneNode[]): Extract<SceneNode, { kind: 'text' }>[] =>
  nodes.filter((n): n is Extract<SceneNode, { kind: 'text' }> => n.kind === 'text')

describe('diagram looks: opacity', () => {
  it('draws a fill, an outline and a whole shape see-through by their own opacity, on nodes and groups', () => {
    const s = structure(
      'x',
      [node('a', { group: 'g', look: { fill: 'tab1', fillOpacity: 0.2, strokeOpacity: 0.6 } })],
      [],
      {
        groups: [
          {
            id: 'g',
            label: 'g',
            kind: 'cluster',
            look: { fill: 'tab0', fillOpacity: 0.12, strokeOpacity: 0.9, opacity: 0.8 },
          },
        ],
      },
    )
    const nodes = all(s)
    const frame = nodes.find((n) => n.id === 'group:g')
    expect(frame).toMatchObject({ kind: 'rect', fill: 'tab0', fillOpacity: 0.12, strokeOpacity: 0.9, opacity: 0.8 })
    const body = nodes.find((n) => n.kind === 'rect' && n.id === undefined && n.fill === 'tab1')
    expect(body).toMatchObject({ fillOpacity: 0.2, strokeOpacity: 0.6 })
    expect((body as { opacity?: number }).opacity).toBeUndefined()
  })

  it('fades an edge and its arrow head', () => {
    const s = structure(
      'x',
      [node('a'), node('b', { position: [100, 0] })],
      [{ from: 'a', to: 'b', look: { opacity: 0.4 } }],
    )
    expect(all(s).find((n) => n.id === 'edge:0')).toMatchObject({ kind: 'path', opacity: 0.4, arrow: true })
  })
})

describe('diagram looks: edge label style', () => {
  const edge = (look: object | undefined, label = 'noise'): Structure =>
    structure(
      'x',
      [node('a'), node('b', { position: [200, 0] })],
      [{ from: 'a', to: 'b', label, ...(look ? { look } : {}) }],
    )

  it('keeps the muted two-points-smaller label by default and takes colour, size and weight from the look', () => {
    const plain = texts(all(edge(undefined))).find((t) => t.text === 'noise')
    expect(plain).toMatchObject({ fill: 'muted', size: 10 })
    expect(plain?.weight).toBeUndefined()
    const styled = texts(all(edge({ labelColour: 'fg', labelSize: 13, labelWeight: 'bold' }))).find(
      (t) => t.text === 'noise',
    )
    expect(styled).toMatchObject({ fill: 'fg', size: 13, weight: 'bold' })
  })

  it('applies the label style to TeX lines and to self-loop labels', () => {
    const t = all(edge({ labelColour: 'tab1', labelSize: 14 }, '$q(x)$')).find((n) => n.kind === 'tex')
    expect(t).toMatchObject({ fill: 'tab1', size: 16 })
    const loop = structure(
      'x',
      [node('a')],
      [{ from: 'a', to: 'a', label: 'again', look: { labelColour: 'tab2', labelWeight: 'bold' } }],
    )
    expect(texts(all(loop)).find((x) => x.text === 'again')).toMatchObject({ fill: 'tab2', weight: 'bold' })
  })
})

describe('diagram looks: TeX labels are sized by their typeset width', () => {
  const widthOf = (label: string, shape: 'rect' | 'circle' = 'rect'): number => {
    const s = structure('x', [node('a', { label, look: { shape } })], [])
    const body = all(s).find((n) => n.kind === 'rect' || n.kind === 'circle')
    return body?.kind === 'rect' ? body.width : body?.kind === 'circle' ? 2 * body.r : 0
  }

  it('fits a box to the formula, within 15% of the typeset width (the label is drawn at size 14, as the text is 12)', () => {
    for (const [source, per100] of [
      ['q(x_t | x_0)', 409.6],
      ['\\epsilon_\\theta(\\mathbf{x}_t, t)', 399],
      ['\\lVert \\epsilon - \\epsilon_\\theta \\rVert^2', 463.7],
      ['p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)', 632.4],
    ] as const) {
      const typeset = (per100 * 14) / 100
      const box = widthOf(`$${source}$`)
      expect(box - 20, source).toBeGreaterThan(typeset * 0.85)
      expect(box - 20, source).toBeLessThan(typeset * 1.15)
    }
  })

  it('no longer sizes a small formula by its source text', () => {
    // `\mathbf{x}_t` was 12 characters wide as a source and is about 15 px typeset: a circle stays at its 36 px minimum.
    expect(widthOf('$\\mathbf{x}_t$', 'circle')).toBe(36)
    expect(texExtent('\\mathbf{x}_t', 14).width).toBeLessThan(18)
  })

  it('widens a group to a TeX edge label and counts TeX widths in notes', () => {
    const s = structure('x', [node('a', { note: '$x_0 \\sim \\mathcal{N}(0, I)$' })], [])
    expect(diagram(s).width).toBeGreaterThan(texExtent('x_0 \\sim \\mathcal{N}(0, I)', 12).width)
  })
})

describe('diagram looks: pill nodes, weight and size', () => {
  it('draws a pill as a rect with a radius of half its height, and keeps its outline', () => {
    const s = structure('x', [node('a', { label: 'TRAINING', look: { shape: 'pill' } })], [])
    const body = all(s).find((n) => n.kind === 'rect')
    expect(body?.kind === 'rect' && body.radius).toBe(body?.kind === 'rect' ? body.height / 2 : -1)
    expect(body?.kind === 'rect' && body.width).toBeGreaterThan(80)
  })

  it('puts a straight, an orthogonal and a self-loop edge on the pill outline', () => {
    const s = structure(
      'x',
      [
        node('a', { label: 'a long title', look: { shape: 'pill' } }),
        node('b', { position: [300, 0], look: { shape: 'pill' } }),
        node('c', { position: [150, 120], look: { shape: 'pill' } }),
      ],
      [
        { from: 'a', to: 'b' },
        { from: 'a', to: 'c' },
        { from: 'a', to: 'a', label: 'loop' },
      ],
    )
    for (const route of ['straight', 'orthogonal'] as const) {
      const edges = all(s, { route }).filter((n) => n.kind === 'path' && n.id?.startsWith('edge:'))
      expect(edges).toHaveLength(3)
      for (const e of edges)
        if (e.kind === 'path') for (const [x, y] of e.points) expect(Number.isFinite(x + y)).toBe(true)
    }
    // An orthogonal edge into the left end of a pill meets its half circle, not the corner of its box.
    const left = structure(
      'x',
      [node('a', { position: [-200, 0] }), node('p', { label: 'pill', look: { shape: 'pill' } })],
      [{ from: 'a', to: 'p' }],
    )
    const route = all(left, { route: 'orthogonal' }).find((n) => n.id === 'edge:0')
    const pill = all(left).find((n) => n.kind === 'rect' && n.radius !== undefined && n.radius > 10)
    if (route?.kind !== 'path' || pill?.kind !== 'rect') throw new Error('missing')
    const end = route.points[route.points.length - 1] as readonly [number, number]
    expect(end[0]).toBeLessThanOrEqual(pill.x + pill.width / 2 + 0.5)
    expect(end[0]).toBeGreaterThanOrEqual(pill.x - 0.5)
  })

  it('sets the label weight and size of a node, and sizes the box for them', () => {
    const plain = structure('x', [node('a', { label: 'some title' })], [])
    const bold = structure('x', [node('a', { label: 'some title', look: { fontWeight: 'bold', fontSize: 18 } })], [])
    const t = texts(all(bold)).find((x) => x.text === 'some title')
    expect(t).toMatchObject({ size: 18, weight: 'bold' })
    expect(texts(all(plain)).find((x) => x.text === 'some title')?.weight).toBeUndefined()
    const body = (s: Structure): number => {
      const r = all(s).find((n) => n.kind === 'rect')
      return r?.kind === 'rect' ? r.width : 0
    }
    expect(body(bold)).toBeGreaterThan(body(plain) * 1.4)
  })
})

describe('diagram looks: group labels', () => {
  const grouped = (look: object | undefined): Structure =>
    structure('x', [node('a', { group: 'g' }), node('b', { group: 'g', position: [200, 0] })], [], {
      groups: [{ id: 'g', label: 'the group', kind: 'cluster', ...(look ? { look } : {}) }],
    })
  const frame = (s: Structure): { x: number; y: number; width: number; height: number } => {
    const r = all(s).find((n) => n.id === 'group:g')
    if (r?.kind !== 'rect') throw new Error('no frame')
    return r
  }
  const label = (s: Structure): Extract<SceneNode, { kind: 'text' }> => {
    const t = texts(all(s)).find((x) => x.text === 'the group')
    if (t === undefined) throw new Error('no label')
    return t
  }

  it('keeps the top left, regular, 24 px label as the default', () => {
    const f = frame(grouped(undefined))
    const l = label(grouped(undefined))
    expect(l.x).toBe(f.x + 8)
    expect(l.y).toBe(f.y + 14)
    expect(l.size).toBe(11)
    expect(l.weight).toBeUndefined()
    // 24 above the nodes (28 high at the top of the box) and 12 below.
    expect(f.height).toBe(24 + 28 + 12)
  })

  it('takes weight, size and position, and the room for the label grows with its size', () => {
    const big = grouped({ labelWeight: 'bold', labelSize: 20, labelPosition: 'top-centre' })
    const f = frame(big)
    const l = label(big)
    expect(l).toMatchObject({ size: 20, weight: 'bold', anchor: 'middle' })
    expect(l.x).toBeCloseTo(f.x + f.width / 2, 6)
    expect(f.height).toBe(20 + 13 + 28 + 12)
    expect(l.y).toBe(f.y + 20 + 3)

    const bottom = grouped({ labelPosition: 'bottom-left' })
    const [fb, lb] = [frame(bottom), label(bottom)]
    expect(lb.y).toBe(fb.y + fb.height - 7)
    expect(fb.height).toBe(12 + 28 + 24)
  })

  it('is as wide as its label however large', () => {
    const s = structure('x', [node('a', { group: 'g' })], [], {
      groups: [
        {
          id: 'g',
          label: 'a very long group title indeed',
          kind: 'cluster',
          look: { labelSize: 24, labelWeight: 'bold' },
        },
      ],
    })
    const r = all(s).find((n) => n.id === 'group:g')
    expect(r?.kind === 'rect' && r.width).toBeGreaterThan(30 * 24 * 0.6)
  })
})

describe('diagram looks: notes', () => {
  const noted = (note: StructureNode['note'], group = false): Structure =>
    structure('x', [node('a', { ...(note === undefined ? {} : { note }), ...(group ? { group: 'g' } : {}) })], [], {
      groups: group ? [{ id: 'g', label: 'g', kind: 'cluster' }] : [],
    })
  const nodeBox = (s: Structure): { y0: number; y1: number } => {
    const r = all(s).find((n) => n.kind === 'rect' && n.id === undefined)
    if (r?.kind !== 'rect') throw new Error('no box')
    return { y0: r.y, y1: r.y + r.height }
  }

  it('keeps a string note below the node, muted and untied', () => {
    const s = noted('under')
    const t = texts(all(s)).find((x) => x.text === 'under')
    expect(t?.fill).toBe('muted')
    expect(t?.y).toBeGreaterThan(nodeBox(s).y1)
    expect(all(s).filter((n) => n.kind === 'path')).toHaveLength(0)
  })

  it('takes a record note above the node, in ink, with a dashed tie, and grows the scene and the group frame', () => {
    const s = noted({ text: 'data', position: 'above', tie: 'dashed', colour: 'fg', size: 13, weight: 'bold' }, true)
    const nodes = all(s)
    const t = texts(nodes).find((x) => x.text === 'data')
    expect(t).toMatchObject({ fill: 'fg', size: 13, weight: 'bold', anchor: 'middle' })
    const box = nodeBox(s)
    expect(t?.y).toBeLessThan(box.y0)
    const tie = nodes.find((n) => n.kind === 'path')
    expect(tie).toMatchObject({ kind: 'path', dash: [3, 3], stroke: 'fg' })
    if (tie?.kind !== 'path') throw new Error('no tie')
    const [a, b] = tie.points as readonly [readonly [number, number], readonly [number, number]]
    expect(a[1]).toBeCloseTo(box.y0, 6)
    expect(b[1]).toBeLessThan(a[1])
    expect(b[1]).toBeGreaterThan((t?.y ?? 0) - 1)
    const frame = nodes.find((n) => n.id === 'group:g')
    if (frame?.kind !== 'rect') throw new Error('no frame')
    // The note is inside the frame, and the frame inside the scene.
    expect(frame.y).toBeLessThan((t?.y ?? 0) - 6)
    expect(diagram(s).height).toBeGreaterThan(frame.height)
  })

  it('puts a note beside its node, solid-tied, and keeps a note on the side asked for', () => {
    for (const position of ['left', 'right', 'below'] as const) {
      const s = noted({ text: 'beside', position, tie: true })
      const t = texts(all(s)).find((x) => x.text === 'beside')
      const tie = all(s).find((n) => n.kind === 'path')
      expect(tie?.kind === 'path' && tie.dash).toBeUndefined()
      expect(t?.anchor).toBe(position === 'left' ? 'end' : position === 'right' ? 'start' : 'middle')
    }
  })

  it('moves a tied note without a side out by the room for its tie', () => {
    const plain = texts(all(noted('x1'))).find((x) => x.text === 'x1')
    const tied = texts(all(noted({ text: 'x1', tie: true }))).find((x) => x.text === 'x1')
    expect((tied?.y ?? 0) - (plain?.y ?? 0)).toBeGreaterThanOrEqual(8)
  })
})
