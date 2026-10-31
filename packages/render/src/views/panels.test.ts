import {
  imageSteps,
  markedTable,
  sampleVectorField,
  stripRecord,
  structureSteps,
  vectorFieldRecord,
} from 'mdml-engine/exposition'
import { structure } from 'mdml-engine/scene'
import { describe, expect, it } from 'vitest'
import { vectorFieldAt, vectorFieldLayers, vectorFieldView } from './fields.ts'
import { imageGridView, imageStepIndex } from './images.ts'
import { markedTableModel, markedTableScene } from './marked-tables.ts'
import { stillOf, viewFor } from './registry.ts'
import { stripView } from './strips.ts'
import { structureStepsScene, structureValueRange } from './structure-steps.ts'

const dataOf = (s: { data?: unknown } | undefined): unknown[] => (s?.data as unknown[] | undefined) ?? []
type Series = { name?: string; type?: string; data?: unknown[]; lineStyle?: { type?: string } }
const seriesOf = (o: unknown): Series[] => (o as { series: Series[] }).series

const GRID = [-1, -0.5, 0, 0.5, 1]
// A damped oscillator x' = y, y' = -x - 0.5 y: a stable spiral at the origin.
const spiral = sampleVectorField((x, y) => [y, -x - 0.5 * y], GRID, GRID, {
  features: ['x', 'v'],
  nullclines: [
    {
      name: 'x′ = 0',
      paths: [
        [
          [-1, 0],
          [1, 0],
        ],
      ],
    },
  ],
  fixedPoints: [{ at: [0, 0], stability: 'stable', label: 'rest' }],
})

describe('the vector field view', () => {
  it('interpolates the grid bilinearly and holds it at the edge', () => {
    const f = vectorFieldAt(
      vectorFieldRecord({
        x: [0, 1],
        y: [0, 1],
        u: [
          [0, 2],
          [0, 2],
        ],
        v: [
          [0, 0],
          [4, 4],
        ],
      }),
    )
    expect(f(0.5, 0.5)).toEqual([1, 2])
    expect(f(2, 2)).toEqual([2, 4])
    expect(f(-3, 0)).toEqual([0, 0])
  })

  it('draws arrows, dashed nullclines and fixed points by class, and names the axes', () => {
    const layers = vectorFieldLayers(spiral, { density: 5 })
    expect(layers).toHaveLength(3)
    expect(dataOf(layers[0]?.series[0] as { data?: unknown }).length).toBeGreaterThan(10)
    expect(layers[1]?.series[0]?.name).toBe('x′ = 0')
    expect(layers[2]?.series.map((s) => (s as Series).name)).toEqual(['stable'])
    const option = vectorFieldView(spiral) as { xAxis: { name?: string }[]; yAxis: { name?: string }[] }
    expect(option.xAxis[0]?.name).toBe('x')
    expect(option.yAxis[0]?.name).toBe('v')
  })

  it('can leave out the nullclines and fixed points, and draw streamlines that follow the flow', () => {
    expect(vectorFieldLayers(spiral, { nullclines: false, fixedPoints: false })).toHaveLength(1)
    const [stream] = vectorFieldLayers(spiral, { style: 'streamlines', nullclines: false, fixedPoints: false })
    const path = (dataOf(stream?.series[0] as { data?: unknown }) as (number | null)[][]).filter((p) => p[0] !== null)
    expect(path.length).toBeGreaterThan(20)
    // Streamlines are drawn inside the grid.
    expect(path.every((p) => Math.abs(p[0] as number) <= 1 && Math.abs(p[1] as number) <= 1)).toBe(true)
    expect(vectorFieldLayers(spiral, { style: 'both', nullclines: false, fixedPoints: false })).toHaveLength(2)
  })
})

describe('the strip view', () => {
  const strip = stripRecord({
    tokens: ['the', 'cat', 'sat'],
    tracks: [
      { name: 'noun', values: [0.1, 0.9, 0.2] },
      { name: 'verb', values: [0.1, 0.05, 0.8] },
    ],
    tags: [{ name: 'tag', labels: ['DET', 'N', 'V'] }],
    path: [0, 0, 1],
    marks: [2],
  })

  it('draws the cells, the tag labels, the path through its cells and the outlined marks', () => {
    const option = stripView(strip) as { xAxis: { data?: string[] }[]; yAxis: { data?: string[] }[] }
    const names = seriesOf(option).map((s) => s.name)
    expect(names).toEqual(['__matrix', '__tags', '__path', '__marks'])
    expect(option.xAxis[0]?.data).toEqual(['the', 'cat', 'sat'])
    // Rows are listed bottom-up: the tag row first, then the tracks reversed.
    expect(option.yAxis[0]?.data).toEqual(['tag', 'verb', 'noun'])
    const path = seriesOf(option).find((s) => s.name === '__path')
    // Position 2's path is the verb track, the second row from the top, at category index 1.
    expect((dataOf(path) as [number, number][]).map((p) => p[1])).toEqual([2, 2, 1])
    const tags = seriesOf(option).find((s) => s.name === '__tags')
    expect(tags?.data).toHaveLength(3)
  })

  it('draws a strip of nothing but tracks without tags, path or marks as the cells alone', () => {
    const plain = stripRecord({ tokens: ['a'], tracks: [{ name: 'p', values: [1] }] })
    expect(seriesOf(stripView(plain)).map((s) => s.name)).toEqual(['__matrix'])
  })
})

describe('the marked table view', () => {
  const table = markedTable({
    columns: ['test', 'p'],
    rowNames: ['one', 'two'],
    cells: [
      ['t', 0.0312],
      ['z', null],
    ],
    marks: [
      { row: 0, status: 'warn' },
      { row: 0, col: 1, status: 'bad', note: 'p < 0.05' },
      { row: 0, status: 'decision' },
      { col: 1, status: 'info' },
    ],
    digits: 2,
  })

  it('resolves the marks per cell: the last tint wins, a decision outlines, a missing cell is a dash', () => {
    const model = markedTableModel(table)
    expect(model.rowNames).toEqual(['one', 'two'])
    expect(model.cells[0]?.map((c) => c.text)).toEqual(['t', '0.031'])
    expect(model.cells[0]?.map((c) => c.status)).toEqual(['warn', 'info'])
    expect(model.cells[0]?.map((c) => c.decision)).toEqual([true, true])
    expect(model.cells[0]?.[1]?.note).toBe('p < 0.05')
    expect(model.cells[1]?.[1]).toMatchObject({ text: '—', status: 'info', decision: false, numeric: false })
  })

  it('draws a scene as wide as its columns, with a tint and an outline where the marks say', () => {
    const s = markedTableScene(table)
    expect(s.kind).toBe('scene')
    expect(s.height).toBe(24 * 3)
    const rects = s.children.filter((n) => n.kind === 'rect')
    // Three tinted cells, and the decided row outlined, its name included.
    expect(rects.filter((r) => r.fill !== 'none')).toHaveLength(3)
    expect(rects.filter((r) => r.stroke === 'accent')).toHaveLength(3)
    const words = s.children.filter((n) => n.kind === 'text').map((n) => (n as { text: string }).text)
    expect(words).toContain('0.031')
    expect(words).toContain('p')
  })
})

describe('the structure view', () => {
  const g = structure(
    'graph',
    ['a', 'b', 'c'].map((id) => ({ id, label: id, role: 'latent' as const })),
    [
      { from: 'a', to: 'b' },
      { from: 'b', to: 'c', directed: false },
    ],
  )
  const record = structureSteps({
    structure: g,
    valueName: 'belief',
    steps: [
      { step: 0, nodes: { a: { value: 0 } }, caption: 'start' },
      {
        step: 5,
        nodes: { a: { value: 1, size: 2 }, b: { category: 1, label: 'B' }, c: { position: [3, 4] } },
        edges: [
          { from: 'a', to: 'b', width: 3, label: 'm', dashed: true },
          { from: 'b', to: 'c', hidden: true },
        ],
        caption: 'later',
      },
    ],
  })
  const kinds = (s: ReturnType<typeof structureStepsScene>): Record<string, number> => {
    const out: Record<string, number> = {}
    for (const n of s.children) out[n.kind] = (out[n.kind] ?? 0) + 1
    return out
  }

  it('reads the latest step at or before the viewed one', () => {
    const early = structureStepsScene(record, 3)
    const late = structureStepsScene(record, 9)
    const text = (s: ReturnType<typeof structureStepsScene>): string[] =>
      s.children.flatMap((n) => (n.kind === 'text' ? [n.text] : []))
    expect(text(early)).toContain('start')
    expect(text(late)).toContain('later')
    expect(text(late)).toContain('B')
    expect(text(late)).toContain('m')
    expect(text(late)).toContain('belief: 0 to 1')
  })

  it('hides an edge a step hides, sizes a node, colours by class and by value', () => {
    expect(kinds(structureStepsScene(record, 0))).toMatchObject({ line: 2, circle: 3 })
    expect(kinds(structureStepsScene(record, 5))).toMatchObject({ line: 1, circle: 3 })
    const circles = structureStepsScene(record, 5).children.filter((n) => n.kind === 'circle') as {
      r: number
      fill: string
    }[]
    expect(circles[0]?.r).toBeCloseTo(28, 6)
    expect(circles[1]?.fill).toBe('#ff7f0e')
    expect(circles[0]?.fill).toMatch(/^#/)
    const lines = structureStepsScene(record, 5).children.filter((n) => n.kind === 'line') as {
      arrow?: boolean
      dash?: readonly number[]
    }[]
    expect(lines[0]).toMatchObject({ arrow: true, dash: [5, 4] })
  })

  it('keeps nodes where they are until a step moves them, and fits the value range over every step', () => {
    const at = (step: number, i: number): [number, number] => {
      const c = structureStepsScene(record, step).children.filter((n) => n.kind === 'circle')[i] as {
        cx: number
        cy: number
      }
      return [c.cx, c.cy]
    }
    expect(at(0, 0)).toEqual(at(5, 0))
    expect(at(0, 2)).not.toEqual(at(5, 2))
    expect(structureValueRange(record)).toEqual([0, 1])
    expect(structureValueRange({ ...record, range: [2, 3] })).toEqual([2, 3])
  })
})

describe('the image grid view', () => {
  const record = imageSteps({
    method: 'classifier',
    height: 2,
    width: 2,
    groups: ['wrong', 'right'],
    steps: [
      {
        iteration: 0,
        images: [
          [[0, 0, 0, 0]],
          [
            [1, 1, 1, 1],
            [0, 1, 0, 1],
          ],
        ],
        labels: [['3 as 8'], ['1', '7']],
      },
      {
        iteration: 10,
        images: [
          [[1, 0, 0, 1]],
          [
            [1, 1, 1, 1],
            [0, 1, 0, 1],
          ],
        ],
        labels: [['4 as 9'], ['1', '7']],
      },
    ],
  })

  it('picks the latest step at or before the viewed one and writes the captions under the images', () => {
    expect(imageStepIndex(record, 3)).toBe(0)
    expect(imageStepIndex(record, 10)).toBe(1)
    expect(imageStepIndex(record, -5)).toBe(0)
    const caption = (step: number): string[] =>
      (dataOf(seriesOf(imageGridView(record, step)).find((s) => s.name === '__captions')) as { name: string }[]).map(
        (d) => d.name,
      )
    expect(caption(3)).toEqual(['3 as 8', '1', '7'])
    expect(caption(12)).toEqual(['4 as 9', '1', '7'])
  })

  it('has no captions series or objective when the step has no labels', () => {
    const plain = imageSteps({
      method: 'm',
      height: 1,
      width: 1,
      groups: ['g'],
      steps: [
        { iteration: 0, images: [[[0]]], objective: 1 },
        { iteration: 1, images: [[[1]]], objective: 0.5 },
      ],
    })
    const names = seriesOf(imageGridView(plain)).map((s) => s.name)
    expect(names).not.toContain('__captions')
    expect(names).not.toContain('objective')
  })
})

describe('the registry of the new kinds', () => {
  it('names a component for each and draws a headless still', () => {
    expect(viewFor('vectorField').component).toBe('VectorFieldView')
    expect(viewFor('strip').component).toBe('StripView')
    expect(viewFor('markedTable').component).toBe('MarkedTableView')
    expect(viewFor('structureSteps').component).toBe('StructureStepsView')
    const g = structure('g', [{ id: 'a', label: 'a', role: 'latent' }], [])
    expect(stillOf(spiral)).toHaveProperty('series')
    expect(stillOf(stripRecord({ tokens: ['a'], tracks: [{ name: 'p', values: [1] }] }))).toHaveProperty('series')
    expect(stillOf(markedTable({ columns: ['a'], cells: [[1]] }))).toHaveProperty('children')
    expect(stillOf(structureSteps({ structure: g, steps: [{ step: 0 }] }))).toHaveProperty('children')
  })
})
