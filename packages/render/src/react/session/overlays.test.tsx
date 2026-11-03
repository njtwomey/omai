import { dict, int } from 'mdml-engine/space'
import { tensor } from 'mdml-engine/tensor'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  session,
  stateFromJson,
  stateToJson,
} from 'mdml-engine/trace'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import {
  channelFocus,
  channelLayers,
  channelMarks,
  channelNumbers,
  channelRegions,
  isProbed,
  readChannel,
  regionPoints,
} from './overlays.ts'
import { SessionPanels } from './SessionPanels.tsx'
import { type LiveSession, STILL_HANDLE } from './useSession.ts'

describe('row channel values', () => {
  it('reads numbers from arrays, tensors and records with values', () => {
    expect(channelNumbers([1, 2, 3])).toEqual([1, 2, 3])
    expect(channelNumbers({ values: [4, 5] })).toEqual([4, 5])
    expect(channelNumbers(tensor([1, 2]))).toEqual([1, 2])
    expect(channelNumbers(undefined)).toBeUndefined()
  })

  it('reads a category per row from an array or from named sets, later sets winning', () => {
    expect(channelMarks([null, 'a', true, false], 4)).toEqual([undefined, 'a', 'marked', undefined])
    expect(channelMarks({ sets: { flagged: [0, 1], chosen: [1, 9] } }, 3)).toEqual(['flagged', 'chosen', undefined])
    expect(channelMarks(7, 3)).toBeUndefined()
  })

  it('reads regions (one or many) and a focus with a row', () => {
    const box = { lower: [0, null], upper: [1, 2] }
    const disc = { centre: [0, 0], radii: [1, 2] }
    expect(channelRegions(box)).toHaveLength(1)
    expect(channelRegions([box, disc, { nonsense: 1 }])).toHaveLength(2)
    expect(channelRegions(undefined)).toEqual([])
    expect(channelFocus({ row: 3, neighbours: [1] })?.row).toBe(3)
    expect(channelFocus({ row: -1 })).toBeUndefined()
    expect(channelFocus([])).toBeUndefined()
  })

  it('closes open box sides at the bounds and samples an ellipse around its centre', () => {
    const open = regionPoints({ lower: [null, 0], upper: [1, null] }, [-5, 5, -6, 6])
    expect(open).toEqual([
      [-5, 0],
      [1, 0],
      [1, 6],
      [-5, 6],
    ])
    const ring = regionPoints({ centre: [1, 1], radii: [2, 1] }, [0, 0, 0, 0])
    expect(ring).toHaveLength(72)
    expect(ring[0]).toEqual([3, 1])
  })
})

describe('row channel layers', () => {
  const rows = { mode: 'light' as const, x: [0, 1, 2, 3], y: [0, 1, 0, 1] }

  it('adds nothing without channels', () => {
    const l = channelLayers(rows)
    expect(l.under).toEqual([])
    expect(l.rows).toBeUndefined()
    expect(l.over).toEqual([])
  })

  it('colours the rows by value with a colour bar, and a diverging one is centred on zero', () => {
    const l = channelLayers({ ...rows, values: { numbers: [1, 2, 3, 4], diverging: false, label: 'score' } })
    expect(l.rows?.series).toHaveLength(1)
    expect(l.over[0]?.colourBar).toMatchObject({ min: 1, max: 4, label: 'score' })
    const d = channelLayers({ ...rows, values: { numbers: [-1, 2, 0, 1], diverging: true } })
    expect(d.over[0]?.colourBar).toMatchObject({ min: -2, max: 2 })
  })

  it('draws one marker series per category, regions beneath, and the focus over the rows', () => {
    const l = channelLayers({
      ...rows,
      groups: [0, 0, 1, 1],
      sizes: { numbers: [1, 2, 3, 4] },
      marks: { categories: ['a', undefined, 'b', 'a'], names: { a: 'labelled' } },
      regions: [{ lower: [0, 0], upper: [1, 1], label: 'rule' }],
      focus: { row: 1, neighbours: [0, 2], path: [3, 2, 1], radius: 0.5, point: [2, 2] },
    })
    expect(l.rows?.series).toHaveLength(2)
    expect(l.under).toHaveLength(1)
    const names = l.over.flatMap((layer) => layer.series.map((s) => (s as { name?: string }).name))
    expect(names).toEqual(
      expect.arrayContaining(['__neighbours', 'path', '__radius', 'related point', 'labelled', 'b', 'chosen']),
    )
  })
})

// A method whose probes give a channel each, and one tensor path of the tape.
interface State {
  readonly k: number
}
const toy: Method<State, Config> = defineMethod<State, Config>({
  name: 'toy',
  config: dict({ n: int(1, 5, { default: 1 }) }),
  make: () => ({
    name: 'toy',
    config: {},
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('score', [s.k, 1, 2])
      ctx.log('level', s.k)
      return { k: s.k + 1 }
    },
    done: (s) => (s.k >= 5 ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as State,
  }),
  contract: {
    level: scalar(),
    score: { kind: 'tensor', role: 'state', shape: [3], description: 'a score per row' } as never,
    'rows/scores': derived<State>({ description: 'a score per row', compute: (s) => [s.k, 1, 2] }),
    'rows/focus': derived<State>({ description: 'the chosen row', compute: () => ({ row: 0, neighbours: [1, 2] }) }),
  },
  defaults: { level: Log(), score: Log({ detail: 'full' }) },
})

function liveOf(steps: number): LiveSession<State, Config> {
  const s = session(toy, { mode: 'eager', steps })
  return {
    session: s,
    version: 0,
    head: s.t,
    done: s.status !== undefined,
    computing: false,
    compute: () => undefined,
    halt: () => undefined,
    advance: () => undefined,
    lastMs: 0,
    stale: false,
    held: false,
    ...STILL_HANDLE,
    apply: () => s,
  }
}

describe('reading a channel of a session', () => {
  it('probes a derived path at the step and reads a tensor path from the tape up to it', () => {
    const { session: s } = liveOf(5)
    expect(isProbed(s, { path: 'rows/scores' })).toBe(true)
    expect(isProbed(s, { path: 'score' })).toBe(false)
    expect(readChannel(s, 2, { path: 'rows/scores' })).toEqual([2, 1, 2])
    expect((readChannel(s, 3, { path: 'score' }) as number[]).slice(1)).toEqual([1, 2])
    expect(readChannel(s, 0, { path: 'nothing' })).toBeUndefined()
  })

  it('draws a rows panel with every channel to markup', () => {
    const html = renderToString(
      <SessionPanels
        id="test/rows"
        live={liveOf(5)}
        at={2}
        x={tensor([
          [0, 0],
          [1, 1],
          [2, 0],
        ])}
        y={tensor([0, 1, 1])}
        layout={[
          {
            kind: 'rows',
            values: { path: 'rows/scores', label: 'score' },
            size: { path: 'score' },
            marks: { path: 'rows/scores' },
            regions: { path: 'rows/scores' },
            focus: { path: 'rows/focus' },
          },
        ]}
      />,
    )
    expect(html).toContain('rows at step 2')
    expect(html).not.toContain('unavailable')
  })
})
