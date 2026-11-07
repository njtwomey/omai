import { type ReactElement, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { themeName } from '../theme/echarts.ts'
import { inAct, mountElement, resetGlobals } from './_harness.tsx'
import { Chart, type ChartProps } from './Chart.tsx'

interface FakeChart {
  readonly theme: string
  readonly renderer: unknown
  readonly calls: [string, ...unknown[]][]
  readonly handlers: Map<string, (params: unknown) => void>
  group?: string
  disposed: boolean
}

// ECharts needs a real canvas: stand in for it with an instance that records what the component asks of it.
const made: FakeChart[] = []
const connected: string[] = []
vi.mock('./echarts.ts', () => ({
  loadEcharts: () => ({
    init: (_el: unknown, theme: string, options: { renderer: unknown }) => {
      const chart: FakeChart = { theme, renderer: options.renderer, calls: [], handlers: new Map(), disposed: false }
      made.push(chart)
      return {
        setOption: (option: unknown, flags: unknown) => chart.calls.push(['setOption', option, flags]),
        on: (name: string, fn: (params: unknown) => void) => {
          chart.handlers.set(name, fn)
          chart.calls.push(['on', name])
        },
        off: (name: string, fn: (params: unknown) => void) => {
          if (chart.handlers.get(name) === fn) chart.handlers.delete(name)
          chart.calls.push(['off', name])
        },
        dispose: () => {
          chart.disposed = true
        },
        isDisposed: () => chart.disposed,
        resize: () => undefined,
        set group(g: string) {
          chart.group = g
        },
      }
    },
    connect: (g: string) => connected.push(g),
  }),
}))

afterEach(() => {
  made.length = 0
  connected.length = 0
  resetGlobals()
})

const setOptions = (c: FakeChart): unknown[][] => c.calls.filter((x) => x[0] === 'setOption').map((x) => x.slice(1))

// The host draws whatever `host.props` holds now; a test changes it and redraws.
const onClick = vi.fn()
const host: { props: ChartProps; redraw: () => void } = { props: { option: {} }, redraw: () => undefined }
function Host(): ReactElement {
  const [, set] = useState(0)
  host.redraw = () => set((n) => n + 1)
  return <Chart {...host.props} onEvents={{ click: onClick }} />
}

describe('Chart', () => {
  it('creates an instance for the theme, draws and redraws its option, wires events and groups, and disposes it', async () => {
    const change = (next: Partial<ChartProps>): Promise<void> => {
      host.props = { ...host.props, ...next }
      return inAct(host.redraw)
    }
    host.props = { option: { series: [{ type: 'line', data: [1] }] }, mode: 'dark', group: 'g' }
    const m = await mountElement(<Host />)
    const first = made[0] as FakeChart
    expect([made.length, first.theme, first.renderer]).toEqual([1, themeName('dark'), 'canvas'])
    // The option is drawn as a replacement, at creation and when the instance is set; the group is connected once.
    expect(setOptions(first)[0]).toEqual([host.props.option, { notMerge: true }])
    expect(setOptions(first).at(-1)).toEqual([host.props.option, { notMerge: true, lazyUpdate: true }])
    expect([first.group, connected]).toEqual(['g', ['g']])
    // An event reaches the handler with its parameters and the instance; the colour-bar listener is bound too.
    first.handlers.get('click')?.({ x: 1 })
    expect(onClick).toHaveBeenCalledWith({ x: 1 }, expect.objectContaining({ isDisposed: expect.any(Function) }))
    expect([...first.handlers.keys()].sort()).toEqual(['click', 'datarangeselected'])

    // A new option replaces; with `merge` it merges. The instance is kept.
    await change({ option: { series: [{ type: 'bar', data: [2] }] } })
    expect(setOptions(first).at(-1)).toEqual([host.props.option, { notMerge: true, lazyUpdate: true }])
    await change({ option: { series: [{ type: 'bar', data: [3] }] }, merge: true })
    expect(setOptions(first).at(-1)).toEqual([host.props.option, { notMerge: false, lazyUpdate: true }])
    expect(made.length).toBe(1)

    // Another colour mode disposes the instance and creates one for the new theme, drawing the current option.
    await change({ mode: 'light' })
    expect([first.disposed, made.length, made[1]?.theme]).toEqual([true, 2, themeName('light')])
    expect(setOptions(made[1] as FakeChart)[0]).toEqual([host.props.option, { notMerge: true }])

    // Unmounting disposes and unbinds every handler it bound.
    await m.unmount()
    const last = made[1] as FakeChart
    expect(last.disposed).toBe(true)
    expect(last.handlers.size).toBe(0)
  })
})
