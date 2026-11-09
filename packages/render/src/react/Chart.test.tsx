import { type ReactElement, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { themeName } from '../theme/echarts.ts'
import { tokens } from '../theme/tokens.ts'
import { inAct, mountElement, resetGlobals } from './_harness.tsx'
import { Chart, type ChartProps } from './Chart.tsx'
import { ThemeProvider } from './theme.tsx'

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
const registered: [string, { readonly textStyle: { readonly color: string } }][] = []
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
    registerTheme: (name: string, theme: (typeof registered)[number][1]) => registered.push([name, theme]),
  }),
}))

afterEach(() => {
  made.length = 0
  connected.length = 0
  registered.length = 0
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

describe('Chart text alternative', () => {
  it('is an image labelled by the reader\'s text, else the option\'s own description, else its title, else "Chart"', async () => {
    const labelOf = (m: Awaited<ReturnType<typeof mountElement>>): unknown =>
      m.elements().find((e) => e.props['data-omai-chart'] !== undefined)?.props
    const cases: readonly [ChartProps, string][] = [
      [{ option: {}, label: 'Loss over steps' }, 'Loss over steps'],
      [
        { option: { aria: { label: { description: 'A vector field' } }, title: { text: 'ignored' } } },
        'A vector field',
      ],
      [{ option: { title: [{ text: 'Loss' }] } }, 'Loss'],
      [{ option: {} }, 'Chart'],
    ]
    for (const [props, want] of cases) {
      const m = await mountElement(<Chart {...props} />)
      expect(labelOf(m)).toMatchObject({ role: 'img', 'aria-label': want })
      await m.unmount()
    }
    // A label of the reader's own is also the description ECharts writes to the element, in place of a generated one.
    host.props = { option: { series: [] }, label: 'Loss over steps' }
    const m = await mountElement(<Host />)
    const drawn = setOptions(made.at(-1) as FakeChart).at(-1)?.[0] as { aria: unknown }
    expect(drawn.aria).toEqual({ enabled: true, label: { description: 'Loss over steps' } })
    await m.unmount()
  })

  it('follows a nested ThemeProvider: its mode names the theme, and overridden tokens register a theme of their own, once', async () => {
    const m = await mountElement(
      <ThemeProvider mode="dark">
        <ThemeProvider mode="light">
          <Chart option={{}} />
        </ThemeProvider>
        <Chart option={{}} />
      </ThemeProvider>,
    )
    expect(made.map((c) => c.theme)).toEqual([themeName('light'), themeName('dark')])
    // The standard themes are the core's own: nothing is registered for them.
    expect(registered).toEqual([])
    await m.unmount()

    const styled = await mountElement(
      <ThemeProvider mode="light" tokens={{ fg: '#123456' }}>
        <Chart option={{}} />
        <Chart option={{}} />
      </ThemeProvider>,
    )
    const names = made.slice(2).map((c) => c.theme)
    expect(names[0]).toMatch(/^omai-light-/)
    expect(names[1]).toBe(names[0])
    expect(registered.map(([n]) => n)).toEqual([names[0]])
    expect(registered[0]?.[1].textStyle.color).toBe('#123456')
    expect(registered[0]?.[1].textStyle.color).not.toBe(tokens('light').fg)
    await styled.unmount()
  })
})
