import { act, type ReactElement, useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { plotArea } from '../figure/figure.ts'
import { invoke, mountElement, resetGlobals } from './_harness.tsx'
import type { ChartProps } from './Chart.tsx'
import { FigureFrame } from './FigureFrame.tsx'
import { ResponsiveChart, type ResponsiveChartProps, type ResponsiveOverlayContext } from './ResponsiveChart.tsx'

// The chart draws on a canvas: stand in for it with a component that records its props and, once mounted, hands the
// component an instance with the option's grid and axes (as ECharts' `getOption` returns them).
const drawn = vi.hoisted(() => ({ props: [] as unknown[] }))
vi.mock('./Chart.tsx', () => ({
  Chart: (props: ChartProps): ReactElement => {
    drawn.props.push(props)
    useEffect(() => {
      const chart = {
        convertToPixel: () => [Number.NaN],
        convertFromPixel: () => [Number.NaN],
        getOption: () => ({
          grid: [{ left: 50, top: 20, width: 300, height: 200 }],
          xAxis: [{ type: 'value', min: 0, max: 10 }],
          yAxis: [{ type: 'value', min: 0, max: 1 }],
        }),
        on: () => undefined,
      }
      props.onInit?.(chart as never)
    }, [props.onInit])
    return <div data-mock-chart="" />
  },
}))

// A ResizeObserver made by hand: a test says what a box measures.
class FakeObserver {
  static all: FakeObserver[] = []
  disconnected = false
  constructor(readonly callback: (entries: { contentRect: { width: number } }[]) => void) {
    FakeObserver.all.push(this)
  }
  observe(): void {}
  disconnect(): void {
    this.disconnected = true
  }
}
const measure = async (width: number): Promise<void> => {
  await act(async () => {
    for (const o of FakeObserver.all) if (!o.disconnected) o.callback([{ contentRect: { width } }])
  })
}

beforeEach(() => {
  FakeObserver.all = []
  drawn.props.length = 0
  ;(globalThis as Record<string, unknown>)['ResizeObserver'] = FakeObserver
})
afterEach(() => {
  delete (globalThis as Record<string, unknown>)['ResizeObserver']
  resetGlobals()
})

const last = (): ChartProps => drawn.props.at(-1) as ChartProps
const sizeOf = (): number[] => [last().figure?.width ?? 0, last().figure?.height ?? 0]

function setup(props: Partial<ResponsiveChartProps> = {}): { readonly build: ReturnType<typeof vi.fn> } {
  const build = vi.fn((fig) => ({ title: { text: `${fig.width}x${fig.height}` } }))
  host.element = <ResponsiveChart build={build} {...props} />
  return { build }
}
const host: { element: ReactElement } = { element: <div /> }

describe('ResponsiveChart measuring', () => {
  it('draws nothing for a first measurement of 0 by 0, then builds for the width it gets', async () => {
    const { build } = setup()
    await mountElement(host.element)
    await measure(0)
    expect([build.mock.calls.length, drawn.props.length]).toEqual([0, 0])
    await measure(400)
    expect(build).toHaveBeenCalledTimes(1)
    // The page figure's aspect, 1.6, gives the height; the option is the builder's for that figure.
    expect(sizeOf()).toEqual([400, 250])
    expect(last().option).toEqual({ title: { text: '400x250' } })
  })

  it('ignores a box that is hidden later and keeps the last size', async () => {
    const { build } = setup()
    await mountElement(host.element)
    await measure(400)
    const calls = build.mock.calls.length
    const renders = drawn.props.length
    await measure(0)
    await measure(Number.NaN)
    expect(build.mock.calls.length).toBe(calls)
    expect(drawn.props.length).toBe(renders)
    expect(sizeOf()).toEqual([400, 250])
    await measure(320)
    expect(sizeOf()).toEqual([320, 200])
  })

  it('rebuilds only when the width changes by a pixel, not on every observer tick', async () => {
    const { build } = setup()
    await mountElement(host.element)
    await measure(400)
    for (let i = 0; i < 5; i++) await measure(400)
    await measure(400.3)
    expect(build).toHaveBeenCalledTimes(1)
    await measure(401)
    expect(build).toHaveBeenCalledTimes(2)
    await measure(401)
    expect(build).toHaveBeenCalledTimes(2)
    expect(sizeOf()[0]).toBe(401)
  })

  it('stops observing on unmount', async () => {
    setup()
    const m = await mountElement(host.element)
    expect(FakeObserver.all.filter((o) => !o.disconnected)).toHaveLength(1)
    await m.unmount()
    expect(FakeObserver.all.every((o) => o.disconnected)).toBe(true)
  })
})

describe('ResponsiveChart shape', () => {
  it('takes narrowAspect below narrowBelow (520 by default), and aspect from there up', async () => {
    setup({ aspect: 2, narrowAspect: 1 })
    await mountElement(host.element)
    await measure(600)
    expect(sizeOf()).toEqual([600, 300])
    await measure(520)
    expect(sizeOf()).toEqual([520, 260])
    await measure(500)
    expect(sizeOf()).toEqual([500, 500])
    await measure(700)
    expect(sizeOf()).toEqual([700, 350])
  })

  it('moves the threshold with narrowBelow, and takes a ratio and a type size', async () => {
    setup({ aspect: '16:9', narrowAspect: 1, narrowBelow: 800, fontSize: 14 })
    await mountElement(host.element)
    await measure(700)
    expect(sizeOf()).toEqual([700, 700])
    expect(last().figure?.fontSize).toBe(14)
    await measure(900)
    expect(sizeOf()).toEqual([900, 506])
  })

  it('keeps the plotting area square, not the figure, with square', async () => {
    setup({ square: true, aspect: 3 })
    await mountElement(host.element)
    await measure(400)
    const fig = last().figure
    if (fig === undefined) throw new Error('no figure')
    const area = plotArea(fig)
    expect(area.width).toBe(area.height)
    expect(fig.width).toBe(400)
    expect(fig.height).not.toBe(fig.width)
  })

  it('passes the colour mode and the chart props on', async () => {
    const onInit = vi.fn()
    setup({ mode: 'dark', renderer: 'svg', group: 'g', onInit })
    await mountElement(host.element)
    await measure(400)
    expect([last().mode, last().renderer, last().group, last().figure?.mode]).toEqual(['dark', 'svg', 'g', 'dark'])
    expect(onInit).toHaveBeenCalledTimes(1)
  })
})

describe('ResponsiveChart overlay', () => {
  it('draws the overlay over the chart once it exists, with the projection and the size, and follows a resize', async () => {
    const seen: ResponsiveOverlayContext[] = []
    setup({
      overlay: (ctx) => {
        seen.push(ctx)
        return <circle r={3} />
      },
    })
    const m = await mountElement(host.element)
    expect(seen).toEqual([])
    await measure(400)
    const ctx = seen.at(-1)
    expect(ctx?.size).toEqual({ width: 400, height: 250 })
    // The projection is the chart's: read from its option, so a drag never asks ECharts.
    expect(ctx?.projection.project(5, 0.5)).toEqual([200, 120])
    expect(ctx?.projection.unproject(200, 120)).toEqual([5, 0.5])
    const svg = m.elements().find((e) => e.tag === 'svg')
    expect(svg?.props['width']).toBe(400)
    await measure(500)
    expect(seen.at(-1)?.size).toEqual({ width: 500, height: 313 })
  })

  it('takes a node as the overlay too, and draws none without one', async () => {
    setup({ overlay: <rect width={4} height={4} /> })
    const m = await mountElement(host.element)
    await measure(400)
    expect(m.elements().some((e) => e.tag === 'rect')).toBe(true)
    setup()
    const bare = await mountElement(host.element)
    await measure(400)
    expect(bare.elements().some((e) => e.tag === 'svg')).toBe(false)
  })
})

describe('ResponsiveChart in a FigureFrame', () => {
  it('takes the frame aspect on the page, and the size the frame gives its content when expanded', async () => {
    const { build } = setup()
    const m = await mountElement(
      <FigureFrame width={600} height={300}>
        {() => host.element}
      </FigureFrame>,
      { browser: true },
    )
    await measure(400)
    expect(sizeOf()).toEqual([400, 200])
    await invoke(
      m.elements().find((e) => e.props['aria-label'] === 'Expand'),
      'onClick',
    )
    // Content that is a function of the size is drawn at the modal's size, which keeps the frame's shape.
    const [w, h] = sizeOf()
    expect(w).toBeGreaterThan(600)
    expect((w as number) / (h as number)).toBeCloseTo(2, 1)
    expect(build.mock.calls.at(-1)?.[0]).toMatchObject({ width: w, height: h })
  })

  it('draws at the frame page size, magnified by the frame, when the content is not a function', async () => {
    setup()
    const m = await mountElement(
      <FigureFrame width={600} height={300}>
        {host.element}
      </FigureFrame>,
      { browser: true },
    )
    await measure(400)
    await invoke(
      m.elements().find((e) => e.props['aria-label'] === 'Expand'),
      'onClick',
    )
    expect(sizeOf()).toEqual([600, 300])
  })
})
