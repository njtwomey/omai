import { type ReactElement, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { lines } from '../charts/lines.ts'
import { withMarginals } from '../figure/grid.ts'
import { inAct, mountElement, resetGlobals } from './_harness.tsx'
import { Chart } from './Chart.tsx'
import type { FrameSize } from './FigureFrame.tsx'
import { FigureGrid } from './FigureGrid.tsx'
import { PixelRatioProvider, usePixelRatio } from './pixel-ratio.tsx'

// ECharts needs a real canvas: an `init` that records the options it is given, and a frame size a test sets by hand
// (an expanded FigureFrame is the only thing that provides one).
const inits = vi.hoisted(() => ({ made: [] as { renderer: unknown; options: Record<string, unknown> }[] }))
const frame = vi.hoisted(() => ({ current: undefined as FrameSize | undefined }))
vi.mock('./echarts.ts', () => ({
  loadEcharts: () => ({
    init: (_el: unknown, _theme: string, options: Record<string, unknown>) => {
      inits.made.push({ renderer: options['renderer'], options })
      return {
        setOption: () => undefined,
        on: () => undefined,
        off: () => undefined,
        dispose: () => undefined,
        isDisposed: () => false,
        resize: () => undefined,
      }
    },
    connect: () => undefined,
    registerTheme: () => undefined,
  }),
}))
vi.mock('./FigureFrame.tsx', async (original) => ({
  ...(await original<typeof import('./FigureFrame.tsx')>()),
  useFrameSize: (): FrameSize | undefined => frame.current,
}))

beforeEach(() => {
  ;(globalThis as Record<string, unknown>)['devicePixelRatio'] = 2
})
afterEach(() => {
  inits.made.length = 0
  frame.current = undefined
  delete (globalThis as Record<string, unknown>)['devicePixelRatio']
  resetGlobals()
})

// The devicePixelRatio of each instance made (undefined: the option is absent and ECharts keeps the browser's).
const ratios = (): unknown[] => inits.made.map((m) => m.options['devicePixelRatio'])

const host: { set: (r: number | undefined) => void; redraw: () => void } = {
  set: () => undefined,
  redraw: () => undefined,
}
function Host(): ReactElement {
  const [ratio, setRatio] = useState<number | undefined>(1)
  const [, again] = useState(0)
  host.set = setRatio
  host.redraw = () => again((n) => n + 1)
  return <Chart option={{}} {...(ratio === undefined ? {} : { ratio })} />
}

describe('the pixel ratio of a Chart', () => {
  it('passes no devicePixelRatio when nothing sets one', async () => {
    await mountElement(<Chart option={{}} />)
    expect(inits.made).toHaveLength(1)
    expect('devicePixelRatio' in (inits.made[0]?.options ?? {})).toBe(false)
  })

  it('is the browser ratio times the `ratio` prop', async () => {
    await mountElement(<Chart option={{}} ratio={1.5} />)
    expect(ratios()).toEqual([3])
  })

  it('takes the ratio of a PixelRatioProvider, the innermost winning', async () => {
    await mountElement(
      <PixelRatioProvider ratio={1.25}>
        <Chart option={{}} />
        <PixelRatioProvider ratio={3}>
          <Chart option={{}} />
        </PixelRatioProvider>
      </PixelRatioProvider>,
    )
    expect(ratios()).toEqual([2.5, 6])
  })

  it('lets the prop win over the provider and the frame', async () => {
    frame.current = { width: 800, height: 500, expanded: true, scale: 2 }
    await mountElement(
      <PixelRatioProvider ratio={3}>
        <Chart option={{}} ratio={0.5} />
      </PixelRatioProvider>,
    )
    expect(ratios()).toEqual([1])
  })

  it('composes the expanded frame scale with the provider ratio, and keeps the frame alone as before', async () => {
    frame.current = { width: 800, height: 500, expanded: true, scale: 2 }
    await mountElement(<Chart option={{}} />)
    expect(ratios()).toEqual([4])
    await mountElement(
      <PixelRatioProvider ratio={1.5}>
        <Chart option={{}} />
      </PixelRatioProvider>,
    )
    expect(ratios()).toEqual([4, 6])
    // A frame that is not magnified (no scale) leaves the provider's ratio as it is.
    frame.current = { width: 800, height: 500, expanded: false }
    await mountElement(
      <PixelRatioProvider ratio={1.5}>
        <Chart option={{}} />
      </PixelRatioProvider>,
    )
    expect(ratios()).toEqual([4, 6, 3])
  })

  it('does not clamp what the host gives', async () => {
    await mountElement(<Chart option={{}} ratio={8} />)
    expect(ratios()).toEqual([16])
  })

  it('remakes the instance when the ratio changes, and not otherwise', async () => {
    await mountElement(<Host />)
    expect(ratios()).toEqual([2])
    await inAct(host.redraw)
    expect(inits.made).toHaveLength(1)
    await inAct(() => host.set(2))
    expect(ratios()).toEqual([2, 4])
    await inAct(() => host.set(undefined))
    expect(inits.made).toHaveLength(3)
    expect('devicePixelRatio' in (inits.made[2]?.options ?? {})).toBe(false)
  })

  it('is passed to an SVG chart too (vectors ignore it)', async () => {
    await mountElement(<Chart option={{}} renderer="svg" ratio={2} />)
    expect(inits.made[0]?.renderer).toBe('svg')
  })

  it('reaches the chart of a FigureGrid', async () => {
    const grid = withMarginals()
    await mountElement(<FigureGrid grid={grid} panels={{ main: { layers: [lines([{ y: [1, 2] }])] } }} ratio={1.5} />)
    expect(ratios()).toEqual([3])
  })
})

const seen: (number | undefined)[] = []
function Read(): null {
  seen.push(usePixelRatio())
  return null
}

describe('usePixelRatio', () => {
  it('reads the nearest provider, and undefined without one', async () => {
    seen.length = 0
    await mountElement(
      <>
        <Read />
        <PixelRatioProvider ratio={2}>
          <Read />
          <PixelRatioProvider ratio={0.5}>
            <Read />
          </PixelRatioProvider>
        </PixelRatioProvider>
      </>,
    )
    expect(seen.slice(0, 3)).toEqual([undefined, 2, 0.5])
  })
})
