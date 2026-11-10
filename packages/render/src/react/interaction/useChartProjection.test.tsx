import { type ReactElement, useEffect } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { inAct, mountElement, mountHook, resetGlobals } from '../_harness.tsx'
import { DraggablePoints } from './DraggablePoints.tsx'
import { InteractionOverlay } from './InteractionOverlay.tsx'
import type { PixelConverter } from './projection.ts'
import { useChartProjection } from './useChartProjection.ts'

afterEach(resetGlobals)

// A chart whose grid the test moves: the option is read again at each `finished`, as a redrawn chart's would be.
function fakeChart(): { readonly chart: PixelConverter; width: number; finish(): void } {
  const handlers: (() => void)[] = []
  const state = {
    width: 100,
    finish: (): void => {
      for (const h of handlers) h()
    },
    chart: {
      convertToPixel: () => [Number.NaN, Number.NaN],
      convertFromPixel: () => [Number.NaN, Number.NaN],
      getOption: () => ({
        grid: [{ left: 0, top: 0, width: state.width, height: 100, outerBoundsMode: 'none' }],
        xAxis: [{ type: 'value', min: 0, max: 10 }],
        yAxis: [{ type: 'value', min: 0, max: 10 }],
      }),
      on: (_name: string, h: () => void) => {
        handlers.push(h)
      },
    } satisfies PixelConverter,
  }
  return state
}

// Made once, outside the host: React skips an element whose props did not change.
const tools = <DraggablePoints points={[[5, 5]]} onChange={() => undefined} />

function Host(props: { readonly chart: PixelConverter }): ReactElement {
  const { chart } = props
  const { projection, onInit } = useChartProjection()
  useEffect(() => onInit(chart), [onInit, chart])
  return projection === null ? (
    <g />
  ) : (
    <InteractionOverlay projection={projection} width={200} height={100}>
      {tools}
    </InteractionOverlay>
  )
}

describe('useChartProjection', () => {
  it('gives a new projection after a redraw moved the box, so a context holding it changes', async () => {
    const fake = fakeChart()
    const m = await mountHook(() => useChartProjection())
    await inAct(() => m.current.onInit(fake.chart))
    const first = m.current.projection
    expect(first?.project(5, 5)).toEqual([50, 50])
    await inAct(fake.finish)
    // Nothing moved: the same projection.
    expect(m.current.projection).toBe(first)
    fake.width = 200
    await inAct(fake.finish)
    expect(m.current.projection).not.toBe(first)
    expect(m.current.projection?.project(5, 5)).toEqual([100, 50])
    expect(m.current.projection?.rect?.width).toBe(200)
    await m.unmount()
  })

  it('tools held as a stable element follow the chart when it redraws', async () => {
    const fake = fakeChart()
    const m = await mountElement(<Host chart={fake.chart} />)
    const cx = (): unknown => m.elements().find((e) => e.props['role'] === 'button')?.props['cx']
    expect(cx()).toBe(50)
    fake.width = 200
    await inAct(fake.finish)
    expect(cx()).toBe(100)
    await m.unmount()
  })
})
