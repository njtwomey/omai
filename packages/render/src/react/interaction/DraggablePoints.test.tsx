import type { Point2 } from 'omai-engine/numerics/geometry'
import { type ReactElement, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { type FoundElement, inAct, invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { DraggablePoints, type DraggablePointsProps } from './DraggablePoints.tsx'
import { InteractionOverlay } from './InteractionOverlay.tsx'
import { Probe } from './Probe.tsx'
import { linearProjection } from './projection.ts'

let restore: (() => void) | undefined
afterEach(() => {
  restore?.()
  restore = undefined
  resetGlobals()
})

// A 100 by 100 pixel figure over 0..10 on both axes: ten pixels a unit, y up.
const projection = linearProjection({ x: [0, 10], y: [0, 10], width: 100, height: 100 })
const log: Point2[][] = []
const probed: (Point2 | null)[] = []

function Host(props: {
  readonly start?: Point2[]
  readonly rest?: Partial<DraggablePointsProps>
  readonly probe?: boolean
}): ReactElement {
  const [points, setPoints] = useState<Point2[]>(props.start ?? [])
  return (
    <InteractionOverlay projection={projection} width={100} height={100}>
      {props.probe === true ? <Probe onProbe={(p) => probed.push(p)} /> : null}
      <DraggablePoints
        points={points}
        onChange={(p) => {
          log.push(p)
          setPoints(p)
        }}
        {...props.rest}
      />
    </InteractionOverlay>
  )
}

type Mount = Awaited<ReturnType<typeof mountElement>>
// The overlay's one catching rectangle: transparent, as large as the figure.
const catcher = (m: Mount): FoundElement[] =>
  m.elements().filter((e) => e.tag === 'rect' && e.props['fill'] === 'transparent')
const handles = (m: Mount): FoundElement[] => m.elements().filter((e) => e.props['role'] === 'button')

const listeners: Record<string, () => void> = {}
const at = (x: number, y: number): object => ({
  clientX: x,
  clientY: y,
  pointerId: 1,
  stopPropagation: () => undefined,
  currentTarget: {
    setPointerCapture: () => undefined,
    hasPointerCapture: () => true,
    style: {},
    addEventListener: (name: string, fn: () => void) => {
      listeners[name] = fn
    },
    removeEventListener: () => undefined,
  },
})

async function mount(element: ReactElement): Promise<Mount> {
  log.length = 0
  probed.length = 0
  const m = await mountElement(element)
  const proto = Object.getPrototypeOf(m.elements()[0]?.node) as { getBoundingClientRect: () => object }
  const original = proto.getBoundingClientRect
  restore = () => {
    proto.getBoundingClientRect = original
  }
  proto.getBoundingClientRect = () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 })
  return m
}

describe('DraggablePoints place', () => {
  it('draws no catching rectangle without a tool that listens for empty space', async () => {
    const m = await mount(<Host start={[[5, 5]]} />)
    expect(catcher(m)).toHaveLength(0)
    expect(handles(m)).toHaveLength(1)
    await m.unmount()
  })

  it('a press on empty space makes the point when there is none, and moves it when there is one', async () => {
    const m = await mount(<Host rest={{ place: true }} />)
    expect(catcher(m)).toHaveLength(1)
    await invoke(catcher(m)[0], 'onPointerDown', at(30, 20))
    expect(log.at(-1)).toEqual([[3, 8]])
    await invoke(catcher(m)[0], 'onPointerDown', at(70, 90))
    expect(log.at(-1)).toEqual([[7, 1]])
    expect(handles(m)).toHaveLength(1)
    await m.unmount()
  })

  it('the point follows the pointer until release, kept where its constraint allows', async () => {
    const m = await mount(<Host rest={{ place: true, constrain: ([x, y]) => [Math.min(x, 5), y] }} />)
    await invoke(catcher(m)[0], 'onPointerDown', at(30, 20))
    expect(log.at(-1)).toEqual([[3, 8]])
    await inAct(() => (listeners['pointermove'] as (e: object) => void)({ clientX: 90, clientY: 50 }))
    expect(log.at(-1)).toEqual([[5, 5]])
    await m.unmount()
  })

  it('without `place` or `add` a press on empty space does nothing; `add` still makes more points', async () => {
    const m = await mount(<Host start={[[5, 5]]} rest={{ add: true }} />)
    await invoke(catcher(m)[0], 'onPointerDown', at(10, 10))
    expect(log.at(-1)).toEqual([
      [5, 5],
      [1, 9],
    ])
    await m.unmount()
  })

  it('stacks with a Probe: both hear the pointer through the one rectangle', async () => {
    const m = await mount(<Host probe rest={{ place: true }} />)
    expect(catcher(m)).toHaveLength(1)
    await invoke(catcher(m)[0], 'onPointerMove', at(40, 60))
    expect(probed.at(-1)).toEqual([4, 4])
    await invoke(catcher(m)[0], 'onPointerDown', at(40, 60))
    expect(log.at(-1)).toEqual([[4, 4]])
    await invoke(catcher(m)[0], 'onPointerLeave', at(40, 60))
    expect(probed.at(-1)).toBeNull()
    await m.unmount()
  })

  it('a handle is above the rectangle: dragging it moves the point without placing another', async () => {
    const m = await mount(<Host start={[[5, 5]]} rest={{ place: true }} />)
    const order = m.elements().map((e) => e.props['role'] === 'button' || e.props['fill'] === 'transparent')
    // The rectangle comes before the handle in the tree, so the handle is drawn above it.
    expect(m.elements().findIndex((e) => e.props['fill'] === 'transparent')).toBeLessThan(
      m.elements().findIndex((e) => e.props['role'] === 'button'),
    )
    expect(order.length).toBeGreaterThan(0)
    await invoke(handles(m)[0], 'onPointerDown', at(50, 50))
    await invoke(handles(m)[0], 'onPointerMove', at(70, 30))
    expect(log.at(-1)).toEqual([[7, 7]])
    await m.unmount()
  })
})
