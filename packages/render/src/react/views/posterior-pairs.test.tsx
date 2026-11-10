import { posteriorPairs } from 'omai-engine/exposition'
import { useState } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { type FakeFrames, fakeFrames, inAct, mountHook, resetGlobals } from '../_harness.tsx'
import { PosteriorPairsPlayer, PosteriorPairsView, stepLabel, usePairsCursor } from './posterior-pairs.tsx'

const draws = (n: number, shift: number) => ({
  a: Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) + shift),
  b: Array.from({ length: n }, (_, i) => Math.cos(i * 1.3) + 0.5 * Math.sin(i * 0.7)),
  c: Array.from({ length: n }, (_, i) => Math.sin(i * 2.1)),
})
const record = posteriorPairs(
  [
    { name: 'nuts', draws: draws(200, 0) },
    { name: 'vi', draws: draws(100, 0.3) },
  ],
  { frames: 5, gridPoints: 12, maxPoints: 30 },
)

let frames: FakeFrames
beforeEach(() => {
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
  resetGlobals()
})

describe('the views on the server', () => {
  it('draw the corner plot with its switches and its title, and the player under it', () => {
    const view = renderToString(<PosteriorPairsView record={record} title="THE PAIRS" />)
    expect(view).toContain('THE PAIRS')
    for (const word of ['methods', 'coordinates', 'nuts', 'vi', 'upper triangle']) expect(view).toContain(word)
    const none = renderToString(<PosteriorPairsView record={record} controls={false} />)
    expect(none).not.toContain('upper triangle')
    const player = renderToString(<PosteriorPairsPlayer record={record} />)
    expect(player).toContain('step ')
    // With a cursor and no handler the page owns the playhead: no player of its own.
    expect(renderToString(<PosteriorPairsPlayer record={record} cursor={50} />)).not.toContain('role="slider"')
  })

  it('label a step as a whole number or to a decimal', () => {
    expect([stepLabel(120), stepLabel(2.5), stepLabel(0.125)]).toEqual(['step 120', 'step 2.5', 'step 0.13'])
  })
})

describe('usePairsCursor', () => {
  it('opens a new record on its end, and on its start when asked', async () => {
    const end = await mountHook(() => usePairsCursor(record))
    expect(end.current.playback.frame).toBe(4)
    expect(end.current.step).toBe(record.cursors[4])
    const start = await mountHook(() => usePairsCursor(record, { startAtFirst: true }))
    expect(start.current.playback.frame).toBe(0)
    expect(start.current.step).toBe(record.cursors[0])
  })

  it('reports the steps the viewer moves the player to, and not the move it makes at mount', async () => {
    const seen: number[] = []
    const m = await mountHook(() => usePairsCursor(record, { onCursor: (s) => seen.push(s) }))
    await inAct(() => m.current.playback.seek(1))
    await inAct(() => m.current.playback.step(1))
    expect(seen.slice(-2)).toEqual([record.cursors[1], record.cursors[2]])
    expect(seen.length).toBeLessThanOrEqual(3)
  })

  it('follows a cursor the page owns and tells it the steps of the player, without echoes', async () => {
    const sent: number[] = []
    const m = await mountHook(() => {
      const [cursor, setCursor] = useState(record.cursors[1] as number)
      const out = usePairsCursor(record, {
        cursor,
        onCursor: (s) => {
          sent.push(s)
          setCursor(s)
        },
      })
      return { ...out, setCursor, cursor }
    })
    // At mount the page's cursor wins: the player is on its frame and has said nothing.
    expect(m.current.playback.frame).toBe(1)
    expect(sent).toEqual([])
    // The page moves between two frames: the player goes to the frame at or before it, and the figure shows the exact step.
    const between = (record.cursors[2] as number) + 0.3
    await inAct(() => m.current.setCursor(between))
    expect(m.current.playback.frame).toBe(2)
    expect(m.current.step).toBe(between)
    expect(sent).toEqual([])
    // The viewer scrubs: the page hears the frame's step.
    await inAct(() => m.current.playback.seek(4))
    expect(sent).toEqual([record.cursors[4]])
    expect(m.current.cursor).toBe(record.cursors[4])
    expect(m.current.step).toBe(record.cursors[4])
  })

  it('plays the cursor along the steps of a page that owns it, to the end', async () => {
    const sent: number[] = []
    const m = await mountHook(() => {
      const [cursor, setCursor] = useState(0)
      return {
        ...usePairsCursor(record, {
          cursor,
          fps: 60,
          onCursor: (s) => {
            sent.push(s)
            setCursor(s)
          },
        }),
        cursor,
      }
    })
    await inAct(() => m.current.playback.play())
    await frames.frame()
    for (let i = 0; i < 20; i++) await frames.frame(100)
    expect(m.current.cursor).toBe(record.cursors.at(-1))
    expect(sent.at(-1)).toBe(record.cursors.at(-1))
    expect(new Set(sent).size).toBe(sent.length)
    expect(sent).toEqual([...sent].sort((x, y) => x - y))
  })
})
