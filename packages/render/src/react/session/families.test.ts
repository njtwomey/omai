import type { Run } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import type { Layer } from '../../charts/plot.ts'
import { figure } from '../../figure/index.ts'
import { curveFamily, eventSteps, familyOption, hasFamily, intervalStrip, logAxes, slopeLayers } from './families.ts'

// A run of one metric (and a few more) logged at steps 1 to 4.
const metric = (name: string, values: number[]): Run['events'] =>
  values.map((value, i) => ({ type: 'metric' as const, runId: 'r', step: i + 1, name, value }))
const run = (loss: number[], extra: Run['events'] = []): Run => ({
  schema: 'omai.run/1',
  events: [...metric('loss', loss), ...extra],
})
const names = (l: { series: readonly unknown[] }): (string | undefined)[] =>
  l.series.map((s) => (s as { name?: string }).name)

describe('curve families', () => {
  it('knows when a panel asks for a family', () => {
    expect(hasFamily({})).toBe(false)
    expect(hasFamily({ log: 'y' })).toBe(true)
    expect(hasFamily({ bands: [] })).toBe(false)
    expect(logAxes('xy')).toEqual({ x: true, y: true })
    expect(logAxes(undefined)).toEqual({ x: false, y: false })
  })

  it('draws many runs faint with their mean in bold', () => {
    const runs = [
      { label: 'a', run: run([4, 3, 2, 1]) },
      { label: 'b', run: run([2, 3, 4, 5]) },
    ]
    const [curves] = curveFamily(runs, ['loss'], { mode: 'light', runs: true })
    expect(names(curves as never)).toEqual(['__run0-0', '__run0-1', 'mean'])
    const mean = ((curves as Layer).series[2] as { data: [number, number][] }).data.map((p) => p[1])
    expect(mean).toEqual([3, 3, 3, 3])
  })

  it('keeps one colour per variant when the runs are not repeats', () => {
    const runs = [
      { label: 'a', run: run([4, 3, 2, 1]) },
      { label: 'b', run: run([2, 3, 4, 5]) },
    ]
    const [curves] = curveFamily(runs, ['loss'], { mode: 'light' })
    expect(names(curves as never)).toEqual(['a', 'b'])
  })

  it('adds bands beneath, a truth line, slopes and event marks over the curves', () => {
    const events = metric('alarm', [0, 0, 1, 0])
    const r: Run = {
      schema: 'omai.run/1',
      events: [
        ...metric('loss', [4, 3, 2, 1]),
        ...metric('lo', [3, 2, 1, 0.5]),
        ...metric('hi', [5, 4, 3, 2]),
        ...events,
      ],
    }
    const layers = curveFamily([{ label: '', run: r }], ['loss'], {
      mode: 'light',
      bands: [{ path: 'loss', lower: 'lo', upper: 'hi' }],
      truth: { value: 2, label: 'truth' },
      slopes: [{ slope: -1 }],
      events: [{ path: 'alarm', label: 'alarm' }],
      log: 'xy',
    })
    // The curves, one band, then the truth, the slope and the alarm.
    expect(layers).toHaveLength(5)
    expect(names(layers[2] as never)).toEqual(['truth'])
    expect(names(layers[3] as never)).toEqual(['slope -1'])
    expect(names(layers[4] as never)).toEqual(['alarm'])
    expect(eventSteps(r, 'alarm')).toEqual([3])
  })

  it('leaves non-positive points off log axes', () => {
    const [curves] = curveFamily([{ label: '', run: run([1, 0, 2, 3]) }], ['loss'], { mode: 'light', log: 'y' })
    expect(((curves as Layer).series[0] as { data: unknown[] }).data).toHaveLength(3)
  })

  it('anchors a reference slope at the first positive point', () => {
    const [l] = slopeLayers([{ slope: -2 }], { x: [1, 2], y: [8, 4] }, [1, 4], 'light')
    const data = ((l as Layer).series[0] as { data: [number, number][] }).data
    expect(data[0]).toEqual([1, 8])
    expect(data[1]?.[1]).toBe(8 * 4 ** -2)
  })

  it('builds the chart with log axes and a marker at the viewed step', () => {
    const option = familyOption([{ label: '', run: run([4, 3, 2, 1]) }], ['loss'], {
      log: 'y',
      t: 2,
      mode: 'light',
      figure: figure({ width: 300, height: 200 }),
    }) as { yAxis: { type: string }[]; series: { name?: string }[] }
    expect(option.yAxis[0]?.type).toBe('log')
    expect(option.series.some((s) => s.name === '__step')).toBe(true)
  })
})

describe('interval strip', () => {
  const input = {
    steps: [1, 2, 3, 4],
    centre: [1, 2, 3, 4],
    lower: [0, 1, 2.9, 3],
    upper: [2, 3, 4, 5],
    truth: 1.5,
  }

  it('counts the covered and the missed among the trials seen, and fades those ahead', () => {
    const [bars] = intervalStrip({ ...input, t: 3 }, 'light')
    // Trials 1 and 2 cover 1.5, trial 3 misses (lower 2.9), trial 4 is ahead.
    expect(names(bars as never)).toEqual(['covered 2', 'missed 1', '__ahead'])
  })

  it('draws a per-trial truth as marks instead of a line', () => {
    const layers = intervalStrip({ ...input, truth: [2, 2, 2, 2], t: 4 }, 'light')
    expect(layers.some((l) => names(l).includes('truth'))).toBe(true)
  })
})
