/**
 * Player frames for `AgentSteps`: agents moving in a world, frame by frame. Agents with headings are arrows turned to
 * their heading and coloured by it (a flock flying together shares a colour); agents with states take their state's
 * colour (an epidemic's susceptible, exposed, infectious and recovered). The record's series run beneath, the current
 * frame marked.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { AgentSteps } from 'mdml-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { lines } from '../charts/lines.ts'
import { layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

// An arrow pointing up, centred on its middle; rotated to each agent's heading.
const ARROW = 'path://M0,-7 L4.5,6 L0,3 L-4.5,6 Z'

/**
 * The figure aspect (width over height) that fits an agents record: the world over its series.
 *
 * @param record - The record.
 * @returns The aspect.
 */
export function agentAspect(record: AgentSteps): number {
  const world = record.world.width / record.world.height
  return (record.series?.length ?? 0) > 0 ? world / (1 + 0.38) : world
}

/**
 * Agent steps as frames for a player: the world at each recorded frame, with the series beneath.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The frames.
 */
export function agentFrames(record: AgentSteps, options: ViewOptions = {}): ViewFrames {
  const fig = viewFigure(options, { aspect: agentAspect(record) })
  const mode = fig.mode
  const count = Math.max(1, record.frames.length)
  const { width, height } = record.world
  const stateColour = (k: number): string => record.stateColours?.[k] ?? categoricalColour(k)
  const times = record.frames.map((f) => f.t)
  const timeName = record.timeName ?? 'step'
  const option = (frame: number): EChartsOption => {
    const at = Math.min(Math.max(frame, 0), count - 1)
    const f = record.frames[at]
    const data = (f?.x ?? []).map((x, i) => {
      const heading = f?.heading?.[i]
      const state = f?.state?.[i]
      const colour =
        state !== undefined
          ? stateColour(state)
          : heading !== undefined
            ? `hsl(${Math.round((((heading * 180) / Math.PI) % 360) + 360) % 360}, 75%, 62%)`
            : stateColour(0)
      return {
        value: [x, f?.y[i] ?? 0],
        ...(heading === undefined ? {} : { symbolRotate: (heading * 180) / Math.PI - 90 }),
        itemStyle: { color: colour },
      }
    })
    const swarm: SeriesOption = {
      type: 'scatter',
      name: '__agents',
      data,
      symbol: f?.heading === undefined ? 'circle' : ARROW,
      symbolSize: f?.heading === undefined ? 6 : 11,
      silent: true,
      animation: false,
    } as SeriesOption
    const world: Panel = {
      layers: [layer([swarm], { x: [0, width], y: [0, height] })],
      options: {
        x: { show: false, limits: [0, width], splitLine: false },
        y: { show: false, limits: [0, height], splitLine: false },
        aspect: 'equal',
      },
    }
    const series = record.series ?? []
    if (series.length === 0)
      return subplots(
        gridSpec(1, 1, { cells: [{ id: 'world', row: 0, col: 0 }] }),
        { world },
        { figure: fig, legend: false, tooltip: false },
      )
    const below: Panel = {
      layers: [
        lines(
          series.map((s, k) => ({
            x: times,
            y: s.values,
            name: s.name,
            colour:
              record.states?.includes(s.name) === true
                ? stateColour(record.states.indexOf(s.name))
                : categoricalColour(k),
            width: 1.75,
          })),
          { mode },
        ),
        points(
          series.map((s, k) => ({
            x: [times[at] ?? 0],
            y: [s.values[at] ?? 0],
            name: `__now${k}`,
            colour:
              record.states?.includes(s.name) === true
                ? stateColour(record.states.indexOf(s.name))
                : categoricalColour(k),
            size: 8,
          })),
          { mode },
        ),
      ],
      options: {
        x: { label: timeName, limits: [times[0] ?? 0, times.at(-1) ?? 1] },
        y: { label: record.seriesName ?? '' },
      },
    }
    const grid = gridSpec(2, 1, {
      heightRatios: [1, 0.38],
      hspace: 48,
      cells: [
        { id: 'world', row: 0, col: 0 },
        { id: 'below', row: 1, col: 0 },
      ],
    })
    return subplots(grid, { world, below }, { figure: fig, legend: series.length > 1, tooltip: false })
  }
  return {
    count,
    label: (frame) => {
      const f = record.frames[Math.min(Math.max(frame, 0), count - 1)]
      const tail = (record.series ?? [])
        .map((s) => `${s.name} ${(s.values[Math.min(Math.max(frame, 0), count - 1)] ?? 0).toFixed(2)}`)
        .join(' · ')
      return `${timeName} ${f?.t ?? 0}${tail === '' ? '' : ` · ${tail}`}`
    },
    option,
  }
}
