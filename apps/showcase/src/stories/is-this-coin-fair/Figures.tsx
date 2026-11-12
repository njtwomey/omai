import { annotations, type Layer, lines, type Panel, subplots } from 'omai-render/charts'
import { type Figure, gridSpec } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result, Stage } from './compute.ts'
import { STAGES, TOLERANCE } from './data.ts'

const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`

/** The title of a panel: how many flips, and once they are known, how many of them were heads (on a second line when narrow). */
function titleOf(flips: number, stage: Stage | undefined, narrow: boolean): string {
  const seen = flips === 0 ? 'no flips yet' : plural(flips, 'flip', 'flips')
  if (stage === undefined || flips === 0) return seen
  return `${seen}${narrow ? '\n' : ', '}${plural(stage.heads, 'head', 'heads')}`
}

/** The stripe that marks 'fair' in every panel. */
const fairWindow = (mode: 'light' | 'dark'): Layer =>
  annotations({ regions: [{ x: [TOLERANCE[0], TOLERANCE[1]], colour: 'fgSecondary' }] }, { mode })

/**
 * The strip of six panels, one per moment: the same axes (the chance of heads from 0 to 1) in each, with the stripe that is
 * 'fair'. Before play the panels are empty frames with a dashed line for the flat belief everyone starts from; with a result
 * each holds the posterior curve, shaded where it holds 90% of the belief, and the last also marks the true bias.
 *
 * @param props - The result, once there is one.
 * @returns The chart.
 */
export function StripFigure(props: { readonly result: Result | undefined }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      // Six panels in two rows of three, or three rows of two on a narrow screen.
      const narrow = figure.width < 560
      const columns = narrow ? 2 : 3
      const grid = gridSpec(narrow ? 3 : 2, columns, { wspace: narrow ? 28 : 34, hspace: narrow ? 72 : 56 })
      const panels: Record<string, Panel> = {}
      STAGES.forEach((flips, i) => {
        const stage = result?.stages[i]
        const top = stage === undefined ? 2 : Math.max(...stage.density) * 1.18
        const layers = [fairWindow(mode)]
        if (result === undefined || stage === undefined) {
          layers.push(lines([{ x: [0, 1], y: [1, 1], dash: 'dashed', colour: 'fgSecondary', width: 1.5 }], { mode }))
        } else {
          const core = result.grid.map((g) => (g >= stage.lo && g <= stage.hi ? 1 : 0))
          layers.push(
            lines([{ x: result.grid, y: stage.density, area: 0.12, width: 2.2 }], { mode }),
            lines(
              [
                {
                  x: result.grid.filter((_, k) => core[k] === 1),
                  y: stage.density.filter((_, k) => core[k] === 1),
                  area: 0.42,
                  width: 0,
                },
              ],
              { mode },
            ),
          )
          if (i === STAGES.length - 1)
            layers.push(
              lines(
                [{ x: [result.truth, result.truth], y: [0, top], dash: 'dashed', colour: 'fgSecondary', width: 1.6 }],
                {
                  mode,
                },
              ),
              annotations(
                { labels: narrow ? [] : [{ at: [Math.min(result.truth + 0.14, 0.88), top * 0.6], text: 'true bias' }] },
                { mode },
              ),
            )
        }
        const row = Math.floor(i / columns)
        const lastRow = row === Math.ceil(STAGES.length / columns) - 1
        const id = `r${row}c${i % columns}`
        panels[id] = {
          layers,
          options: {
            title: titleOf(flips, stage, narrow),
            x: { label: lastRow ? 'chance of heads' : '', limits: [0, 1] as [number, number] },
            y: { label: '', limits: [0, top] as [number, number], show: false, splitLine: false },
          },
        }
      })
      return subplots(grid, panels, { figure, mode, legend: false, tooltip: false })
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.75} narrowAspect={0.95} narrowBelow={560} build={build} />
}

/**
 * The coin and the question, drawn before anything runs: a coin with a question mark on it, and the line of every possible
 * bias, from 'never heads' to 'always heads', with 'fair' marked in the middle and a question mark on the line where the real
 * bias lies.
 *
 * @returns The picture.
 */
export function CoinArt(): ReactNode {
  return (
    <svg
      viewBox="0 0 560 190"
      role="img"
      aria-label="A coin with a question mark on it, and a line from never heads to always heads with a question mark on it"
      className="block h-auto w-full text-fg"
    >
      <title>A coin and a line of possible biases</title>
      <circle cx="96" cy="95" r="70" fill="var(--wb-surface)" stroke="var(--wb-accent)" strokeWidth="4" />
      <circle cx="96" cy="95" r="56" fill="none" stroke="var(--wb-accent)" strokeWidth="1.5" strokeDasharray="3 5" />
      <text
        x="96"
        y="118"
        textAnchor="middle"
        fontSize="64"
        fontWeight="700"
        fill="var(--wb-accent)"
        fontFamily="Georgia, serif"
      >
        ?
      </text>
      <g fontFamily="system-ui, sans-serif" fontSize="13" fill="var(--wb-muted)">
        <line x1="210" y1="100" x2="540" y2="100" stroke="var(--wb-muted)" strokeWidth="2" />
        <line x1="210" y1="92" x2="210" y2="108" stroke="var(--wb-muted)" strokeWidth="2" />
        <line x1="540" y1="92" x2="540" y2="108" stroke="var(--wb-muted)" strokeWidth="2" />
        <rect x="358.5" y="86" width="33" height="28" rx="3" fill="var(--wb-fg)" fillOpacity="0.14" />
        <line x1="375" y1="84" x2="375" y2="116" stroke="var(--wb-fg)" strokeOpacity="0.55" strokeWidth="2" />
        <text x="210" y="136" textAnchor="start">
          never heads
        </text>
        <text x="540" y="136" textAnchor="end">
          always heads
        </text>
        <text x="375" y="136" textAnchor="middle">
          fair
        </text>
        <text
          x="460"
          y="66"
          textAnchor="middle"
          fontSize="38"
          fontWeight="700"
          fill="var(--wb-accent)"
          fontFamily="Georgia, serif"
        >
          ?
        </text>
        <text x="460" y="84" textAnchor="middle">
          the real bias is somewhere here
        </text>
      </g>
    </svg>
  )
}

/**
 * The hundred flips, in order, as small marks in rows of twenty (filled for heads, hollow for tails); the last flip of each stage
 * the strip looks at (the 1st, 3rd, 10th, 30th and 100th) is ringed, so each panel of the strip is everything up to a ringed mark.
 *
 * @param props - The flips.
 * @returns The tape.
 */
export function FlipTape(props: { readonly flips: readonly number[] }): ReactNode {
  const { flips } = props
  const marks = flips.map((f, at) => ({ at, heads: f === 1, ringed: STAGES.includes(at + 1) }))
  return (
    <div
      className="mx-auto grid max-w-[34rem] grid-cols-[repeat(20,minmax(0,1fr))] gap-[5px] p-2"
      role="img"
      aria-label="The 100 flips, in order"
    >
      {marks.map((mark) => (
        <span
          key={mark.at}
          className={`aspect-square rounded-full border border-accent ${mark.heads ? 'bg-accent' : 'bg-transparent'} ${mark.ringed ? 'outline outline-1 outline-offset-2 outline-fg' : ''}`}
        />
      ))}
    </div>
  )
}
