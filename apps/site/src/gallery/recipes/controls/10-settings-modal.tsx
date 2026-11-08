import { bool, choice, dict, int, real, type SpaceValue } from 'omai-engine/space'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, FigureFrame, SettingsButton, SpaceSettings, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CodeBlock } from '../../../shell/CodeBlock.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Settings in a modal',
  question: 'How do I give a figure a settings button whose changes only take effect when I confirm them?',
  explain:
    "A `SettingsButton` is a gear that opens a modal holding a draft of the settings. Edit as many controls as you like: nothing outside changes until Apply, and Cancel, Escape or the close button throw the draft away (a press on the backdrop only closes it while nothing has been touched). `SpaceSettings` is the common case, the draft being an engine `Space` drawn by `SpaceControls`, with Reset to the space's defaults. Hand either to a `FigureFrame` as `settings` and it sits in the toolbar beside export and expand, so a figure with many options keeps its page clean. The applied value is below; open the modal, change several things, cancel, and see that it did not move.",
}

// region
const space = dict({
  window: int(1, 41, { default: 9, label: 'smoothing window' }),
  fit: choice(['moving average', 'median', 'none'], { default: 'moving average', label: 'smoother' }),
  noise: real(0, 1.5, { default: 0.6, label: 'noise' }),
  seed: int(0, 99, { default: 7, label: 'seed' }),
  points: bool({ default: true, label: 'show points' }),
  logY: bool({ default: false, label: 'log y axis' }),
})

type Settings = { window: number; fit: string; noise: number; seed: number; points: boolean; logY: boolean }
const DEFAULT: Settings = { window: 9, fit: 'moving average', noise: 0.6, seed: 7, points: true, logY: false }
const FIGURE = figure({ width: 640, height: 360 })
const N = 160

// A small deterministic generator, so a seed always draws the same noise.
function noiseAt(seed: number, i: number): number {
  let x = (Math.imul(seed + 1, 0x9e3779b1) ^ Math.imul(i + 1, 0x85ebca6b)) >>> 0
  x = Math.imul(x ^ (x >>> 15), 0x2c1b3c6d) >>> 0
  x = Math.imul(x ^ (x >>> 12), 0x297a2d39) >>> 0
  return ((x ^ (x >>> 15)) >>> 0) / 4294967296 - 0.5
}

function smooth(y: readonly number[], window: number, how: string): number[] {
  if (how === 'none' || window <= 1) return [...y]
  const half = Math.floor(window / 2)
  return y.map((_, i) => {
    const w = y.slice(Math.max(0, i - half), Math.min(y.length, i + half + 1))
    if (how === 'median') return [...w].sort((a, b) => a - b)[Math.floor(w.length / 2)] as number
    return w.reduce((a, b) => a + b, 0) / w.length
  })
}

export default function SettingsModal(): ReactNode {
  const mode = useColourMode()
  const [applied, setApplied] = useState<Settings>(DEFAULT)
  const [title, setTitle] = useState('A noisy signal')
  const option = useMemo(() => {
    const x = Array.from({ length: N }, (_, i) => i)
    const truth = x.map((i) => 2.2 + Math.sin(i / 14) + 0.4 * Math.sin(i / 3.3))
    const y = truth.map((v, i) => Math.max(0.05, v + applied.noise * 2 * noiseAt(applied.seed, i)))
    const smoothed = lines(
      [
        {
          x,
          y: smooth(y, applied.window, applied.fit),
          name: applied.fit === 'none' ? 'data' : applied.fit,
          width: 2.5,
        },
      ],
      { mode, colourOffset: 1 },
    )
    return cartesian([...(applied.points ? [points([{ x, y, name: 'data' }], { mode })] : []), smoothed], {
      mode,
      x: { label: 'index' },
      y: { label: 'value', ...(applied.logY ? { scale: 'log' as const } : {}) },
    })
  }, [applied, mode])
  return (
    <div className="flex flex-col gap-3">
      <FigureFrame
        figure={FIGURE}
        title={title}
        filename="settings-modal"
        settings={
          <SpaceSettings
            appearance="toolbar"
            title="Figure settings"
            space={space}
            value={applied as unknown as SpaceValue}
            onApply={(v) => setApplied(v as unknown as Settings)}
            intro="Nothing here changes the figure until you press Apply."
          />
        }
      >
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
      <div className="flex items-center gap-2 text-xs text-(--wb-muted)">
        <span>
          A plain <code>SettingsButton</code> for anything that is not a space (here, a title):
        </span>
        <SettingsButton<string> value={title} onApply={setTitle} title="Figure title" defaults="A noisy signal">
          {(draft, setDraft) => (
            <label className="flex flex-col gap-1 text-xs font-medium">
              title
              <input
                className="h-8 rounded-md border border-(--wb-grid) bg-(--wb-bg) px-2 text-sm text-(--wb-fg)"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
              />
            </label>
          )}
        </SettingsButton>
      </div>
      <CodeBlock title="applied" code={JSON.stringify({ ...applied, title }, null, 2)} />
    </div>
  )
}
// endregion
