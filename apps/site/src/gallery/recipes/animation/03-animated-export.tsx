import { circle, line, rect, type Scene, scene, sceneTimeline, text } from 'mdml-engine/scene'
import { animatedSvg, encodeGif, timelineFrames } from 'mdml-render/animation'
import { ControlPanel, NumberField, Toggle, useColourMode } from 'mdml-render/react'
import { svgDataUrl } from 'mdml-render/svg'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Animated SVG and GIF',
  question: 'How do I save an animation as a file that plays by itself?',
  explain:
    'A `Timeline` of scenes needs no player to be seen: `animatedSvg` writes every frame into one SVG shown in turn by CSS keyframes (the picture here is that file, in an `<img>`), and `timelineFrames` with `encodeGif` makes a GIF. Both run without React, so slides and posts can be built from the same timeline.',
}

// A ball bouncing under gravity, one scene per frame.
function bounce(frames: number): Scene[] {
  return Array.from({ length: frames }, (_, i) => {
    const t = (i / frames) * 2
    const phase = t % 1
    const y = 150 - 120 * (1 - (2 * phase - 1) ** 2)
    const x = 30 + (i / (frames - 1)) * 260
    return scene(320, 180, [
      rect(0, 0, 320, 180, { fill: 'surface' }),
      line(10, 166, 310, 166, { stroke: 'muted' }),
      circle(x, y, 14, { fill: 'tab1' }),
      text(310, 20, `frame ${i + 1}`, { size: 11, fill: 'muted', anchor: 'end' }),
    ])
  })
}

function save(bytes: Uint8Array | string, type: string, name: string): void {
  const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

export default function AnimatedExport(): ReactNode {
  const mode = useColourMode()
  const [fps, setFps] = useState(20)
  const [once, setOnce] = useState(false)
  const [busy, setBusy] = useState(false)
  const timeline = useMemo(() => sceneTimeline(bounce(40), { fps }), [fps])
  const svg = useMemo(() => animatedSvg(timeline, { mode, once }), [timeline, mode, once])
  const button =
    'rounded-md border border-(--wb-grid) px-3 py-1.5 text-xs text-(--wb-fg) hover:bg-(--wb-surface) disabled:opacity-50'
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="frames per second" value={fps} onChange={setFps} min={2} max={40} step={1} />
        <Toggle label="play once" value={once} onChange={setOnce} />
      </ControlPanel>
      {/* The key makes the image restart when the file changes. */}
      <img key={svg.length + fps} src={svgDataUrl(svg)} width={480} height={270} alt="A ball bouncing" />
      <div className="flex gap-2">
        <button type="button" className={button} onClick={() => save(svg, 'image/svg+xml', 'bounce.svg')}>
          Save animated SVG
        </button>
        <button
          type="button"
          className={button}
          disabled={busy}
          onClick={() => {
            setBusy(true)
            timelineFrames(timeline, { mode })
              .then((frames) => save(encodeGif(frames, { fps }), 'image/gif', 'bounce.gif'))
              .finally(() => setBusy(false))
          }}
        >
          {busy ? 'Encoding…' : 'Save GIF'}
        </button>
      </div>
    </div>
  )
}
