import { ControlPanel, NumberField, Player, SceneFrame, Toggle, usePlayback } from 'mdml-render/react'
import { uNet, uNetOrder, unetLevels } from 'mdml-render/structures'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'U-Net',
  question: 'How do I draw a U-Net or an encoder–decoder, and step through its forward pass?',
  explain:
    '`uNet(levels)` draws the familiar U: convolution blocks as tall as their spatial size and as wide as their channels, downsampling on the left, upsampling on the right, skip connections across (drop them for a plain encoder–decoder) and, for a diffusion model, a time embedding feeding every block. `uNetOrder` lists the blocks in forward order, so the player can light them up one by one.',
}

// The blocks' ids, read aloud.
function describe(id: string): string {
  const [kind, i = '0'] = id.split(':')
  const level = Number(i) + 1
  if (kind === 'enc') return `encoder, level ${level}`
  if (kind === 'dec') return `decoder, level ${level}`
  if (kind === 'skip') return `skip from level ${level}`
  return kind === 'mid' ? 'bottleneck' : 'output'
}

export default function UNetRecipe(): ReactNode {
  const [skips, setSkips] = useState(true)
  const [depth, setDepth] = useState(4)
  const [time, setTime] = useState(false)
  const levels = useMemo(() => unetLevels(128, [32, 64, 128, 256, 512, 1024].slice(0, depth + 1)), [depth])
  const order = useMemo(() => uNetOrder(levels, { skips }), [levels, skips])
  const playback = usePlayback(order.length, { fps: 2 })
  const now = order[Math.min(playback.frame, order.length - 1)] ?? 'out'
  const drawing = useMemo(
    () => uNet(levels, { skips, timeEmbedding: time, outputChannels: 3, active: [now] }),
    [levels, skips, time, now],
  )
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={3}>
        <Toggle label="skip connections (off: encoder–decoder)" value={skips} onChange={setSkips} />
        <NumberField label="levels down" value={depth} onChange={setDepth} min={2} max={5} step={1} />
        <Toggle label="time embedding (diffusion)" value={time} onChange={setTime} />
      </ControlPanel>
      <SceneFrame scene={drawing} filename="u-net" playback={playback} />
      <Player controller={playback} stamp={(f) => `${f + 1} of ${order.length}: ${describe(order[f] ?? 'out')}`} />
    </div>
  )
}
