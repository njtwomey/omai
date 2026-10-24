import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, heatmap, type RampSpec } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, Select, Toggle, useColourMode } from 'whitebox-render/react'
import { colourBlindCheck, divergingRamp, sequentialRamp, simulateColourVision, tab10 } from 'whitebox-render/theme'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Palettes and ramps',
  question: 'How do I colour categories and continuous values?',
  explain:
    'Categories use tab10 in order (`tab0` to `tab9`, also scene tokens); continuous values use sequential ramps (`viridis`, `magma`, and `blues` and `greys`, which start from the theme surface) or diverging ramps with a pale midpoint, for signed fields with `centre: 0`. `simulateColourVision` and `colourBlindCheck` show how the colours read with a colour-vision deficiency.',
}

const ramps = ['viridis', 'magma', 'blues', 'greys', 'rdbu', 'coolwarm'] as const
const visions = ['normal', 'deuteranopia', 'protanopia', 'tritanopia'] as const
const fig = figure({ width: 420, aspect: 1.3 })
const n = 30
const z = Array.from({ length: n }, (_, i) =>
  Array.from({ length: n }, (_, j) => Math.sin((j - n / 2) / 5) * Math.cos((i - n / 2) / 6)),
)

// Swatches as one strip with hard colour stops.
function Swatches(props: { readonly colours: readonly string[]; readonly label: string }): ReactNode {
  const k = props.colours.length
  const stops = props.colours.map((c, i) => `${c} ${(100 * i) / k}% ${(100 * (i + 1)) / k}%`).join(', ')
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 text-sm text-[var(--wb-muted)]">{props.label}</span>
      <div
        className="rounded"
        title={props.colours.join(' ')}
        style={{ width: 28 * k, height: 22, background: `linear-gradient(to right, ${stops})` }}
      />
    </div>
  )
}

export default function Palettes(): ReactNode {
  const mode = useColourMode()
  const [ramp, setRamp] = useState<(typeof ramps)[number]>('viridis')
  const [vision, setVision] = useState<(typeof visions)[number]>('normal')
  const [signed, setSigned] = useState(false)
  const { seenTab, seenRamp, option } = useMemo(() => {
    const see = (cs: readonly string[]): string[] =>
      vision === 'normal' ? [...cs] : cs.map((c) => simulateColourVision(c, vision))
    const stops = ramp === 'rdbu' || ramp === 'coolwarm' ? divergingRamp(ramp) : sequentialRamp(ramp, mode)
    const seenRamp = see(stops)
    return {
      seenTab: see(tab10),
      seenRamp,
      option: cartesian(
        [heatmap(z, { x: [-1, 1], y: [-1, 1], ramp: seenRamp as RampSpec, mode, ...(signed ? { centre: 0 } : {}) })],
        { mode, figure: fig, aspect: 'equal' },
      ),
    }
  }, [mode, ramp, vision, signed])
  const report = colourBlindCheck(tab10.slice(0, 4))
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Select label="ramp" value={ramp} options={ramps} onChange={setRamp} />
        <Select label="vision" value={vision} options={visions} onChange={setVision} />
        <Toggle label="centre at zero" value={signed} onChange={setSigned} />
      </ControlPanel>
      <div className="flex flex-wrap items-start gap-6">
        <div className="flex flex-col gap-2">
          <Swatches label="tab10" colours={seenTab} />
          <Swatches label={ramp} colours={seenRamp} />
          <p className="max-w-sm text-sm text-[var(--wb-muted)]">
            Least ΔE between tab0 to tab3: normal {report.normal.toFixed(1)}, deuteranopia{' '}
            {report.deuteranopia.toFixed(1)}, protanopia {report.protanopia.toFixed(1)}, tritanopia{' '}
            {report.tritanopia.toFixed(1)}.
          </p>
        </div>
        <FigureFrame figure={fig} filename="ramp">
          <Chart option={option} />
        </FigureFrame>
      </div>
    </div>
  )
}
