import type { ReactNode } from 'react'
import { useColourMode } from 'whitebox-render/react'
import { colourBlindCheck, divergingRamp, sequentialRamp, tab10 } from 'whitebox-render/theme'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Palettes and ramps',
  question: 'Which colours does whitebox use for categories and for continuous values?',
  explain:
    'Categories use tab10 in order (`tab0` to `tab9`, also scene tokens); continuous values use sequential ramps (`viridis`, `magma`, and `blues` and `greys`, which start from the theme surface) or diverging ramps with a pale midpoint. `colourBlindCheck` measures how far apart colours stay under simulated colour-vision deficiencies.',
}

function Swatches(props: { readonly colours: readonly string[]; readonly label: string }): ReactNode {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 text-sm text-[var(--wb-muted)]">{props.label}</span>
      <div className="flex overflow-hidden rounded">
        {props.colours.map((c) => (
          <div key={c} title={c} style={{ background: c, width: 32, height: 24 }} />
        ))}
      </div>
    </div>
  )
}

export default function Palettes(): ReactNode {
  const mode = useColourMode()
  const report = colourBlindCheck(tab10.slice(0, 4))
  return (
    <div className="flex flex-col gap-2">
      <Swatches label="tab10" colours={tab10} />
      {(['viridis', 'magma', 'blues', 'greys'] as const).map((name) => (
        <Swatches key={name} label={name} colours={sequentialRamp(name, mode)} />
      ))}
      <Swatches label="rdbu" colours={divergingRamp('rdbu')} />
      <Swatches label="coolwarm" colours={divergingRamp('coolwarm')} />
      <p className="text-sm text-[var(--wb-muted)]">
        Least ΔE between tab0 to tab3: normal {report.normal.toFixed(1)}, deuteranopia {report.deuteranopia.toFixed(1)},
        protanopia {report.protanopia.toFixed(1)}, tritanopia {report.tritanopia.toFixed(1)}.
      </p>
    </div>
  )
}
