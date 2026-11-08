import { smoothSolvers } from 'omai-engine/optim'
import { defaults, type SpaceValue } from 'omai-engine/space'
import { ControlPanel, ModelSelector, type SlotKnobs, SlotVariants } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import { CodeBlock } from '../../../shell/CodeBlock.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Model selector and slot variants',
  question: 'How do I list the options of a slot, each with its own settings: choose one, or compare them all?',
  explain:
    "`ModelSelector` lists the options of a union space (the models of a task, the solvers of a slot); pressing a name chooses it, pressing its gear opens that option's knobs in a popover, and the knobs set for an option are remembered while you try another. `SlotVariants` is the same list for a comparison: nothing is chosen, every option that applies is a variant (its colour is the colour its curve is drawn in), and each keeps the knobs you give it. An option a rule rules out is listed greyed, with the reason as its title. Here the solvers of `smoothSolvers`, with Newton ruled out as if the objective were not twice differentiable.",
}

// region
const names = smoothSolvers.names
const describe = (name: string): string => smoothSolvers.modeOf({ kind: name }).description ?? ''
const descriptions = Object.fromEntries(names.map((n) => [n, describe(n)]))
const why = (n: string): string | undefined => (n === 'newton' ? 'needs a twice-differentiable objective' : undefined)
const admissible = (n: string): boolean => n !== 'newton'

export default function SlotSelectorsRecipe(): ReactNode {
  const [chosen, setChosen] = useState<SpaceValue>(() => defaults(smoothSolvers.space))
  const [knobs, setKnobs] = useState<SlotKnobs>({})
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <ModelSelector
          space={smoothSolvers.space}
          value={chosen}
          onChange={setChosen}
          label="one solver, chosen"
          descriptions={descriptions}
          admissible={admissible}
          why={why}
        />
        <SlotVariants
          space={smoothSolvers.space}
          knobs={knobs}
          onChange={setKnobs}
          label="every solver, compared"
          descriptions={descriptions}
          admissible={admissible}
          why={why}
        />
      </ControlPanel>
      <div className="grid gap-3 md:grid-cols-2">
        <CodeBlock title="chosen" code={JSON.stringify(chosen, null, 2)} />
        <CodeBlock title="knobs held for the variants" code={JSON.stringify(knobs, null, 2)} />
      </div>
    </div>
  )
}
// endregion
