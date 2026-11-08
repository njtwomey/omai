import { classificationLosses, penalties } from 'omai-engine/losses'
import { defaults, dict, type SpaceValue } from 'omai-engine/space'
import {
  Choice,
  ControlPanel,
  SpaceControls,
  settleAdmissible,
  type TableSlot,
  tableRules,
  tableSlot,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CodeBlock } from '../../../shell/CodeBlock.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Validity greying',
  question: 'How do I show the options that do not apply to the current choices, and why they do not?',
  explain:
    "An option of a slot that the other choices rule out is drawn greyed, in place, with the reason as its title (hover it), not hidden: the reader sees what exists and what it would need. The rule is read from the option tables' own properties: `tableSlot(table, (properties) => reason | undefined)` names what stands in the way and `tableRules` turns the slots into the `admissible` and `why` that `SpaceControls` takes (`modeSlot` does the same from a mode table's `needs`, and `settleAdmissible` moves a value off an option that has just gone grey). Switch the solver family: a gradient method needs a smooth objective, so the hinge loss and the l1 penalties go grey; a sparse solver needs a penalty that zeroes coefficients, so ridge and none go grey.",
}

// region
const space = dict({ loss: classificationLosses.space, penalty: penalties.space })

type Family = 'gradient' | 'sparse'

// What a solver family needs of the objective, read from the tables' properties; the reason is the option's title.
function slots(family: Family): Record<string, TableSlot> {
  return family === 'gradient'
    ? {
        loss: tableSlot(classificationLosses, (p) =>
          p.smooth ? undefined : 'not smooth: a gradient needs a direction at every point',
        ),
        penalty: tableSlot(penalties, (p) =>
          p.smooth ? undefined : 'not smooth: its kink at zero stalls a gradient method',
        ),
      }
    : { penalty: tableSlot(penalties, (p) => (p.sparse ? undefined : 'does not drive coefficients to exactly zero')) }
}

export default function ValidityGreyingRecipe(): ReactNode {
  const [family, setFamily] = useState<Family>('gradient')
  const [value, setValue] = useState<SpaceValue>(() => defaults(space))
  const rule = useMemo(() => tableRules(slots(family)), [family])
  // Changing the family can rule out the option in use: move it to the first that applies.
  const choose = (next: Family): void => {
    setFamily(next)
    setValue(settleAdmissible(space, value, slots(next)))
  }
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="solver family"
          value={family}
          options={[
            { value: 'gradient', label: 'gradient', description: 'follows the gradient: needs a smooth objective' },
            { value: 'sparse', label: 'sparse', description: 'soft-thresholds: needs a sparse penalty' },
          ]}
          onChange={choose}
        />
        <SpaceControls
          space={space}
          value={value}
          onChange={setValue}
          admissible={rule.admissible}
          {...(rule.why === undefined ? {} : { why: rule.why })}
        />
      </ControlPanel>
      <CodeBlock title="value" code={JSON.stringify(value, null, 2)} />
    </div>
  )
}
// endregion
