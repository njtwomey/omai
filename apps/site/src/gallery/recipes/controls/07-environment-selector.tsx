import {
  type ActionKind,
  ENVIRONMENT_FAMILIES,
  type EnvironmentFamily,
  environmentRegistry,
  type ObservationKind,
} from 'mdml-engine/environments'
import {
  ControlPanel,
  EnvironmentSelector,
  firstFrame,
  SceneFrame,
  SeedInput,
  Select,
  selectEnvironment,
  tagLabel,
  useEnvironment,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Environment selector',
  question: 'How do I let the reader choose an environment by its family and its spaces?',
  explain:
    '`EnvironmentSelector` lists the tagged environment registry of mdml-engine (`environmentRegistry`: every environment definition and the differentiable control models, tagged by family, by whether episodes end at terminal states, and by the kinds of action and observation derived from their spaces). Its props are the filters (`family`, `actions`, `observations`, `episodic`, `multiAgent`), a search box narrows further, and the chosen entry shows its description, its spaces as gymnasium prints them and its definition’s options as controls. `useEnvironment` makes the selection once per change; here its first frame is drawn from a seed.',
}

type Any<T extends string> = 'any' | T
const ACTIONS: readonly ActionKind[] = ['discrete', 'continuous', 'multi']
const OBSERVATIONS: readonly ObservationKind[] = ['vector', 'grid', 'image', 'discrete']

export default function EnvironmentSelectorRecipe(): ReactNode {
  const [family, setFamily] = useState<Any<EnvironmentFamily>>('any')
  const [actions, setActions] = useState<Any<ActionKind>>('any')
  const [observations, setObservations] = useState<Any<ObservationKind>>('any')
  const [seed, setSeed] = useState(0)
  const [selection, setSelection] = useState(() => selectEnvironment(environmentRegistry, 'cartPole'))
  const env = useEnvironment(environmentRegistry, selection)
  const scene = useMemo(() => firstFrame(env, seed), [env, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4} title="filters">
        <Select
          label="family"
          value={family}
          options={['any', ...ENVIRONMENT_FAMILIES.map((f) => ({ value: f, label: tagLabel(f) }))]}
          onChange={setFamily}
          cycle={false}
        />
        <Select label="actions" value={actions} options={['any', ...ACTIONS]} onChange={setActions} cycle={false} />
        <Select
          label="observations"
          value={observations}
          options={['any', ...OBSERVATIONS]}
          onChange={setObservations}
          cycle={false}
        />
        <SeedInput value={seed} onChange={setSeed} label="reset seed" />
      </ControlPanel>
      <div className="flex flex-wrap items-start gap-3">
        <ControlPanel columns={1} className="w-80">
          <EnvironmentSelector
            {...(family === 'any' ? {} : { family })}
            {...(actions === 'any' ? {} : { actions })}
            {...(observations === 'any' ? {} : { observations })}
            value={selection}
            onChange={setSelection}
          />
        </ControlPanel>
        <SceneFrame
          scene={scene}
          maxWidth={480}
          maxHeight={360}
          filename="environment"
          caption={`${selection.name}, first frame`}
        />
      </div>
    </div>
  )
}
