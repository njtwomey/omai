/**
 * An environment selector over a tagged registry: the filters are its props (families, kinds of action and
 * observation, episodic, multi-agent), a search box narrows further, each entry shows its title and tags, and the
 * chosen entry shows its description, its spaces and its definition's options as controls. Make the environment with
 * `useEnvironment`.
 */

import {
  type ActionKind,
  type EnvironmentEntry,
  type EnvironmentFamily,
  type EnvironmentRegistry,
  environmentRegistry,
  findEnvironment,
  type ObservationKind,
  whereEnvironments,
} from 'omai-engine/environments'
import type { SpaceValue } from 'omai-engine/space'
import { type ReactNode, useMemo } from 'react'
import { useByContent } from '../_content.ts'
import { cx, labelClass } from './classes.ts'
import { EntryList, tagLabel } from './EntryList.tsx'
import {
  type AnyEnvironment,
  type EnvironmentSelection,
  formatSpace,
  makeSelection,
  selectEnvironment,
} from './environments.ts'
import { SpaceControls } from './SpaceControls.tsx'

/** Props of {@link EnvironmentSelector}: the filters of a registry query, the selection and the controls to draw. */
export interface EnvironmentSelectorProps {
  /** The registry to choose from (default the library's). */
  readonly registry?: EnvironmentRegistry
  readonly family?: EnvironmentFamily | readonly EnvironmentFamily[]
  readonly actions?: ActionKind | readonly ActionKind[]
  readonly observations?: ObservationKind | readonly ObservationKind[]
  /** Episodic tasks only (`true`), continuing only (`false`), or both. */
  readonly episodic?: boolean
  /** Multi-agent environments only (`true`), single-agent only (`false`), or both. */
  readonly multiAgent?: boolean
  readonly value: EnvironmentSelection
  readonly onChange: (selection: EnvironmentSelection) => void
  /** Show the search box (default true). */
  readonly search?: boolean
  /** Show the definition's options as controls (default true). */
  readonly options?: boolean
  /** Options to leave out. */
  readonly exclude?: readonly string[]
  readonly label?: ReactNode
  readonly disabled?: boolean
}

/**
 * The tags shown for an environment entry: its family, kinds of action and observation, episodic or continuing, and
 * its agents when there are several.
 *
 * @param entry - The entry.
 * @returns The tags, as chips read them.
 */
export function environmentTags(entry: EnvironmentEntry): string[] {
  const tags = [
    tagLabel(entry.family),
    `${entry.actions} actions`,
    `${entry.observations} observations`,
    entry.episodic ? 'episodic' : 'continuing',
  ]
  if (entry.agents !== undefined) tags.push(`${entry.agents.length} agents`)
  return tags
}

/**
 * Choose an environment from a registry filtered by the props, each entry with its title and tags; the chosen entry's
 * description, spec and spaces are shown under the list, with its options as controls. Choosing another entry starts
 * it from its default options.
 *
 * @param props - The filters, the selection and what to call with a new one.
 * @returns The list, the readout and the controls, for a `ControlPanel` to lay out.
 * @index function React component: pick an environment from the registry: `<EnvironmentSelector value={sel} onChange={setSel} />`
 */
export function EnvironmentSelector(props: EnvironmentSelectorProps): ReactNode {
  const {
    registry = environmentRegistry,
    family,
    actions,
    observations,
    episodic,
    multiAgent,
    value,
    onChange,
    search = true,
    options = true,
    exclude = [],
    label = 'environment',
    disabled = false,
  } = props
  const entries = whereEnvironments(registry, {
    ...(family === undefined ? {} : { family }),
    ...(actions === undefined ? {} : { actions }),
    ...(observations === undefined ? {} : { observations }),
    ...(episodic === undefined ? {} : { episodic }),
    ...(multiAgent === undefined ? {} : { multiAgent }),
  })
  const chosen = findEnvironment(registry, value.name)
  return (
    <>
      <EntryList
        label={label}
        rows={entries.map((e) => ({
          name: e.name,
          title: e.title,
          tags: environmentTags(e),
          description: `${e.description} ${e.spec.id}`,
        }))}
        value={value.name}
        onChange={(name) => onChange(selectEnvironment(registry, name))}
        search={search}
        placeholder="Search environments"
        disabled={disabled}
      />
      {chosen === undefined ? null : <EnvironmentReadout entry={chosen} />}
      {chosen?.options === undefined || !options ? null : (
        <SpaceControls
          space={chosen.options}
          value={value.options}
          onChange={(o) => onChange({ ...value, options: o as Record<string, SpaceValue> })}
          exclude={exclude}
          disabled={disabled}
        />
      )}
    </>
  )
}

/** The chosen entry's description, id, time limit and spaces. */
function EnvironmentReadout(props: { readonly entry: EnvironmentEntry }): ReactNode {
  const { entry } = props
  const rows: readonly (readonly [string, string])[] = [
    ['id', entry.spec.id],
    ...(entry.spec.maxEpisodeSteps === undefined
      ? []
      : [['time limit', `${entry.spec.maxEpisodeSteps} steps`] as const]),
    [
      entry.agents === undefined ? 'observations' : `observations (${entry.agents[0] ?? ''})`,
      formatSpace(entry.observationSpace),
    ],
    [entry.agents === undefined ? 'actions' : `actions (${entry.agents[0] ?? ''})`, formatSpace(entry.actionSpace)],
  ]
  return (
    <div className="flex min-w-36 flex-col gap-1 text-xs">
      <p className="text-(--wb-fg)">{entry.description}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className={cx(labelClass, 'whitespace-nowrap')}>{k}</dt>
            <dd className="min-w-0 break-words font-mono text-[11px] text-(--wb-fg)">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

/**
 * The environment a selection names, made once per change of selection, of whatever form its entry has.
 *
 * @param registry - The registry.
 * @param selection - The selection.
 * @returns The environment.
 */
export function useEnvironment(registry: EnvironmentRegistry, selection: EnvironmentSelection): AnyEnvironment {
  // The selection is compared by content, not identity: a page may rebuild it on every render.
  const options = useByContent(selection.options)
  const { name } = selection
  return useMemo(() => makeSelection(registry, { name, options }), [registry, name, options])
}
