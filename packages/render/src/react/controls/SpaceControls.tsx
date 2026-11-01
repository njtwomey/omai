/**
 * Controls generated from an engine `Space`: one control per dimension, drawn as its kind calls for (number fields,
 * segmented choices, selects, toggles), with conditional dimensions appearing as their conditions come to hold.
 */

import type { Space, SpaceValue } from 'mdml-engine/space'
import type { ReactNode } from 'react'
import { Choice } from './Choice.tsx'
import { cx, labelClass } from './classes.ts'
import { NumberField } from './NumberField.tsx'
import type { OptionValue } from './options.ts'
import { Select } from './Select.tsx'
import {
  type Admissibility,
  type ControlSpec,
  greyOptions,
  type SpaceControlOptions,
  spaceControls,
  valueAt,
  withValueAt,
} from './space-controls.ts'
import { Toggle } from './Toggle.tsx'

/** Props of {@link SpaceControls}. */
export interface SpaceControlsProps<V extends SpaceValue> extends SpaceControlOptions {
  readonly space: Space
  readonly value: V
  readonly onChange: (value: V) => void
  /** Top-level keys to leave out (a dict space's dimensions the page sets itself). */
  readonly exclude?: readonly string[]
  readonly disabled?: boolean
  /**
   * Whether an option of a choice may be chosen, given the path of the control and the option: an option that may not
   * is drawn disabled, titled with the reason. The control stays generic; the page supplies the rule (a mode table's
   * `admissible`, say). Without it every option is available.
   */
  readonly admissible?: Admissibility['admissible']
  /** The note on an inadmissible option, shown as its title. */
  readonly why?: Admissibility['why']
}

/**
 * Controls generated from an engine `Space`: one per dimension, each drawn as its kind calls for (see `spaceControls`),
 * with conditional dimensions appearing as their conditions come to hold. The controls are returned side by side, for
 * a {@link ControlPanel} or any other container to lay out.
 *
 * @param props - The space, its value and what to call with a new value.
 * @returns The controls.
 */
export function SpaceControls<V extends SpaceValue>(props: SpaceControlsProps<V>): ReactNode {
  const { space, value, onChange, exclude = [], disabled = false } = props
  const rule: Admissibility | undefined =
    props.admissible === undefined
      ? undefined
      : { admissible: props.admissible, ...(props.why === undefined ? {} : { why: props.why }) }
  const options: SpaceControlOptions = props.segmentedUpTo === undefined ? {} : { segmentedUpTo: props.segmentedUpTo }
  const controls = spaceControls(space, value, options).filter((c) => !exclude.includes(String(c.path[0])))
  const set = (spec: ControlSpec, leaf: SpaceValue): void => onChange(withValueAt(space, value, spec.path, leaf) as V)
  return controls.map((c) => <ControlFor key={c.id} spec={c} value={value} set={set} disabled={disabled} rule={rule} />)
}

function ControlFor(props: {
  readonly spec: ControlSpec
  readonly value: SpaceValue
  readonly set: (spec: ControlSpec, leaf: SpaceValue) => void
  readonly disabled: boolean
  readonly rule: Admissibility | undefined
}): ReactNode {
  const { spec, value, set, disabled, rule } = props
  const current = valueAt(value, spec.path)
  switch (spec.kind) {
    case 'number':
      return (
        <NumberField
          label={spec.label}
          value={typeof current === 'number' ? current : Number.NaN}
          onChange={(v) => set(spec, v)}
          {...spec.spec}
          {...(spec.unit === undefined ? {} : { unit: spec.unit })}
          disabled={disabled}
        />
      )
    case 'choice':
    case 'select': {
      const Component = spec.kind === 'choice' ? Choice : Select
      return (
        <Component<OptionValue>
          label={spec.label}
          value={current as OptionValue}
          options={greyOptions(spec.options, spec.path, rule, spec.descriptions)}
          onChange={(v) => set(spec, v)}
          disabled={disabled}
        />
      )
    }
    case 'toggle':
      return (
        <Toggle
          label={spec.label}
          value={current === true || current === 1}
          onChange={(v) => set(spec, typeof current === 'number' ? (v ? 1 : 0) : v)}
          disabled={disabled}
        />
      )
    case 'group':
      return (
        <fieldset className="flex min-w-0 flex-col gap-2 border-0 p-0">
          {spec.label === '' ? null : <legend className={cx(labelClass, 'mb-1')}>{spec.label}</legend>}
          <div className="flex flex-wrap gap-2">
            {spec.children.map((c) => (
              <ControlFor key={c.id} spec={c} value={value} set={set} disabled={disabled} rule={rule} />
            ))}
          </div>
        </fieldset>
      )
  }
}
