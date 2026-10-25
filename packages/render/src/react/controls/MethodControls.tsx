/**
 * Controls for a method's hyperparameters, generated from the `hyperSpace` its estimator declares, starting from the
 * hyperparameters the estimator was made with.
 */

import type { Estimator } from 'mdml-engine/learn'
import { defaults, type SpaceValue } from 'mdml-engine/space'
import type { ReactNode } from 'react'
import { SpaceControls } from './SpaceControls.tsx'

/** Props of {@link MethodControls}. */
export interface MethodControlsProps<E extends Estimator> {
  /** The estimator as the page made it: its `hyperSpace` gives the controls, its demo hyperparameters their start. */
  readonly estimator: E
  /** The current hyperparameters (default: the space's defaults, which are the estimator's own values). */
  readonly value?: Readonly<Record<string, SpaceValue>>
  readonly onChange: (hyper: Readonly<Record<string, SpaceValue>>) => void
  /** Hyperparameters to leave out. */
  readonly exclude?: readonly string[]
  readonly disabled?: boolean
}

/**
 * Controls for a method's hyperparameters, generated from its `hyperSpace` (estimators declare one for tuning and for
 * this), starting from the hyperparameters the estimator was made with. Pass the result to `estimator.with(hyper)`.
 * An estimator without a space gets no controls.
 *
 * @param props - The estimator, the current hyperparameters and what to call with new ones.
 * @returns The controls, side by side for a `ControlPanel` to lay out.
 */
export function MethodControls<E extends Estimator>(props: MethodControlsProps<E>): ReactNode {
  const { estimator, onChange, exclude = [], disabled = false } = props
  const space = estimator.hyperSpace
  if (space === undefined || space.kind !== 'dict') return null
  const value = props.value ?? (defaults(space) as Readonly<Record<string, SpaceValue>>)
  return (
    <SpaceControls
      space={space}
      value={value}
      onChange={(v) => onChange(v as Readonly<Record<string, SpaceValue>>)}
      exclude={exclude}
      disabled={disabled}
    />
  )
}
