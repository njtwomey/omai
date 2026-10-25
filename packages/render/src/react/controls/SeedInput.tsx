/**
 * A random-seed control: a whole number to type or step, and a die that rerolls it to a seed derived from the current
 * one, so a page's rerolls are reproducible.
 */

import type { ReactNode } from 'react'
import { cx, iconButtonClass } from './classes.ts'
import { DiceIcon } from './icons.tsx'
import { NumberField } from './NumberField.tsx'
import { MAX_SEED, rerollSeed } from './seed.ts'

/** Props of {@link SeedInput}. */
export interface SeedInputProps {
  readonly value: number
  readonly onChange: (seed: number) => void
  readonly label?: ReactNode
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A random seed: a whole number to type or step, and a die that rerolls it (to a seed derived from the current one,
 * so a page's rerolls are reproducible too).
 *
 * @param props - The seed and its label (default "seed").
 * @returns The seed field.
 */
export function SeedInput(props: SeedInputProps): ReactNode {
  const { value, onChange, label = 'seed', disabled = false, className } = props
  return (
    <div className={cx('flex w-full min-w-36 max-w-xs items-end gap-1.5', className)}>
      <NumberField
        label={label}
        value={value}
        onChange={onChange}
        integer
        min={0}
        max={MAX_SEED}
        ladder={false}
        slider={false}
        disabled={disabled}
        className="min-w-0 flex-1"
      />
      <button
        type="button"
        className={cx(iconButtonClass, 'rounded-md')}
        aria-label="New seed"
        title="New seed"
        disabled={disabled}
        onClick={() => onChange(rerollSeed(value))}
      >
        <DiceIcon />
      </button>
    </div>
  )
}
