/**
 * The small rows a figure frame holds beside its figure: `Controls`, a wrapping row for the inputs that drive the
 * figure, and `Readouts`, the label and value pairs the figure reports. Both are plain styled elements over the
 * `--wb-*` tokens, used by the `controls` and `readouts` slots of `FigureFrame` and equally on their own.
 */

import type { CSSProperties, ReactElement, ReactNode } from 'react'
import { useTokens } from './theme.tsx'

/** One readout: a label and the value it shows. */
export interface ReadoutItem {
  readonly label: ReactNode
  readonly value: ReactNode
}

/** Props of {@link Readouts}. */
export interface ReadoutsProps {
  /** The label and value pairs, in order. */
  readonly items: readonly ReadoutItem[]
  readonly className?: string
  readonly style?: CSSProperties
}

/**
 * Label and value pairs as a description list (`<dl>`): each label in the muted colour above its value in the
 * monospace face with tabular numerals, so figures that change as a slider moves do not jitter. The pairs wrap onto
 * further rows when the container is narrow. Nothing is drawn for an empty list.
 *
 * @param props - The pairs.
 * @returns The list, or `null` when there are no items.
 * @index function React component: label and value pairs as a dl: `<Readouts items={[{ label: 'loss', value: '0.31' }]} />`
 * @example
 * ```tsx
 * import { Readouts } from 'omai-render/react'
 * const r = <Readouts items={[{ label: 'epoch', value: 12 }, { label: 'loss', value: '0.314' }]} />
 * ```
 */
export function Readouts(props: ReadoutsProps): ReactElement | null {
  const t = useTokens()
  if (props.items.length === 0) return null
  return (
    <dl
      className={props.className}
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '6px 24px',
        margin: 0,
        fontVariantNumeric: 'tabular-nums',
        ...props.style,
      }}
    >
      {props.items.map((item, i) => (
        // The pairs are positional and carry no identity of their own.
        // biome-ignore lint/suspicious/noArrayIndexKey: a fixed ordered list of pairs
        <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <dt style={{ font: `11px ${t.font}`, color: `var(--wb-muted, ${t.muted})` }}>{item.label}</dt>
          <dd
            style={{
              margin: 0,
              font: `600 13px ${t.monoFont}`,
              color: `var(--wb-fg, ${t.fg})`,
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {item.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/** Props of {@link Controls}. */
export interface ControlsProps {
  /** The controls: `NumberField`, `Toggle`, `Select`, `Choice` or any element. */
  readonly children?: ReactNode
  /** Names the group for assistive technology (`role="group"`); without it the row is a plain container. */
  readonly label?: string
  readonly className?: string
  readonly style?: CSSProperties
}

/**
 * A row for a figure's controls: a wrapping flex row with a consistent gap (12px across, 12px between rows), the
 * controls aligned along their bottom so a field and a toggle share a baseline-ish line. Give a control a `width` to
 * size it inside the row (`<NumberField width={176} />`).
 *
 * @param props - The controls and an optional group name.
 * @returns The row.
 * @index function React component: a wrapping row of controls: `<Controls><NumberField .../><Toggle .../></Controls>`
 * @example
 * ```tsx
 * import { Controls, NumberField, Toggle } from 'omai-render/react'
 * const row = (
 *   <Controls label="Model">
 *     <NumberField label="k" value={3} onChange={() => {}} min={1} width={176} />
 *     <Toggle label="log scale" value={false} onChange={() => {}} />
 *   </Controls>
 * )
 * ```
 */
export function Controls(props: ControlsProps): ReactElement {
  return (
    <div
      {...(props.label === undefined ? {} : { role: 'group', 'aria-label': props.label })}
      className={props.className}
      style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-end', gap: 12, ...props.style }}
    >
      {props.children}
    </div>
  )
}
