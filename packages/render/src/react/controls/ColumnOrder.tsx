/**
 * A row of column chips to reorder by dragging (or with Alt and the arrow keys), each with an optional flip toggle, for
 * the axis order and direction of parallel coordinates or a table's columns.
 */

import { type KeyboardEvent, type ReactNode, useState } from 'react'
import { cx, labelClass } from './classes.ts'

/** Props of {@link ColumnOrder}. */
export interface ColumnOrderProps {
  /** The columns, in their current order. */
  readonly order: readonly string[]
  readonly onOrder: (order: readonly string[]) => void
  /** Columns shown inverted; with `onInvert`, each chip has a flip toggle. */
  readonly inverted?: readonly string[]
  readonly onInvert?: (inverted: readonly string[]) => void
  readonly label?: ReactNode
  readonly className?: string
}

/**
 * Move one item of a list to another position.
 *
 * @param list - The list.
 * @param from - The item's index.
 * @param to - Its new index.
 * @returns A new list.
 * @example
 * ```ts
 * import { moveItem } from 'mdml-render/react'
 * moveItem(['a', 'b', 'c'], 0, 2) // => ['b', 'c', 'a']
 * ```
 */
export function moveItem<T>(list: readonly T[], from: number, to: number): T[] {
  const out = [...list]
  const [item] = out.splice(from, 1)
  if (item !== undefined) out.splice(Math.max(0, Math.min(to, out.length)), 0, item)
  return out
}

/**
 * A row of column chips to reorder by dragging (or Alt + ← / → on a focused chip), each with an optional flip toggle:
 * the axis order and direction of a parallel-coordinates chart, a table's columns, a heatmap's rows.
 *
 * @param props - The order, its handler, and optionally the inverted columns and theirs.
 * @returns The chips.
 */
export function ColumnOrder(props: ColumnOrderProps): ReactNode {
  const { order, onOrder, inverted = [], onInvert } = props
  const [dragging, setDragging] = useState<number | null>(null)
  const [over, setOver] = useState<number | null>(null)
  const flip = (c: string): void => {
    if (onInvert === undefined) return
    onInvert(inverted.includes(c) ? inverted.filter((x) => x !== c) : [...inverted, c])
  }
  const nudge = (e: KeyboardEvent<HTMLButtonElement>, i: number): void => {
    if (!e.altKey || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return
    e.preventDefault()
    onOrder(moveItem(order, i, i + (e.key === 'ArrowLeft' ? -1 : 1)))
  }
  return (
    <div className={cx('flex flex-col gap-1', props.className)}>
      {props.label === undefined ? null : <span className={labelClass}>{props.label}</span>}
      <ul className="flex flex-wrap gap-1.5">
        {order.map((c, i) => (
          <li
            key={c}
            draggable
            onDragStart={(e) => {
              setDragging(i)
              e.dataTransfer.effectAllowed = 'move'
              e.dataTransfer.setData('text/plain', c)
            }}
            onDragOver={(e) => {
              e.preventDefault()
              setOver(i)
            }}
            onDragLeave={() => setOver((o) => (o === i ? null : o))}
            onDrop={(e) => {
              e.preventDefault()
              if (dragging !== null && dragging !== i) onOrder(moveItem(order, dragging, i))
              setDragging(null)
              setOver(null)
            }}
            onDragEnd={() => {
              setDragging(null)
              setOver(null)
            }}
            className={cx(
              'inline-flex cursor-grab items-center gap-1 rounded-md border px-2 py-1 text-xs text-(--wb-fg) select-none active:cursor-grabbing',
              dragging === i ? 'opacity-40' : '',
              over === i && dragging !== i ? 'border-(--wb-accent)' : 'border-(--wb-grid)',
            )}
          >
            <button
              type="button"
              aria-label={`Move ${c} (Alt + ← / →)`}
              title="Drag to move (or Alt + ← / →)"
              onKeyDown={(e) => nudge(e, i)}
              className="cursor-grab rounded text-(--wb-muted) outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
            >
              ⋮⋮
            </button>
            {c}
            {onInvert === undefined ? null : (
              <button
                type="button"
                aria-pressed={inverted.includes(c)}
                aria-label={`Invert ${c}`}
                title={inverted.includes(c) ? 'Inverted: click to restore' : 'Invert this axis'}
                onClick={() => flip(c)}
                className={cx(
                  'ml-0.5 rounded px-1 leading-none outline-none hover:bg-(--wb-surface) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50',
                  inverted.includes(c) ? 'text-(--wb-accent)' : 'text-(--wb-muted)',
                )}
              >
                ↕
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
