import { type CSSProperties, type ReactNode, useId, useState } from 'react'
import { cx } from './classes.ts'
import { ChevronDownIcon } from './icons.tsx'

/** Props of {@link ControlPanel}. */
export interface ControlPanelProps {
  readonly children: ReactNode
  /** A heading; with one, the panel can be collapsed from it. */
  readonly title?: ReactNode
  /** Columns of the grid the controls are laid out in (default 1). */
  readonly columns?: number
  /** Rows of the grid: with rows, controls fill each column top to bottom before the next. */
  readonly rows?: number
  /**
   * Named areas of a bento grid, one string per row (CSS `grid-template-areas`, e.g. `['data data seed', 'rate k k']`);
   * place each control with a {@link PanelArea} of the same name.
   */
  readonly areas?: readonly string[]
  /** Whether the title collapses the panel (default true when there is a title). */
  readonly collapsible?: boolean
  /** Open at first (default true). */
  readonly defaultOpen?: boolean
  readonly className?: string
}

/**
 * A panel of controls: laid out in a given number of columns or rows, or in named bento areas, and collapsible from
 * its title. Place it beside or above a figure with {@link ControlLayout}.
 *
 * @param props - The controls and their layout.
 * @returns The panel.
 */
export function ControlPanel(props: ControlPanelProps): ReactNode {
  const { children, title, columns, rows, areas, className } = props
  const collapsible = props.collapsible ?? title !== undefined
  const [open, setOpen] = useState(props.defaultOpen ?? true)
  const bodyId = useId()
  const grid: CSSProperties =
    areas !== undefined
      ? {
          gridTemplateAreas: areas.map((r) => `"${r}"`).join(' '),
          gridTemplateColumns: `repeat(${Math.max(...areas.map((r) => r.trim().split(/\s+/).length))}, minmax(0, 1fr))`,
        }
      : rows !== undefined
        ? {
            gridTemplateRows: `repeat(${rows}, auto)`,
            gridAutoFlow: 'column',
            gridAutoColumns: 'minmax(0, 1fr)',
          }
        : { gridTemplateColumns: `repeat(${columns ?? 1}, minmax(0, 1fr))` }
  return (
    <section className={cx('rounded-lg border border-(--wb-grid) bg-(--wb-surface) text-(--wb-fg)', className)}>
      {title === undefined ? null : collapsible ? (
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-xs font-semibold tracking-wide text-(--wb-muted) uppercase outline-none hover:text-(--wb-fg) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        >
          <span className="truncate">{title}</span>
          <ChevronDownIcon size={14} className={cx('shrink-0 transition-transform', !open && '-rotate-90')} />
        </button>
      ) : (
        <div className="px-3 py-2 text-xs font-semibold tracking-wide text-(--wb-muted) uppercase">{title}</div>
      )}
      {open || !collapsible ? (
        <div
          id={bodyId}
          className={cx('grid items-end gap-x-4 gap-y-3 px-3 pb-3', title === undefined && 'pt-3')}
          style={grid}
        >
          {children}
        </div>
      ) : null}
    </section>
  )
}

/**
 * One cell of a {@link ControlPanel}'s bento grid, by area name; or, without areas, a cell spanning columns or rows.
 *
 * @param props - The area's name (or spans) and its controls.
 * @returns The cell.
 */
export function PanelArea(props: {
  readonly children: ReactNode
  readonly name?: string
  readonly columnSpan?: number
  readonly rowSpan?: number
  readonly className?: string
}): ReactNode {
  const { children, name, columnSpan, rowSpan, className } = props
  const style: CSSProperties = {
    ...(name === undefined ? {} : { gridArea: name }),
    ...(columnSpan === undefined ? {} : { gridColumn: `span ${columnSpan} / span ${columnSpan}` }),
    ...(rowSpan === undefined ? {} : { gridRow: `span ${rowSpan} / span ${rowSpan}` }),
  }
  return (
    <div className={cx('flex min-w-0 flex-col gap-3 [&>*]:max-w-none', className)} style={style}>
      {children}
    </div>
  )
}

/** Where {@link ControlLayout} puts the controls. */
export type ControlPlacement = 'left' | 'right' | 'top' | 'bottom'

/**
 * A figure with its controls: in a sidebar to the left or right (stacked above the figure on narrow screens), or in a
 * bar above or below it. The figure keeps its own size; the layout only places the two.
 *
 * @param props - The controls, where they go, the sidebar's width and the figure.
 * @returns The layout.
 */
export function ControlLayout(props: {
  readonly controls: ReactNode
  readonly children: ReactNode
  readonly placement?: ControlPlacement
  /** The sidebar's width in pixels (default 240). */
  readonly sidebarWidth?: number
  readonly className?: string
}): ReactNode {
  const { controls, children, placement = 'left', sidebarWidth = 240, className } = props
  if (placement === 'top' || placement === 'bottom')
    return (
      <div className={cx('flex flex-col gap-3', className)}>
        {placement === 'top' ? controls : null}
        <div className="min-w-0">{children}</div>
        {placement === 'bottom' ? controls : null}
      </div>
    )
  return (
    <div
      className={cx(
        'flex flex-col gap-4 md:flex-row md:items-start',
        placement === 'right' && 'md:flex-row-reverse',
        className,
      )}
    >
      <div
        className="w-full shrink-0 md:w-(--wb-sidebar)"
        style={{ '--wb-sidebar': `${sidebarWidth}px` } as CSSProperties}
      >
        {controls}
      </div>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}
