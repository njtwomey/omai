import type { ReactNode } from 'react'

/**
 * Greys out what it holds while the settings no longer match it: in live mode a change of a setting is held until play,
 * and the charts still show the run of the old settings. They are dimmed and drained of colour, with a notice, so they
 * are not mistaken for the result of what the controls now say.
 *
 * @param props - The session's `stale` flag, and the charts.
 * @returns The charts, dimmed when stale.
 */
export function StaleVeil(props: { readonly stale: boolean; readonly children: ReactNode }): ReactNode {
  const { stale, children } = props
  return (
    <div className="relative">
      <div className={`transition duration-200 ${stale ? 'opacity-35 grayscale' : ''}`} aria-busy={stale}>
        {children}
      </div>
      {stale ? (
        <div
          role="status"
          className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 rounded-md border border-line bg-surface px-3 py-1.5 text-xs font-medium text-fg shadow"
        >
          The settings have changed: press play to update
        </div>
      ) : null}
    </div>
  )
}
