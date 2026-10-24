import { type ReactNode, useState } from 'react'
import { CodeBlock } from './CodeBlock.tsx'

/** Where a recipe's code goes relative to its figure. */
export type RecipeLayout = 'below' | 'beside'

/**
 * A recipe: its live figure and its own source, the code below the figure or beside it (on wide screens), switchable
 * by the reader.
 */
export function RecipeView(props: {
  readonly children: ReactNode
  readonly code: string
  readonly file?: string
  readonly layout?: RecipeLayout
}): ReactNode {
  const { children, code, file } = props
  const [layout, setLayout] = useState<RecipeLayout>(props.layout ?? 'below')
  const beside = layout === 'beside'
  return (
    <div className="flex flex-col gap-3">
      <div
        className={
          beside ? 'grid gap-4 xl:grid-cols-[minmax(0,auto)_minmax(0,1fr)] xl:items-start' : 'flex flex-col gap-4'
        }
      >
        <div className="min-w-0 overflow-x-auto">{children}</div>
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="hidden justify-end xl:flex">
            <button
              type="button"
              onClick={() => setLayout(beside ? 'below' : 'beside')}
              className="rounded px-1.5 py-0.5 text-xs text-muted hover:bg-surface hover:text-fg"
            >
              {beside ? 'Code below the figure' : 'Code beside the figure'}
            </button>
          </div>
          <CodeBlock code={code} {...(file === undefined ? {} : { title: file })} />
        </div>
      </div>
    </div>
  )
}
