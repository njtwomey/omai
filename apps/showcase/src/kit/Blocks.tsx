import type { ReactNode } from 'react'
import { CodeBlock } from '../../../site/src/shell/CodeBlock.tsx'
import { snippetOf } from '../../../site/src/shell/snippet.ts'

/**
 * A figure with its caption, which comes in with a short fade when it first appears (a story renders its figures once the
 * result exists, so the fade is the moment the computation becomes a picture). `wide` lets it run a little past the text column
 * on large screens.
 *
 * @param props - The caption and the figure.
 * @returns The figure.
 */
export function Figure(props: {
  readonly caption: ReactNode
  readonly children: ReactNode
  readonly wide?: boolean
}): ReactNode {
  return (
    <figure className={`story-block story-reveal m-0 ${props.wide === true ? 'lg:-mx-14' : ''}`}>
      <div className="rounded-lg border border-line bg-surface p-2 sm:p-3">{props.children}</div>
      <figcaption className="mt-2.5 px-1 font-sans text-sm leading-relaxed text-muted">{props.caption}</figcaption>
    </figure>
  )
}

/**
 * Prose (or anything) that appears only once the result exists, with the same fade as a figure. Use it for the paragraphs that
 * read numbers out of the result.
 *
 * @param props - The content.
 * @returns The content, in a block that fades in.
 */
export function Reveal(props: { readonly children: ReactNode }): ReactNode {
  return <div className="story-reveal story-prose-flow">{props.children}</div>
}

/**
 * A margin note: a side remark that is not part of the argument. On wide screens it hangs in the right margin beside the
 * paragraph it follows; below that it is an inset note in the column.
 *
 * @param props - The note.
 * @returns The note.
 */
export function Aside(props: { readonly children: ReactNode }): ReactNode {
  return (
    <aside className="story-aside rounded-md border-l-2 border-line py-1 pl-3 font-sans text-sm leading-relaxed text-muted">
      {props.children}
    </aside>
  )
}

/**
 * The code that ran, under the heading 'How it is written': the regions of a source file (imported with `?raw`) marked with
 * `// region` comments, so what is shown is the code the worker runs, never a copy. Write the shown code as the site shows
 * code everywhere: tensor operations only, one variable per line, comments on their own line above the code, lines under about
 * a hundred characters.
 *
 * @param props - The raw source, the region to show (default every region, with the imports) and the file's name.
 * @returns The code block.
 */
export function Code(props: {
  readonly source: string
  readonly region?: string
  readonly title: string
  readonly heading?: string
}): ReactNode {
  const { source, region, title, heading = 'How it is written' } = props
  return (
    <section className="story-block font-sans">
      <h2 className="mb-3 text-xl font-semibold tracking-tight">{heading}</h2>
      <CodeBlock code={snippetOf(source, region === undefined ? {} : { region })} title={title} />
    </section>
  )
}
