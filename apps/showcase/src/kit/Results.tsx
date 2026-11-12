import { type ReactNode, useEffect, useRef } from 'react'
import type { StoryRun } from './useStoryRun.ts'

/**
 * What the run produces, placed right after `<Play>`, so the page reads straight down: pressing play changes nothing above the
 * button, and everything the result makes (the figures, the prose that reads numbers out of it, the closing paragraphs and the code)
 * appears below it, in order, with the kit's fade. Until the result exists the block holds one calm line, so there is no empty
 * hole and nothing below it to jump (there is nothing below it: the story's continuation lives inside the block).
 *
 * When a result arrives, the page scrolls only if the top of the block is below the fold at that moment (then it brings the
 * block's top into view, smoothly, instantly when the reader prefers reduced motion). It never scrolls otherwise.
 *
 * @param props - The run, what to build from the result, and optionally the line shown before there is one.
 * @returns The block.
 */
export function Results<R>(props: {
  readonly run: StoryRun<R>
  /** Builds the content from the result. It is called only when a result exists. */
  readonly children: (result: R) => ReactNode
  /** The line shown while there is no result (default: nothing has been computed yet). */
  readonly placeholder?: ReactNode
}): ReactNode {
  const { run, children, placeholder = 'Nothing has been computed yet. The result will appear here.' } = props
  const block = useRef<HTMLDivElement>(null)
  const phase = useRef(run.phase)

  useEffect(() => {
    const was = phase.current
    phase.current = run.phase
    const el = block.current
    if (was !== 'running' || run.phase !== 'done' || el === null) return
    const top = el.getBoundingClientRect().top
    if (top <= window.innerHeight) return
    const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
    window.scrollBy({ top: top - window.innerHeight * 0.3, behavior: calm ? 'instant' : 'smooth' })
  }, [run.phase])

  return (
    <div
      ref={block}
      className={`story-results story-prose-flow ${run.result === undefined ? '' : 'story-reveal'}`}
      data-testid="results"
      data-state={run.result === undefined ? 'empty' : 'filled'}
    >
      {run.result === undefined ? (
        <p className="story-block m-0 font-sans text-sm text-muted">{placeholder}</p>
      ) : (
        children(run.result)
      )}
    </div>
  )
}
