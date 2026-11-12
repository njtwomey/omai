import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StoryContext } from '../../kit/StoryPage.tsx'
import { storyOf } from '../../registry.ts'

// Rendering the page must not start a worker or show a result: only the play button does.
const made: unknown[] = []

beforeEach(() => {
  made.length = 0
  vi.stubGlobal(
    'Worker',
    class {
      constructor(...args: unknown[]) {
        made.push(args)
      }
    },
  )
})
afterEach(() => vi.unstubAllGlobals())

describe('the how-expectation-propagation-works page', () => {
  it('renders the problem, the graph and an idle play button, with no worker and no result', async () => {
    const { default: Story } = await import('./Story.tsx')
    const entry = storyOf('how-expectation-propagation-works')
    if (entry === undefined) throw new Error('the story is not registered')
    const html = renderToString(
      <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
        <Story />
      </StoryContext.Provider>,
    )
    expect(html).toContain('data-phase="idle"')
    expect(html).toContain('The factor graph of the model')
    expect(html).toContain('One update, step by step')
    expect(html).toContain('The rest of the post appears here')
    expect(html).not.toContain('The same run from start to finish')
    expect(made).toEqual([])
  }, 60_000)

  it('reads a settled run and a run that did not settle without saying anything false', async () => {
    const { RunStory } = await import('./Story.tsx')
    const { fit } = await import('./compute.ts')
    const settled = renderToString(<RunStory result={fit(37)} />)
    expect(settled).toContain('This run stopped after')
    expect(settled).toContain('The same run from start to finish')
    expect(settled).not.toContain('This run did not settle')
    // seed 4 does not settle in sixty sweeps: the text says so and does not claim a fixed point
    const result = fit(4)
    expect(result.settled).toBe(false)
    const unsettled = renderToString(<RunStory result={result} />)
    expect(unsettled).toContain('This run did not settle')
    expect(unsettled).not.toContain('This run stopped after')
    expect(unsettled).not.toContain('fixed point of the updates')
  }, 60_000)
})
