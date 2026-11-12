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

describe('the combining-two-measurements page', () => {
  it('renders the set-up, the prior and an idle play button, with no worker, no result and no replay', async () => {
    const { default: Story } = await import('./Story.tsx')
    const entry = storyOf('combining-two-measurements')
    if (entry === undefined) throw new Error('the story is not registered')
    const html = renderToString(
      <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
        <Story />
      </StoryContext.Provider>,
    )
    expect(html).toContain('data-phase="idle"')
    expect(html).toContain('Measure the room twice')
    expect(html).toContain('Nothing has been computed yet')
    expect(html).not.toContain('The product of the messages')
    expect(html).not.toContain('Play the replay')
    expect(made).toEqual([])
  }, 60_000)
})
