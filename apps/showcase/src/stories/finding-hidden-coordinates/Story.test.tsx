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

describe('the finding-hidden-coordinates page before play', () => {
  it('says that the points at the top are those of the first run and that Run again draws new ones', async () => {
    const { default: Story } = await import('./Story.tsx')
    const entry = storyOf('finding-hidden-coordinates')
    if (entry === undefined) throw new Error('the story is not registered')
    const html = renderToString(
      <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
        <Story />
      </StoryContext.Provider>,
    )
    const text = html
      .replace(/<!-- -->/g, '')
      .replace(/<[^>]*>/g, ' ')
      .replace(/\s+/g, ' ')
    expect(text).toContain('The points at the top of the page are the points of the first run.')
    expect(text).toContain('Run again , new points are drawn')
    expect(html.indexOf('The points at the top of the page')).toBeLessThan(html.indexOf('data-testid="play"'))
    expect(made).toEqual([])
  }, 60_000)
})
