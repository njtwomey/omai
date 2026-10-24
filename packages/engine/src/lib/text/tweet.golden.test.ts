import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { reduceLengthening, removeHandles, replaceHtmlEntities, type TweetOptions, tweetTokenize } from './index.ts'

describe('text/tweet against nltk', () => {
  for (const c of loadFixture('text/tweet').cases) {
    it(c.name, () => {
      const t = c.inputs[0] as string
      const got =
        c.fn === 'tweetTokenize'
          ? tweetTokenize(t, c.inputs[1] as TweetOptions)
          : c.fn === 'replaceHtmlEntities'
            ? replaceHtmlEntities(t)
            : c.fn === 'removeHandles'
              ? removeHandles(t)
              : reduceLengthening(t)
      expect(got).toEqual(c.output)
    })
  }
})
