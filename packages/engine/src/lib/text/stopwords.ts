/**
 * English stop-word lists (scikit-learn's and gensim's) and their removal from token lists.
 */

import { OmaiError } from '../base/index.ts'

// The Glasgow information retrieval group's list, as scikit-learn ships it (ENGLISH_STOP_WORDS).
// biome-ignore format: a word list
const SKLEARN: readonly string[] = [
  'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along',
  'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another',
  'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became',
  'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside',
  'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con',
  'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight',
  'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone',
  'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for',
  'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
  'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers',
  'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest',
  'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many',
  'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must',
  'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none',
  'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or',
  'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps',
  'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she',
  'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something',
  'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their',
  'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon',
  'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru',
  'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until',
  'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever',
  'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
  'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet',
  'you', 'your', 'yours', 'yourself', 'yourselves',
]

// gensim's STOPWORDS: scikit-learn's list and these.
// biome-ignore format: a word list
const GENSIM_EXTRA: readonly string[] = [
  'computer', 'did', 'didn', 'does', 'doesn', 'doing', 'don', 'just', 'kg', 'km', 'make', 'quite', 'really',
  'regarding', 'say', 'unless', 'used', 'using', 'various',
]

/** A named stop-word list. */
export type StopWordList = 'sklearn' | 'gensim'

/**
 * An English stop-word list: `sklearn` (scikit-learn's ENGLISH_STOP_WORDS, the Glasgow IR list, 318 words) or `gensim`
 * (gensim's STOPWORDS, the same plus 19 more).
 *
 * @param name - The list (default `sklearn`).
 * @returns The stop words, sorted.
 * @throws OmaiError with code `value` for an unknown list.
 * @concept stop-words
 * @example
 * ```ts
 * import { stopWords } from 'omai-engine/text'
 * [stopWords().length, stopWords('gensim').length, stopWords().includes('the')] // => [318, 337, true]
 * ```
 */
export function stopWords(name: StopWordList = 'sklearn'): string[] {
  if (name === 'sklearn') return [...SKLEARN]
  if (name === 'gensim') return [...SKLEARN, ...GENSIM_EXTRA].sort()
  throw new OmaiError('value', `stopWords: unknown list ${String(name)}`)
}

/**
 * Remove stop words from a token list (case-sensitive, as scikit-learn's vectorisers after lowercasing).
 *
 * @param tokens - The tokens.
 * @param stop - A named list or an explicit set of stop words (default `sklearn`).
 * @returns The tokens that are not stop words.
 * @concept stop-words
 * @example
 * ```ts
 * import { removeStopWords } from 'omai-engine/text'
 * removeStopWords(['the', 'cat', 'sat', 'on', 'the', 'mat']) // => ['cat', 'sat', 'mat']
 * ```
 */
export function removeStopWords(
  tokens: readonly string[],
  stop: StopWordList | ReadonlySet<string> = 'sklearn',
): string[] {
  const set = typeof stop === 'string' ? new Set(stopWords(stop)) : stop
  return tokens.filter((t) => !set.has(t))
}
