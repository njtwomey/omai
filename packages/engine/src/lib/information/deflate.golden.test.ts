import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

const VOCAB = (
  'the of and to in a is that for it as was with be by on not he I this are or his from at which but have an had ' +
  'they you were their one all we can her has there been if more when will would who so no information entropy ' +
  'channel code source noise message bits'
).split(' ')

const lcg = (x: number): number => (75 * x + 74) % 65537

/** The recipe of oracle/cases/information/deflate.py: pseudo-English text, pseudo-random bytes or runs of letters. */
function recipe(kind: string, n: number, seed: number): Uint8Array {
  const out: number[] = []
  let x = seed
  const enc = new TextEncoder()
  while (out.length < n) {
    x = lcg(x)
    if (kind === 'text') {
      const word = VOCAB[x % VOCAB.length] ?? ''
      x = lcg(x)
      const sep = x % 16
      out.push(...enc.encode(word + (sep < 12 ? ' ' : sep < 14 ? ', ' : sep === 14 ? '.\n' : '\n')))
    } else if (kind === 'random') out.push((x >> 3) & 0xff)
    else {
      const byte = 97 + (x % 7)
      x = lcg(x)
      for (let r = 0; r < 1 + (x % 40); r++) out.push(byte)
    }
  }
  return Uint8Array.from(out.slice(0, n))
}

type Params = Readonly<Record<string, unknown>>
const inputOf = (p: Params): Uint8Array => recipe(p['kind'] as string, p['n'] as number, p['seed'] as number)
const options = (p: Params): I.DeflateOptions => ({
  level: p['level'] as number,
  ...(p['strategy'] === undefined ? {} : { strategy: p['strategy'] as I.DeflateStrategy }),
})
const digest = (b: Uint8Array): number[] => [I.crc32(b), b.length]

describe('information/deflate byte for byte against zlib and gzip', () => {
  for (const c of loadFixture('information/deflate').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'gunzip') {
        const packed = Uint8Array.from(toFlat(decode(c.inputs[0]) as Tensor))
        expect(Array.from(I.gzipDecompress(packed))).toEqual(toFlat(decode(c.output) as Tensor))
        return
      }
      const input = inputOf(p)
      const want = toFlat(decode(c.output) as Tensor)
      if (c.fn === 'checksums') {
        expect([I.crc32(input), I.adler32(input)]).toEqual(want)
        return
      }
      if (c.fn === 'zlibDigest') {
        const packed = I.zlibCompress(input, options(p))
        expect(digest(packed)).toEqual(want)
        expect(I.zlibDecompress(packed)).toEqual(input)
        return
      }
      const python = Uint8Array.from(want)
      let ours: Uint8Array
      if (c.fn === 'zlib') {
        ours = I.zlibCompress(input, options(p))
        expect(I.zlibDecompress(python)).toEqual(input)
      } else if (c.fn === 'deflate') {
        ours = I.deflate(input, options(p))
        expect(I.inflate(python)).toEqual(input)
      } else {
        ours = I.gzipCompress(input, {
          level: p['level'] as number,
          mtime: p['mtime'] as number,
          ...(p['filename'] === undefined ? {} : { name: p['filename'] as string }),
        })
        expect(I.gzipDecompress(python)).toEqual(input)
      }
      expect(Array.from(ours)).toEqual(Array.from(python))
    })
  }
})
