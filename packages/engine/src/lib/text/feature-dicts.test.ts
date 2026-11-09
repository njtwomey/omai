import { describe, expect, it } from 'vitest'
import { toDense } from '../numerics/sparse/index.ts'
import { toArray } from '../tensor/index.ts'
import { dictInverse, dictTransform, dictVectorize, featureHash } from './feature-dicts.ts'
import { hashingVectorize } from './features.ts'

describe('dictVectorize', () => {
  const records = [
    { city: 'Paris', temp: 12, tags: ['rain', 'wind'] },
    { city: 'Rome', temp: 20, ok: true },
  ]

  it('numbers and booleans are columns, strings and lists one-hot, columns sorted by name', () => {
    const { vectorizer, matrix } = dictVectorize(records)
    expect(vectorizer.featureNames).toEqual(['city=Paris', 'city=Rome', 'ok', 'tags=rain', 'tags=wind', 'temp'])
    expect(toArray(toDense(matrix))).toEqual([
      [1, 0, 0, 1, 1, 12],
      [0, 1, 1, 0, 0, 20],
    ])
  })

  it('keeps the order of first appearance when sort is off, and takes another separator', () => {
    const { vectorizer } = dictVectorize(records, { sort: false, separator: ':' })
    expect(vectorizer.featureNames).toEqual(['city:Paris', 'temp', 'tags:rain', 'tags:wind', 'city:Rome', 'ok'])
  })

  it('new records ignore what the vocabulary lacks, and the inverse names the non-zero columns', () => {
    const { vectorizer, matrix } = dictVectorize(records)
    expect(toArray(toDense(dictTransform(vectorizer, [{ city: 'Oslo', temp: 5, extra: 1 }])))).toEqual([
      [0, 0, 0, 0, 0, 5],
    ])
    expect(dictInverse(vectorizer, matrix)[0]).toEqual({ 'city=Paris': 1, 'tags=rain': 1, 'tags=wind': 1, temp: 12 })
    expect(dictInverse(vectorizer, [[0, 0, 0, 0, 0, 7]])).toEqual([{ temp: 7 }])
    expect(() => dictInverse(vectorizer, [[1, 2]])).toThrow()
  })

  it('refuses a list that holds numbers', () => {
    expect(() => dictVectorize([{ a: [1, 2] as unknown as string[] }])).toThrow()
  })
})

describe('featureHash', () => {
  it('sums collisions and takes the sign of the hash', () => {
    const m = toArray(toDense(featureHash([{ foo: 1, bar: 2 }, { foo: 3 }], { nFeatures: 4 })))
    expect(m).toEqual([
      [-1, 2, 0, 0],
      [-3, 0, 0, 0],
    ])
    const plain = toArray(toDense(featureHash([{ foo: 1, bar: 2 }], { nFeatures: 4, alternateSign: false })))
    expect(plain).toEqual([[1, 2, 0, 0]])
  })

  it('a string value is the feature name=value with value 1, and zeros are dropped', () => {
    const a = toArray(toDense(featureHash([{ colour: 'red', zero: 0 }], { nFeatures: 32 })))
    const b = toArray(toDense(featureHash([['colour=red']], { nFeatures: 32 })))
    expect(a).toEqual(b)
  })

  it('on bare strings it is the unnormalised hashing vectoriser of the same features', () => {
    const viaText = toArray(toDense(hashingVectorize(['foo bar foo'], { nFeatures: 16, norm: 'none' })))
    expect(toArray(toDense(featureHash([['foo', 'bar', 'foo']], { nFeatures: 16 })))).toEqual(viaText)
  })

  it('refuses a bad width', () => {
    expect(() => featureHash([], { nFeatures: 0 })).toThrow()
  })
})
