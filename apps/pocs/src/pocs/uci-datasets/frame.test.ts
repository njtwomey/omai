import { datasetRegistry, findDataset, loadDataset } from 'mdml-engine/datasets'
import { describe, expect, it } from 'vitest'
import { summarise, toFrame } from './frame.ts'

const frameOf = (name: string) => {
  const entry = findDataset(datasetRegistry, name)
  if (entry === undefined) throw new Error(`no dataset ${name}`)
  const frame = toFrame(loadDataset(entry).data)
  if (frame === undefined) throw new Error(`${name} has no columns`)
  return frame
}

describe('toFrame', () => {
  it('lists the features of a classification set and its classes by name', () => {
    const f = frameOf('iris')
    expect(f.rows).toBe(150)
    expect(f.columns).toHaveLength(5)
    const target = f.columns.at(-1)
    expect(target?.kind).toBe('label')
    expect(target?.target).toBe(true)
    expect(new Set<unknown>(target?.values)).toEqual(new Set(['setosa', 'versicolor', 'virginica']))
  })
  it('lists the columns of a table, keeping its missing values missing', () => {
    const f = frameOf('heartDisease')
    expect(f.rows).toBe(303)
    expect(f.columns.reduce((n, c) => n + summarise(c).missing, 0)).toBe(6)
  })
  it('gives nothing for data that is not a table of columns', () => {
    const entry = findDataset(datasetRegistry, 'airPassengers')
    expect(entry && toFrame(loadDataset(entry).data)).toBeUndefined()
  })
})

describe('summarise', () => {
  it('bins a numeric column and reports its range', () => {
    const s = summarise({ name: 'x', kind: 'numeric', values: [0, 1, 2, 3, Number.NaN], target: false }, 3)
    expect(s.missing).toBe(1)
    expect(s.range).toEqual([0, 1.5, 3])
    expect(s.bins?.reduce((a, b) => a + b, 0)).toBe(4)
  })
  it('counts the levels of a label column, commonest first', () => {
    const s = summarise({ name: 'c', kind: 'label', values: ['a', 'b', 'a', null, 'a', 'b'], target: false }, 12, 1)
    expect(s.levels).toEqual([['a', 3]])
    expect(s.levelCount).toBe(2)
    expect(s.missing).toBe(1)
  })
})
