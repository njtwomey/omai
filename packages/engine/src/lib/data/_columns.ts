import { WhiteboxError } from '../base/index.ts'
import { concat, fromStorage, isTensor, type Tensor, take as takeRows } from '../tensor/index.ts'
import type { TableColumn } from './schema/tabular.ts'

// Row operations on per-row fields: tensors (first axis), plain arrays, and the columns of a table.

function indexTensor(indices: Int32Array): Tensor<'int32'> {
  return fromStorage(indices, [indices.length], 'int32')
}

function takeTableColumn(c: TableColumn, idx: Int32Array): TableColumn {
  switch (c.kind) {
    case 'numeric':
      return { ...c, values: takeRows(c.values, indexTensor(idx), 0) as Tensor<'float64'> }
    case 'categorical':
      return { ...c, codes: takeRows(c.codes, indexTensor(idx), 0) as Tensor<'int32'> }
    case 'boolean':
      return { ...c, values: takeRows(c.values, indexTensor(idx), 0) as Tensor<'bool'> }
    case 'text':
      return { ...c, values: Array.from(idx, (i) => c.values[i] ?? null) }
  }
}

/** The entries of a per-row field at the given rows. */
export function takeColumn(column: unknown, idx: Int32Array): unknown {
  if (isTensor(column)) {
    if (idx.length === 0) {
      const shape = [0, ...column.shape.slice(1)]
      return column.backend.wrap(column.dtype, shape, column.data.slice(0, 0))
    }
    return takeRows(column, indexTensor(idx), 0)
  }
  if (Array.isArray(column)) return Array.from(idx, (i) => column[i])
  if (typeof column === 'object' && column !== null) {
    const out: Record<string, TableColumn> = {}
    for (const [name, c] of Object.entries(column as Record<string, TableColumn>)) out[name] = takeTableColumn(c, idx)
    return out
  }
  throw new WhiteboxError('schema', 'take: a per-row field must be a tensor, an array or table columns')
}

function concatTableColumn(parts: readonly TableColumn[]): TableColumn {
  const first = parts[0] as TableColumn
  switch (first.kind) {
    case 'numeric':
      return {
        ...first,
        values: concat(
          parts.map((p) => (p as typeof first).values),
          0,
        ) as Tensor<'float64'>,
      }
    case 'categorical': {
      for (const p of parts) {
        if (p.kind !== 'categorical' || p.levels.join('\u0000') !== first.levels.join('\u0000'))
          throw new WhiteboxError('value', 'concatRows: categorical columns must share their levels')
      }
      return {
        ...first,
        codes: concat(
          parts.map((p) => (p as typeof first).codes),
          0,
        ) as Tensor<'int32'>,
      }
    }
    case 'boolean':
      return {
        ...first,
        values: concat(
          parts.map((p) => (p as typeof first).values),
          0,
        ) as Tensor<'bool'>,
      }
    case 'text':
      return { ...first, values: parts.flatMap((p) => (p as typeof first).values) }
  }
}

/** Per-row fields stacked along the row axis. */
export function concatColumns(parts: readonly unknown[]): unknown {
  const first = parts[0]
  if (isTensor(first)) return concat(parts as Tensor[], 0)
  if (Array.isArray(first)) return (parts as unknown[][]).flat(1)
  if (typeof first === 'object' && first !== null) {
    const out: Record<string, TableColumn> = {}
    for (const name of Object.keys(first)) {
      out[name] = concatTableColumn(
        parts.map((p) => {
          const c = (p as Record<string, TableColumn>)[name]
          if (c === undefined) throw new WhiteboxError('value', `concatRows: column ${name} is missing`)
          return c
        }),
      )
    }
    return out
  }
  throw new WhiteboxError('schema', 'concatRows: a per-row field must be a tensor, an array or table columns')
}
