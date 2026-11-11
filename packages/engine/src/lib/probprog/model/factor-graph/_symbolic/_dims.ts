/**
 * The named dimensions of the symbolic values: sizes of dimensions, the shape of a tensor over some, the cell of a value at an
 * index of a larger set of dimensions (broadcasting what it lacks), the join of several values' dimensions, and the
 * conversions between a known value of the registry (`NamedValue`) and a symbolic value (`Sym`) of constant cells. All are pure
 * functions of the sizes a model was bound with, and a size that is missing or a value that does not fit is a `Refusal`.
 */

import { fromStorage, toFlat } from '../../../../tensor/index.ts'
import type { NamedValue } from '../../operators.ts'
import { type Cell, constCell, isConstCell, prod, type Sym, unflatten } from '../_cells.ts'
import { Refusal } from '../_refusal.ts'

/** The sizes the model was bound with, by dimension name. */
export type Sizes = Readonly<Record<string, number>>

/** The size of a dimension; a `Refusal` when the model has no size for it. */
export function sizeOf(sizes: Sizes, dim: string): number {
  const n = sizes[dim]
  if (n === undefined)
    throw new Refusal('data', `the dimension '${dim}' has no size (bind data for it or fix its size)`)
  return n
}

/** The sizes of a list of dimensions. */
export function shapeOf(sizes: Sizes, dims: readonly string[]): number[] {
  return dims.map((d) => sizeOf(sizes, d))
}

/** The cell of `sym` at the index `idx` of the dimensions `outDims` (a dimension of `sym` that is not among them is a `Refusal`). */
export function cellAt(sizes: Sizes, sym: Sym, outDims: readonly string[], idx: readonly number[]): Cell {
  let flat = 0
  for (const d of sym.dims) {
    const at = outDims.indexOf(d)
    if (at < 0)
      throw new Refusal('structure', `a value over '${d}' does not fit the dimensions [${outDims.join(', ')}]`)
    flat = flat * sizeOf(sizes, d) + (idx[at] as number)
  }
  return sym.cells[flat] as Cell
}

/** The dimensions of several values, each once, in order of first appearance. */
export function joinDims(syms: readonly Sym[]): string[] {
  const dims: string[] = []
  for (const s of syms) for (const d of s.dims) if (!dims.includes(d)) dims.push(d)
  return dims
}

/** The value over the joined dimensions whose cell `i` is `f` of the cells of `syms` there (an element-wise operation, broadcasting). */
export function zipSyms(sizes: Sizes, syms: readonly Sym[], f: (cells: readonly Cell[], index: number) => Cell): Sym {
  const dims = joinDims(syms)
  const shape = shapeOf(sizes, dims)
  const cells: Cell[] = []
  for (let i = 0; i < prod(shape); i++) {
    const idx = unflatten(i, shape)
    cells.push(
      f(
        syms.map((s) => cellAt(sizes, s, dims, idx)),
        i,
      ),
    )
  }
  return { dims, cells }
}

/** The cells of a symbolic value laid out over a node's dimensions (broadcasting what it lacks). */
export function overDims(sizes: Sizes, sym: Sym, dims: readonly string[]): Cell[] {
  const shape = shapeOf(sizes, dims)
  const out: Cell[] = []
  for (let i = 0; i < prod(shape); i++) out.push(cellAt(sizes, sym, dims, unflatten(i, shape)))
  return out
}

/** A known value of the registry as a symbolic value of constant cells. */
export function fromNamed(nv: NamedValue): Sym {
  if (nv.value.shape.length !== nv.dims.length)
    throw new Refusal('structure', 'a value with event axes cannot stand in a scalar expression')
  return { dims: nv.dims, cells: toFlat(nv.value).map(constCell) }
}

/** A symbolic value whose cells are all constant as a known value; a random cell is a `Refusal`. */
export function toNamed(sizes: Sizes, sym: Sym): NamedValue {
  const shape = shapeOf(sizes, sym.dims)
  const values = sym.cells.map((c) => {
    if (!isConstCell(c)) throw new Refusal('node', 'a random value where a known one is needed')
    return c.aff.offset
  })
  return { value: fromStorage(Float64Array.from(values), shape, 'float64'), dims: sym.dims }
}
