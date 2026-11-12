/**
 * The reference CPU backend, `cpu`: plain loops over row-major typed arrays implementing every kernel of the backend
 * contract, favouring clarity over speed: each kernel allocates a fresh output and never mutates its inputs.
 */

import { wrap } from './_cpu/_util.ts'
import { binary, cast, compare, map, map2, predicate, unary, where } from './_cpu/elementwise.ts'
import { argReduce, logsumexp, reduce, scan } from './_cpu/reduce.ts'
import { argsort } from './_cpu/sort.ts'
import {
  broadcastTo,
  concat,
  matmul,
  pad,
  putAlongAxisAdd,
  slice,
  sliceScatter,
  take,
  takeAlongAxis,
  takeScatter,
  transpose,
} from './_cpu/structure.ts'
import type { Backend } from './backend.ts'

/**
 * The reference CPU backend: plain loops over row-major typed arrays. It favours clarity over speed; every kernel
 * allocates a fresh output and never mutates its inputs.
 *
 * @returns The backend.
 * @concept tensor-backend
 * @example
 * ```ts
 * import { cpu } from 'omai-engine/tensor'
 * cpu.name // => 'cpu'
 * ```
 */
export const cpu: Backend = {
  name: 'cpu',
  wrap,
  unary,
  predicate,
  binary,
  compare,
  map,
  map2,
  reduce,
  logsumexp,
  argReduce,
  scan,
  matmul,
  transpose,
  broadcastTo,
  slice,
  sliceScatter,
  concat,
  pad,
  take,
  takeScatter,
  takeAlongAxis,
  putAlongAxisAdd,
  where,
  cast,
  argsort,
}
