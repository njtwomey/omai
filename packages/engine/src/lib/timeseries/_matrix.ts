/**
 * Small dense matrices as arrays of rows, for the multivariate time-series models: conversion to and from tensors,
 * products, transposes, inverses, Cholesky factors and log-determinants of symmetric positive definite matrices,
 * through the library's linear algebra.
 */

import { cholesky, dgemm, eigh, eigvals, inv, slogdet } from '../numerics/linalg/index.ts'
import { fromStorage, toFlat, type Value } from '../tensor/index.ts'
import { leastSquares } from './_ml.ts'

/** A matrix as an array of rows. */
export type Matrix = number[][]

/** The tensor of a matrix. */
export function tensorOf(a: readonly (readonly number[])[]): Value {
  const rows = a.length
  const cols = a[0]?.length ?? 0
  return fromStorage(Float64Array.from(a.flat()), [rows, cols], 'float64')
}

/** The rows of a matrix tensor. */
export function rowsOf(t: Value, rows: number, cols: number): Matrix {
  const flat = toFlat(t)
  return Array.from({ length: rows }, (_, i) => flat.slice(i * cols, (i + 1) * cols))
}

/** The identity of size n. */
export function identity(n: number): Matrix {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
}

/** The product A B. */
export function multiply(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): Matrix {
  const m = a.length
  const k = b.length
  const n = b[0]?.length ?? 0
  const out = dgemm(a.flat(), b.flat(), m, k, n)
  return Array.from({ length: m }, (_, i) => Array.from(out.subarray(i * n, (i + 1) * n)))
}

/** The transpose. */
export function transpose(a: readonly (readonly number[])[]): Matrix {
  const n = a[0]?.length ?? 0
  return Array.from({ length: n }, (_, j) => a.map((row) => row[j] ?? 0))
}

/** The sum of two matrices (or the difference with `sign = -1`). */
export function add(a: readonly (readonly number[])[], b: readonly (readonly number[])[], sign = 1): Matrix {
  return a.map((row, i) => row.map((v, j) => v + sign * (b[i]?.[j] ?? 0)))
}

/** The inverse of a square matrix. */
export function invert(a: readonly (readonly number[])[]): Matrix {
  return rowsOf(inv(tensorOf(a)), a.length, a.length)
}

/** The lower Cholesky factor of a symmetric positive definite matrix. */
export function choleskyLower(a: readonly (readonly number[])[]): Matrix {
  return rowsOf(cholesky(tensorOf(a)), a.length, a.length)
}

/** The log-determinant of a symmetric positive definite matrix. */
export function logDeterminant(a: readonly (readonly number[])[]): number {
  return toFlat(slogdet(tensorOf(a)).logabsdet)[0] ?? Number.NaN
}

/** The moduli of the eigenvalues of a square matrix. */
export function eigenvalueModuli(a: readonly (readonly number[])[]): number[] {
  const { re, im } = eigvals(tensorOf(a))
  const r = toFlat(re)
  const i = toFlat(im)
  return r.map((v, k) => Math.hypot(v, i[k] ?? 0))
}

/** The eigenvalues (largest first) and orthonormal eigenvectors (as the columns of `vectors`) of a symmetric matrix. */
export function symmetricEigen(a: readonly (readonly number[])[]): { values: number[]; vectors: Matrix } {
  const n = a.length
  const sym = a.map((r, i) => r.map((v, j) => 0.5 * (v + (a[j]?.[i] ?? 0))))
  const { eigenvalues, eigenvectors } = eigh(tensorOf(sym))
  const w = toFlat(eigenvalues)
  const v = rowsOf(eigenvectors, n, n)
  const order = Array.from({ length: n }, (_, i) => n - 1 - i)
  return { values: order.map((i) => w[i] ?? 0), vectors: v.map((row) => order.map((i) => row[i] ?? 0)) }
}

/** The symmetric inverse square root of a symmetric positive definite matrix. */
export function inverseSquareRoot(a: readonly (readonly number[])[]): Matrix {
  const { values, vectors } = symmetricEigen(a)
  const scaled = vectors.map((row) => row.map((v, j) => v / Math.sqrt(values[j] ?? 1)))
  return multiply(scaled, transpose(vectors))
}

/** The columns of a matrix given as rows. */
export function columnsOf(a: readonly (readonly number[])[]): Float64Array[] {
  const n = a[0]?.length ?? 0
  return Array.from({ length: n }, (_, j) => Float64Array.from(a, (r) => r[j] ?? 0))
}

/** A matrix of rows from its columns. */
export function fromColumns(c: readonly ArrayLike<number>[]): Matrix {
  const n = c[0]?.length ?? 0
  return Array.from({ length: n }, (_, i) => c.map((col) => col[i] ?? 0))
}

/** The residuals of the columns of `y` after least squares on the columns of `x` (`y` itself when `x` is empty). */
export function residualsOn(y: readonly ArrayLike<number>[], x: readonly ArrayLike<number>[]): Float64Array[] {
  if (x.length === 0) return y.map((c) => Float64Array.from(c))
  return y.map((c) => {
    const beta = leastSquares(x, c)
    return Float64Array.from({ length: c.length }, (_, i) => {
      let fit = 0
      for (let j = 0; j < x.length; j++) fit += (beta[j] ?? 0) * (x[j]?.[i] ?? 0)
      return (c[i] ?? 0) - fit
    })
  })
}
