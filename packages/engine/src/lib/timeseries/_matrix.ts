/**
 * Small dense matrices as arrays of rows, for the multivariate time-series models: conversion to and from tensors,
 * products, transposes, inverses, Cholesky factors and log-determinants of symmetric positive definite matrices,
 * through the library's linear algebra.
 */

import { cholesky, dgemm, eigvals, inv, slogdet } from '../numerics/linalg/index.ts'
import { fromStorage, toFlat, type Value } from '../tensor/index.ts'

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
