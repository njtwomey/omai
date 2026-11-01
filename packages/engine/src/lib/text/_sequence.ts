// The sequences the edit-distance family compares: a string (by code point) or an array of tokens. Shared by
// `similarity.ts` and `dp-tables.ts`; not re-exported (the type is re-exported by `similarity.ts`).

/** A sequence to compare: a string (compared by code point) or an array of tokens. */
export type Sequence<T> = string | readonly T[]

/** The elements of a sequence: code points of a string, or the array itself. */
export function elements<T>(s: Sequence<T>): readonly (T | string)[] {
  return typeof s === 'string' ? Array.from(s) : s
}
