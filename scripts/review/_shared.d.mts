/** Types for the parts of `_shared.mjs` that the TypeScript scripts use. */

/** One one-line type alias found by {@link aliasLines}. */
export interface AliasLine {
  readonly line: number
  readonly name: string
  readonly exported: boolean
  readonly generic: boolean
  readonly shape: string
  readonly doc: string
  readonly optOut: boolean
}

export declare const ROOT: string
export declare function walk(dir: string, found?: string[]): string[]
export declare const isTest: (f: string) => boolean
export declare const isData: (f: string) => boolean
export declare function aliasLines(text: string): AliasLine[]
