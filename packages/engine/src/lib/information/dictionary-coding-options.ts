/**
 * The slots of the information-theory algorithms that vary as option tables: the LZ77 window (a tiny demonstration
 * window, the usual 4 KiB one, a 32 KiB deflate-sized one) with its longest match, and the Reed-Solomon field and code
 * (GF(2^4) for demonstrations, GF(2^8) for bytes, GF(2^10)) with the parity symbols as a knob. Each table is one `union`
 * space for controls and grids.
 */

import { dict, int, type Option, type OptionTable, optionTable, plainOption } from '../space/index.ts'
import type { Lz77Options } from './dictionary-coding.ts'
import { type ReedSolomonCode, reedSolomon } from './reed-solomon.ts'

function windowOption(
  description: string,
  window: number,
  maxLength: number,
): Option<Lz77Options, Record<string, never>> {
  return plainOption(
    description,
    dict({
      window: int(1, 65536, { default: window, label: 'window' }),
      maxLength: int(1, 1024, { default: maxLength, label: 'longest match' }),
    }),
    (k) => ({ window: k('window'), maxLength: k('maxLength') }),
  )
}

/**
 * The LZ77 windows as one slot: `tiny` (16 symbols back, to watch matches form), `standard` (4096 back, matches up to
 * 255: the default of `lz77Encode`) and `deflate` (32768 back, matches up to 258 as DEFLATE), each with the window and
 * longest match as knobs.
 *
 * @concept dictionary-coding
 * @example
 * ```ts
 * import { lz77Windows } from 'omai-engine/information'
 * lz77Windows.make({ kind: 'deflate' }) // => { window: 32768, maxLength: 258 }
 * ```
 */
export const lz77Windows: OptionTable<Lz77Options, Record<string, never>> = optionTable(
  {
    tiny: windowOption('16 symbols back: matches are short and local', 16, 32),
    standard: windowOption('4096 symbols back, matches up to 255', 4096, 255),
    deflate: windowOption('32 KiB back, matches up to 258, as DEFLATE', 32768, 258),
  },
  { default: 'standard', label: 'LZ77 window', description: 'How far back, and how long, a match may be.' },
)

/**
 * The Reed-Solomon fields as one slot: `gf16` (GF(2^4), words of up to 15 symbols), `gf256` (GF(2^8), bytes as in CDs and
 * QR codes) and `gf1024` (GF(2^10)), each with the number of parity symbols and the first consecutive root as knobs. A
 * table's `make` gives the `ReedSolomonCode`.
 *
 * @concept reed-solomon
 * @example
 * ```ts
 * import { reedSolomonFields } from 'omai-engine/information'
 * reedSolomonFields.make({ kind: 'gf16', nsym: 4 }).field.size // => 16
 * ```
 */
export const reedSolomonFields: OptionTable<ReedSolomonCode, Record<string, never>> = optionTable(
  {
    gf16: plainOption(
      'GF(2^4): words of up to 15 symbols, to see the algebra',
      dict({
        nsym: int(1, 13, { default: 4, label: 'parity symbols' }),
        fcr: int(0, 4, { default: 0, label: 'first root' }),
      }),
      (k) => reedSolomon(k('nsym'), { m: 4, fcr: k('fcr') }),
    ),
    gf256: plainOption(
      'GF(2^8): bytes, words of up to 255 symbols',
      dict({
        nsym: int(1, 253, { default: 8, label: 'parity symbols' }),
        fcr: int(0, 4, { default: 0, label: 'first root' }),
      }),
      (k) => reedSolomon(k('nsym'), { m: 8, fcr: k('fcr') }),
    ),
    gf1024: plainOption(
      'GF(2^10): words of up to 1023 symbols',
      dict({
        nsym: int(1, 1021, { default: 16, label: 'parity symbols' }),
        fcr: int(0, 4, { default: 0, label: 'first root' }),
      }),
      (k) => reedSolomon(k('nsym'), { m: 10, fcr: k('fcr') }),
    ),
  },
  { default: 'gf256', label: 'Reed-Solomon field', description: 'The finite field the code works over.' },
)
