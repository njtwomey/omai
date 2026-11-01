/**
 * What the stepped tokenisers log: BPE encoding logs each merge it applies, WordPiece each piece it takes, and Punkt
 * training the score of each abbreviation candidate.
 */

import { type Contract, defineContract, scalar, tensor } from '../trace/index.ts'

/** The paths the tokeniser algorithms log, as a type. */
export type TokenizerPath = 'tok/merge' | 'tok/piece' | 'tok/symbols' | 'punkt/score' | 'punkt/abbrev'

/**
 * What the tokeniser algorithms log. BPE encoding logs the rank of the merge applied and the symbols left; WordPiece the
 * length of the piece taken; Punkt training the score of each abbreviation candidate and whether it is kept (1) or not (0).
 */
export const tokenizerContract: Contract<unknown, TokenizerPath> = defineContract({
  'tok/merge': scalar({ description: 'the rank of the merge applied at this step (lower ranks go first)' }),
  'tok/piece': scalar({ description: 'the length, in code units, of the piece taken at this step' }),
  'tok/symbols': tensor({ role: 'state', shape: ['symbols'], description: 'the symbol ids of the word so far' }),
  'punkt/score': scalar({
    description: "an abbreviation candidate's score: log-likelihood times the length and period factors",
  }),
  'punkt/abbrev': scalar({ description: '1 when the candidate is an abbreviation after this step, else 0' }),
})
