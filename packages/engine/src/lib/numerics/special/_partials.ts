// Shape-parameter derivatives of the incomplete gamma and beta functions as primitives of their own, shared by the
// functions built on them. Their own derivatives (second order in the shape) use finite differences.
import { special } from './_elementwise.ts'
import * as inc from './_incomplete.ts'

export const gammaincDa: ReturnType<typeof special> = special('gammaincDa', inc.gammaincDa, [null, null])
export const logGammaincDa: ReturnType<typeof special> = special('logGammaincDa', inc.logGammaincDa, [null, null])
export const logGammainccDa: ReturnType<typeof special> = special('logGammainccDa', inc.logGammainccDa, [null, null])
export const betaincDa: ReturnType<typeof special> = special('betaincDa', inc.betaincDa, [null, null, null])
export const betaincDb: ReturnType<typeof special> = special('betaincDb', inc.betaincDb, [null, null, null])
