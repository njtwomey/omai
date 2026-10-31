/**
 * The solver slot of the maximum-likelihood fits of this folder (multinomial, ordinal, count and mixed models): each
 * model states its objective as a `SmoothProblem` and a solver of `smoothSolvers` is its algorithm. The helper itself
 * lives in `optim/slot.ts`, shared by every folder; this file re-exports the parts the folder's models and the
 * kernel-methods slot import.
 */

export { infNorm } from '../optim/minimize/objective.ts'
export { newtonAt, type SlotSpec, type SlotState, slotAlgorithm } from '../optim/slot.ts'
