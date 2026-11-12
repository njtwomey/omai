/**
 * The eight schools of the story: the classic numbers of Rubin (1981), the estimated effect of a coaching programme on SAT scores
 * in eight schools and the standard error of each estimate (taken as known). They are given, not computed, so the page shows them
 * before anything runs and every run, whatever its seed, fits the same data.
 */

/** The estimated effects, school by school. */
export const EFFECTS: readonly number[] = [28, 8, -3, 7, -1, 1, 18, 12]

/** Their standard errors. */
export const ERRORS: readonly number[] = [15, 10, 16, 11, 9, 11, 10, 18]

/** The schools' names, in the order of the data. */
export const NAMES: readonly string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
