/** One line of a program's recovery check: what was measured, the tolerance and whether it holds. */
export interface CheckRow {
  readonly label: string
  readonly value: string
  readonly tolerance: string
  readonly pass: boolean
  /** False for a line that is shown but does not decide the badge (a model known to be wrong for this data). */
  readonly gating?: boolean
}

/** The recovery check of a program run: the rows, and notes that say honestly where EP is poor and why. */
export interface ProgramCheck {
  readonly rows: readonly CheckRow[]
  readonly notes: readonly string[]
}

/** Whether every gating row passes. */
export const passes = (check: ProgramCheck): boolean => check.rows.every((r) => r.gating === false || r.pass)

/** A number to `digits` significant digits, a dash when it is not one. */
export const sig = (v: number, digits = 3): string => (Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '-')
