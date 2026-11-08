/**
 * The mdml-engine version as a constant, kept equal to the package manifest (a test checks it) and recorded in every
 * traced run so that the run can be reproduced.
 */

// Keep in step with the version in packages/engine/package.json.

/**
 * The mdml-engine version, recorded in every trace run so a run can be reproduced.
 *
 * @returns The semver string of this release.
 * @example
 * ```ts
 * import { VERSION } from 'mdml-engine/base'
 * typeof VERSION // => 'string'
 * ```
 */
export const VERSION: string = '0.1.3'
