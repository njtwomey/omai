/**
 * The toy methods the gallery's infrastructure recipes offer, by name: what a worker can make and what a tape file's
 * method name is looked up in.
 */

import { methodRegistry } from 'mdml-render/producer'
import { heavyDescent } from './heavy.ts'
import { orbit } from './orbit.ts'

/** The registry of the gallery's toy methods. */
export const toyRegistry = methodRegistry([heavyDescent, orbit])
