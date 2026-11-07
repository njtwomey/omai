/**
 * The canonical fixed-size geometric tuples: `Vec2`, `Vec3` and `Vec4` (readonly tuples of numbers). A name with a
 * particular meaning (a point, a colour, a complex number, a quaternion, a plane) is a specialisation of one of these,
 * an alias or a labelled tuple of the same arity, never a restatement of the structure. Variable-length vectors and
 * matrices are tensors (`TensorLike` in, `Tensor` out) and are not defined here.
 */

/** A pair of numbers: a 2-D point or offset, or any two-component value. */
export type Vec2 = readonly [x: number, y: number]

/** A triple of numbers: a 3-D point or direction, or any three-component value. */
export type Vec3 = readonly [x: number, y: number, z: number]

/** Four numbers: a homogeneous point, a quaternion, a plane or a box, or any four-component value. */
export type Vec4 = readonly [x: number, y: number, z: number, w: number]
