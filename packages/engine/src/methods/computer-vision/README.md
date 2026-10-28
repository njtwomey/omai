# Computer vision (part 7, application domains)

Classical vision watched (the six methods exist): image processing (filters, anisotropic diffusion, thresholds, morphology, edges), recognition (features, descriptors, boosting over Haar features, template matching, bag of visual words, segmentation by superpixels, active contours and GrabCut), tracking (Lucas-Kanade, mean shift), and camera geometry (homographies, epipolar geometry, calibration, structure from motion), as sessions over `vision`, `signal`, `graphics` and the geometric scenes of `datasets`.

## Capabilities

- Every iterative operation of `vision` as a method with the image on the tape and the picture as a probe (the pattern of the six).
- Filters and edges: a filter as a slot (Gaussian, bilateral, median, Sobel, Canny's stages), the response at each stage.
- Features: Harris corners, SIFT-like keypoints and descriptors, matching between two views with RANSAC's inliers appearing (`stats` has RANSAC).
- Geometry: a homography estimated from matches, the epipolar lines for a point, camera calibration from a planted grid, two-view reconstruction.
- Recognition: bag of visual words, a detector by sliding window over the boosted classifier, non-maximum suppression.
- Convolutional networks for recognition and segmentation live in `convolutional-networks`.

## Examples to make

- Canny edge detection stage by stage: smoothing, gradients, non-maximum suppression, hysteresis.
- Keypoint matching between two views with RANSAC's inliers and the homography's warp.
- Epipolar lines moving as a point is dragged in the left view.
- A sliding-window detector over a scene with the boosted classifier, the detections and their suppression.

## Rendering

- Pictures with points and masks (the `picture` helper), image steps, side-by-side views with lines, boards.
- Missing: a two-image panel with correspondences drawn between them; an interactive point on an image (drag a point, the probe updates).

## Shared

- `shared/`: `_scene.ts` and `_picture.ts` exist and generalise into `methods/shared/pictures.ts`; the image data slot with truth (scenes, shapes, masks, corners).

## Problems

- `vision/` moves here with its six methods. Geometry needs `graphics` cameras and `numerics` (eight-point, DLT) pieces that exist in part.
