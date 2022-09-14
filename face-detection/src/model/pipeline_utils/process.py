import numpy as np
import cv2
from collections import OrderedDict


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))

    
])

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def crop_img(img, roi_box):

    def get_coord(roi_box):
        if roi_box.shape[0] == 4:
            sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
        else:
            sx, sy, ex, ey, _ = [int(round(_)) for _ in roi_box]
        
        return sx, sy, ex, ey

    h, w = img.shape[:2]

    try:
        sx, sy, ex, ey = get_coord(roi_box)
    except:
        roi_box = np.array(roi_box)
        sx, sy, ex, ey = get_coord(roi_box)
        
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


class FaceAligner:
    def __init__(
        self, shape, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None
    ):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        self.M = None
        self.shape = shape
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image):
        # convert the landmark (x, y)-coordinates to a NumPy array

        # simple hack ;)
        if len(self.shape) == 68:
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = self.shape[lStart:lEnd]
        rightEyePts = self.shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype(int).astype(float)
        rightEyeCenter = rightEyePts.mean(axis=0).astype(int).astype(float)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        if dist <= 0:
            return None, False
        else:
            desiredDist = desiredRightEyeX - self.desiredLeftEye[0]
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist

            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = (
                (leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) // 2,
            )

            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += tX - eyesCenter[0]
            M[1, 2] += tY - eyesCenter[1]
            self.M = M

            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            # return the aligned face
            return output, True

    def get_align_box(self, img_align):
        M = self.M  # affine matrix
        shape_trans = cv2.transform(np.array([self.shape]), M)[
            0
        ]  # transform facial landmark to new direction using info provided by align model
        shape_min_x = max(shape_trans[:, 0].min(), 0)
        shape_max_x = min(shape_trans[:, 0].max(), img_align.shape[1])
        shape_min_y = max(shape_trans[:, 1].min(), 0)
        shape_max_y = min(shape_trans[:, 1].max(), img_align.shape[0])

        (nStart, nEnd) = FACIAL_LANDMARKS_68_IDXS["nose"]
        nosePts = shape_trans[nStart:nEnd]
        noseCenter = nosePts.mean(axis=0).astype(int).astype(float)
        offset_H = 1.1 * max(shape_max_y - noseCenter[1], noseCenter[1] - shape_min_y)
        offset_W = 1.1 * max(shape_max_x - noseCenter[0], noseCenter[0] - shape_min_x)

        roi_box_final = np.array(
            [noseCenter[0] - offset_W, noseCenter[1] - offset_H, noseCenter[0] + offset_W, noseCenter[1] + offset_H, -1, ]
        )
        roi_box_final[0] = max(roi_box_final[0], 0)
        roi_box_final[1] = max(roi_box_final[1], 0)
        roi_box_final[2] = min(roi_box_final[2], img_align.shape[1])
        roi_box_final[3] = min(roi_box_final[3], img_align.shape[0])
        roi_box_final = roi_box_final.astype(int)  # use face landmark to crop image
        return roi_box_final