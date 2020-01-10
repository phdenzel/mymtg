import numpy as np
from scipy.spatial import distance
import pprint
import cv2
from base64 import b64decode

import mtg
from mtg.cards import Card


def display_image(image, name='image', flip=True, with_loop=True, zoom=0.75,
                  contours=[], cwidth=3, ccolor=(55, 205, 0)):
    """
    Show image, optionally with contours

    Args:
        image <np.ndarray> - a cv2 image data array

    Kwargs:
        name <str> - name for the display window
        zoom <float> - window zoom
        contours <list> - list of contours
        cwidth <int> - width of contour lines; -1 signifies filled contours

    Return:
        None
    """
    h, w = image.shape[:2]

    def project(image):
        cv2.imshow(name, image)
        if contours:
            if len(image.shape) > 2 and image.shape[-1] == 3:
                bgr = image.copy()
                if bgr.shape[-1] == 4:
                    bgr = cv2.cvtColor(bgr, cv2.BGRA2BGR)
            else:
                bgr = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(bgr, contours, -1, ccolor, cwidth)
            if flip:
                bgr = cv2.flip(bgr, 1)
            cv2.imshow(name, bgr)
        cv2.resizeWindow(name, int(w*zoom), int(h*zoom))

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    while with_loop:
        project(image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    if not with_loop:
        project(image)


def remove_glare(img):
    """
    Reduce the effect of glaring in the image

    Args:
        img <np.ndarray> - a cv2 image data array

    Kwargs:
        None

    Return:
        img <np.ndarray> - a cv2 image data array
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(img_hsv)
    # find all pixels with low saturation
    low_sat = (s < 32) * 255
    # slightly decrease the area of the low-satuared pixels by an erosion iteration
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    low_sat = cv2.erode(low_sat.astype(np.uint8), kernel)
    # set all brightness values, where the pixels are still saturated to 0
    v[low_sat == 0] = 0
    # filter out very bright pixels
    glare = (v > 200) * 255
    # slightly increase the area for each pixel
    glare = cv2.dilate(glare.astype(np.uint8), kernel)
    glare_reduced = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 200
    glare = cv2.cvtColor(glare, cv2.COLOR_GRAY2BGR)
    corrected = np.where(glare, glare_reduced, img)
    return corrected


def enhance_color(img):
    """
    Enhance colors and increase contrast

    Args:
        img <np.ndarray> - a cv2 image data array

    Kwargs:
        None

    Return:
        img <np.ndarray> - a cv2 image data array
    """
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    # convert from BGR to LAB color space and CLAHE the l-channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def equalize(img):
    """
    Equalize the histogram

    Args:
        img <np.ndarray> - a cv2 image data array

    Kwargs:
        None

    Return:
        img <np.ndarray> - a cv2 image data array
    """
    # convert from BGR to YUV color space and apply equalization to the y-channel
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return img
    


def detect_cards(image, min_focus=120, grayscale=True, correct_glare=False,
                 blur_radius=None, dilate_radius=None, thresh_val=80,
                 min_hyst=80, max_hyst=200,
                 verbose=False):
    """
    Detect card in image

    Args:
        image <np.ndarray> - a cv2 image data array

    Kwargs:
        min_focus <int> - minimally acceptable variance of the Laplacian
        grayscale <bool> - use grayscale image for canny-edge detection
        correct_glare <bool> -
        blur_radius <int> -
        dilate_radius <int> -
        thresh_val <int> -
        min_hyst <int> -
        max_hyst <int> -
        min_line_length <int> -
        max_line_gap <int> -
        verbose <bool> - print statements to the command-line

    Return:
        contours <list> - the bounding box coordinates and the card contours inside

    Note:
        Source: A. Rosebrock:
        https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    # check image focus
    focus = cv2.Laplacian(image, cv2.CV_64F).var()
    if focus < min_focus:
        if verbose:
            return None, []
    mindim = min(image.shape[0:2])
    if blur_radius is None:
        blur_radius = int(mindim / 100 + 0.5) // 2 * 2 + 1
    if dilate_radius is None:
        dilate_radius = int(mindim / 67 + 0.5)
    thresh_radius = int(mindim / 20 + 0.5) // 2 * 2 + 1
    # convert to grayscale
    img = image.copy()
    if correct_glare:
        img = remove_glare(img)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif grayscale < 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (255 - img)
    # noise-reducing and edge-preserving filter
    img = cv2.bilateralFilter(img, blur_radius, 25, 25)
    # medianBlur to reduce texture of the background
    img = cv2.medianBlur(img, blur_radius)
    # works good for border detection
    if grayscale:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, thresh_radius, 20)
    else:
        img = cv2.Canny(img, 40, 250)
    kernel = np.ones((dilate_radius, dilate_radius), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    # find contours
    img = img.copy()
    # img = cv2.Canny(img, 40, 250)
    contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # largest edge detection should be card
    return contours


def detect_exp(image, min_focus=120, grayscale=True, correct_glare=False,
               blur_radius=None, dilate_radius=None, thresh_val=80,
               min_hyst=80, max_hyst=200,
               min_line_length=None, max_line_gap=None,
               verbose=False):
    """
    Card detection experiment

    Args:
        image <np.ndarray> - a cv2 image data array

    Kwargs:
        min_focus <int> - minimally acceptable variance of the Laplacian
        grayscale <bool> - use grayscale image for canny-edge detection
        correct_glare <bool> -
        blur_radius <int> -
        dilate_radius <int> -
        thresh_val <int> -
        min_hyst <int> -
        max_hyst <int> -
        min_line_length <int> -
        max_line_gap <int> -
        verbose <bool> - print statements to the command-line

    Return:
        contours <list> - the bounding box coordinates and the card contours inside

    Note:
        Source: A. Rosebrock:
        https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    # check image focus
    focus = cv2.Laplacian(image, cv2.CV_64F).var()
    if focus < min_focus:
        if verbose:
            return None, []
    mindim = min(image.shape[0:2])
    if blur_radius is None:
        blur_radius = int(mindim / 100 + 0.5) // 2 * 2 + 1
    if dilate_radius is None:
        dilate_radius = int(mindim / 67 + 0.5)
    thresh_radius = int(mindim / 20 + 0.5) // 2 * 2 + 1
    if min_line_length is None:
        min_line_length = mindim // 10
    if max_line_gap is None:
        max_line_gap = mindim // 10
    # convert to grayscale
    img = image.copy()
    if correct_glare:
        img = remove_glare(img)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif grayscale < 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (255 - img)
    # noise-reducing and edge-preserving filter
    img = cv2.bilateralFilter(img, blur_radius, 25, 25)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    msk = (np.abs(grad_x) + np.abs(grad_y)) > 40
    img[msk] = np.max(img)
    img[~msk] = 0
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, min_line_length, max_line_gap)
    for l in lines[:10]:
        x1, y1, x2, y2 = l[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_image(image, 'sobel', with_loop=False)
    # find contours
    img = img.copy()
    # img = cv2.Canny(img, 40, 250)
    contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # largest edge detection should be card
    return contours


def bbox_from_contour(contour, method='polygon', max_iter=100):
    """
    Calculate a bounding box with approximate aspect ratio
    """
    if method == 'polygon':
        n_iter = 0
        lb, ub = 0., 1.
        while True:
            n_iter += 1
            k = (lb + ub)/2.
            eps = k * cv2.arcLength(contour, True)
            box = cv2.approxPolyDP(contour, eps, True)
            if n_iter > max_iter:
                break
            if len(box) > 4:
                lb = (lb + ub)/2.
            elif len(box) < 4:
                ub = (lb + ub)/2.
            else:
                break
        box = cv2.convexHull(box)
    elif method == '':
        pass
    else:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    return box


def validate_detection(image, box, aspect=Card.aspect, acceptance=0.15):
    """
    Use perspective to validate detection
    Inspired by Zhang et al.: https://doi.org/10.1016/j.dsp.2006.05.006
    """
    valid = False
    flip = False
    if len(box) == 4:
        b = box.reshape((4, 2))
        b3 = np.insert(b, 2, 1, axis=1)
        v0, u0 = [d/2. for d in image.shape[:-1]]
        # get aspect ratio of the quadrangle in perspective
        dist = [distance.euclidean(b[i], b[(i+1) % len(b)]) for i in range(len(b))]
        w, h = min(dist), max(dist)
        aspect_orig = w / h
        # calculate focal distance
        m1, m2, m4, m3 = b3
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1
        n21, n22, n23 = n2
        n31, n32, n33 = n3
        f2 = ((1.0 / (n23*n33))
             * ( (n21*n31 - (n21*n33+n23*n31)*u0 + n23*n33*u0*u0)
                +(n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0)))
        f = np.sqrt(np.abs(f2))
        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')
        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)
        # calculate the real aspect ratio
        aspect_real = np.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) /
                              np.dot(np.dot(np.dot(n3, Ati), Ai), n3))
        # aspect_real = aspect_estimate
        flip = (aspect_real > 1)
        if flip:
            aspect_real = 1. / aspect_real
        # valid if aspect_real within relative acceptance of aspect
        if abs(aspect - aspect_real)/aspect < acceptance:
            valid = True
    return valid, flip


def get_warp(image, frame, rotate=False, dimensions=Card.dimensions, mm2px=8,
             blur_radius=None, dilate_radius=None, thresh_val=80,
             min_hyst=80, max_hyst=200,
             min_line_length=None, max_line_gap=None):
    """
    """
    dimensions = mm2px * dimensions
    cardW, cardH = dimensions[2]
    box = np.array(frame).astype('float32')
    mindim = min(image.shape[0:2])
    if blur_radius is None:
        blur_radius = int(mindim / 100 + 0.5) // 2 * 2 + 1
    if dilate_radius is None:
        dilate_radius = int(mindim / 67 + 0.5)
    thresh_radius = int(mindim / 20 + 0.5) // 2 * 2 + 1
    if min_line_length is None:
        min_line_length = mindim // 10
    if max_line_gap is None:
        max_line_gap = mindim // 10
    if rotate:
        dimensions = np.roll(dimensions, -1, axis=0)
    transform = cv2.getPerspectiveTransform(box, dimensions)
    warp = cv2.warpPerspective(image, transform, (cardW, cardH))
    # clean up the image
    
    return warp


if __name__ == "__main__":
    # graphics setup
    # orb = cv2.ORB_create()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # get image data
    # scry = mtg.scryfall.ScryfallCardName('atraxa', format='image')
    # decode and convert image byte string
    # img_b64 = b64decode(scry.data)
    # img_np = np.asarray(bytearray(scry.data), dtype=np.uint8)
    # img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    warp = None
    cap = cv2.VideoCapture(0)
    while True:

        # capture frame-by-frame
        ret, img = cap.read()
        img = remove_glare(img)
        img = equalize(img)
        img = enhance_color(img)

        # detect cards
        contours = detect_cards(img.copy())
        # contours = detect_cards2(img.copy())
        contour = max(contours, key=cv2.contourArea)  # assume card is largest object
        box = bbox_from_contour(contour)  # get bounding box around contour
        valid, flip = validate_detection(img.copy(), box)
        if valid:
            frame = [box]
            # get warp transformation
            warp = get_warp(img.copy(), box, rotate=flip)
            warp = cv2.bilateralFilter(warp, 7, 25, 25)
            warp = enhance_color(warp)
        else:
            frame = []
        display_image(img.copy(), contours=frame, name='detection', with_loop=False, flip=False)
        if warp is not None:
            display_image(warp, name='warp', with_loop=False, flip=False)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # inspect image
    # display_image(img, name='image')
    # display_image(img, contours=[contours[1]], name='boxed')
    # display_image(warp, name='warp')
