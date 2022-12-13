import cv2

def crop_around_center(image: object, width: int, height: int, crop_center: object = None):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point.

    Arguments:
    ----------
        image (3d-array): The original image vector.

        width (int): The desired width of the cropped image.

        height (int): The desired height of the cropped image.

        crop_center: ([x, y]): The coordinates of the center point to crop.

    Returns:
    --------
        image (3d-array): Cropped image.

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])

    # 未提供中心座標的時候，預設為圖片正中心
    if(crop_center is None):
        image_center = [int(image_size[0] * 0.5), int(image_size[1] * 0.5)]
    else:
        image_center = [int(image_size[0] * 0.5 + crop_center[0]),
                        int(image_size[1] * 0.5 + crop_center[1])]

    # 如果輸出寬度大於圖片寬度，則將輸出寬度設為圖片寬度
    if(width > image_size[0]):
        width = image_size[0]

    # 如果輸出高度大於圖片高度，則將輸出高度設為圖片高度
    if(height > image_size[1]):
        height = image_size[1]

    # 如果從 x 座標算起，輸出寬度會超出圖片邊界，則將 x 座標往左或往右推移
    x_boundary = (image_center[0] - int(width * 0.5),
                  image_center[0] + int(width * 0.5))

    # 超出左邊界 beyond the left border
    if(x_boundary[0] < 0):
        image_center[0] = image_center[0] + abs(x_boundary[0])

    # 超出右邊界 beyond the right border
    if(x_boundary[1] > image_size[0]):
        image_center[0] = image_center[0] - abs(x_boundary[1] - image_size[0])

    # 如果從 y 座標算起，輸出高度會超出圖片邊界，則將 y 座標往上或往下推移
    y_boundary = (image_center[1] - int(height * 0.5),
                  image_center[1] + int(height * 0.5))

    # 超出上邊界 beyond the upper bound
    if(y_boundary[0] < 0):
        image_center[1] = image_center[1] + abs(y_boundary[0])

    # 超出下邊界 beyond the lower boundary
    if(y_boundary[1] > image_size[1]):
        image_center[1] = image_center[1] - abs(y_boundary[1] - image_size[1])

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    return image[y1:y2, x1:x2]

def load_image_and_resize(
    path: str, new_size: int = 224,
    resize_method: str = "resize", padding_method: str = "wrap", rotate: bool = False,
    crop_ratio: float = 0.5, crop_center: object = None
):
    """Use CV2 to read image, and resize the image to make the aspect ratio.

    Arguments:
    ----------
        path (str): The image path.  
        
        new_size (int): The desired image size.  
        
        resize_method (str): 'resize', 'padding' or 'crop'
            resize: only resize.
            padding: padding and resize.
            crop: crop and resize.

        padding_method (str): 'wrap' or 'constant'
            wrap: take mirrored pixel padding.
            constant: border padding to a fixed value.
            
        crop_ratio (float): The ratio of the cropped area to the original image.  
        
        crop_center ([x, y]): The coordinates of the center point to crop.  
        
        rotate (bool): Whether to rotate the image, the rotation angle is fixed at 270 degrees.

    Returns:
    --------
        image (3d-array): Image vector converted to rgb format.
    """
    image = cv2.imread(path)
    height, width, channel = image.shape

    if resize_method == "padding":
        # resize by ratio
        ratio = new_size / max(height, width)
        new_height, new_width = int(ratio * height), int(ratio * width)
        image = cv2.resize(image, (new_width, new_height))
        # calculate boundaries
        top, bottom = (new_size - new_height) // 2, (new_size - new_height) // 2
        left, right = (new_size - new_width) // 2, (new_size - new_width) // 2
        bottom = bottom + 1 if (top + bottom + height < new_size) else bottom
        right = right + 1 if (left + right + width < new_size) else right
        # padding
        if padding_method == "wrap":
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_WRAP)
        elif padding_method == "constant":
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    elif resize_method == "crop":
        crop_window = min(height, width) * crop_ratio
        image = crop_around_center(image, crop_window, crop_window, crop_center)
        image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
    elif resize_method == "resize":
        image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotate else image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image