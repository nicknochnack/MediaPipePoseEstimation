
def heel_crop(points, frame, frame_count_trim):

    x_coords, y_coords = zip(*points)
    x_padding = 0.06
    y_padding_above = 0.02
    y_padding_below = 0.04

    min_x = min(x_coords)
    min_y = min(y_coords)
    max_x = max(x_coords)
    max_y = max(y_coords)

    min_x -= x_padding
    min_y -= y_padding_above
    max_x += x_padding
    max_y += y_padding_below

    img_height, img_width, _ = frame.shape

    # Map normalized coordinates to pixel values
    min_x = min_x * img_width
    min_y = min_y * img_height
    max_x = max_x * img_width
    max_y = max_y * img_height

    min_x, min_y, max_x, max_y = map(round, [min_x, min_y, max_x, max_y])

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, img_width)
    max_y = min(max_y, img_height)

    image_trimmed = frame[min_y:max_y, min_x:max_x]
    frame_count_trim += 1

    # print([min_x,max_x,min_y,max_y])

    return image_trimmed, frame_count_trim

