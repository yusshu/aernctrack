def clamp(v, minimum, maximum): return max(minimum, min(v, maximum))


def shift_hsv(color, shift):
    return (
        clamp(color[0] + shift, 0, 179),
        clamp(color[1] + shift, 0, 255),
        clamp(color[2] + shift, 0, 255)
    )
