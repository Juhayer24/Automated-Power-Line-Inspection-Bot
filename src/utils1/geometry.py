import math

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def pixel_to_angle(center_x, frame_width, hfov_deg=60.0):
    """
    Figures out what angle we need to turn the camera.
    Takes the x position in the image and converts it to degrees.
    You can tweak hfov_deg if your camera has a different field of view.
    """
    dx = center_x - (frame_width / 2.0)
    angle = (dx / (frame_width / 2.0)) * (hfov_deg / 2.0)
    return angle
