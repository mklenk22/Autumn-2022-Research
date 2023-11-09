import tkinter as tk
import cv2 as cv
import numpy as np


def show_crop():
    x_offset = slide5.get()
    y_offset = slide6.get()
    scaled_x = slide7.get()
    scaled_y = slide8.get()
    r_crop = 1280 - (scaled_x + x_offset)
    l_crop = x_offset
    t_crop = y_offset
    b_crop = 720 - (scaled_y + y_offset)

    string = f'r_crop: {r_crop}\nl_crop: {l_crop}\nt_crop: {t_crop}\nb_crop: {b_crop}'
    broadcast_text.set(string)


def thermal2red(frames):
    """
    Input a thermal image and it will return the frames in black and red scale.
    You will still need to use `imshow` function to show the image
    :param frames: frame from the capture of a thermal camera
    :return: the `res` colorscale of the image
    """
    hsv = cv.cvtColor(frames, cv.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 100])
    upper_white = np.array([120, 20, 255])
    frame_copy = frames.copy()
    mask = cv.inRange(hsv, lower_white, upper_white)
    frame_copy[mask > 0] = (0, 0, 255)
    res = cv.bitwise_and(frame_copy, frame_copy, mask=mask)
    return res


# cap 1 stuff, for thermal camera
cap = cv.VideoCapture(0)
x_dim = 640
y_dim = 480
cap.set(3, x_dim)
cap.set(4, y_dim)

# cap 2 stuff, for rgb camera
cap2 = cv.VideoCapture(0)
rgb_x = 1280
rgb_y = 720
cap2.set(3, rgb_x)
cap2.set(4, rgb_y)
img_x1 = 0
img_x2 = rgb_x
img_y1 = 0
img_y2 = rgb_y

BLACK = [0, 0, 0]


def cmd(*args):
    val1 = slide1.get()
    val2 = slide2.get()
    val3 = slide3.get()
    val4 = slide4.get()
    val5 = slide5.get()
    val6 = slide6.get()
    val7 = slide7.get()
    val8 = slide8.get()
    return [val1, val2, val3, val4, val5, val6, val7, val8]


def therm_dim(y_scalar: float, x_scalar: float, x_dimension: int, y_dimension: int) -> dict:
    new_dimensions = {}
    new_x = x_dimension * x_scalar
    new_y = y_dimension * y_scalar
    new_dimensions['new_x_dim'] = new_x
    new_dimensions['new_y_dim'] = new_y
    return new_dimensions


root = tk.Tk()
root.title("IMAGE SLIDERS")

var1 = tk.DoubleVar()
slide1 = tk.Scale(root, label='x1', from_=0, to=rgb_x, orient=tk.HORIZONTAL, variable=var1, command=cmd, length=300)
slide1.set(0)
slide1.pack()

var2 = tk.DoubleVar()
slide2 = tk.Scale(root, label='x2', from_=0, to=rgb_x, orient=tk.HORIZONTAL, variable=var2, command=cmd, length=300)
slide2.set(1280)
slide2.pack()

var3 = tk.DoubleVar()
slide3 = tk.Scale(root, label='y1', from_=0, to=rgb_y, orient=tk.HORIZONTAL, variable=var3, command=cmd, length=300)
slide3.set(0)
slide3.pack()

var4 = tk.DoubleVar()
slide4 = tk.Scale(root, label='y2', from_=0, to=rgb_y, orient=tk.HORIZONTAL, variable=var4, command=cmd, length=300)
slide4.set(720)
slide4.pack()

var5 = tk.DoubleVar()
slide5 = tk.Scale(root, label='x offset', from_=0, to=1280 - x_dim, orient=tk.HORIZONTAL, variable=var5, command=cmd,
                  length=300)
slide5.set(0)
slide5.pack()

var6 = tk.DoubleVar()
slide6 = tk.Scale(root, label='y offset', from_=0, to=720 - y_dim, orient=tk.HORIZONTAL, variable=var6, command=cmd,
                  length=300)
slide6.set(0)
slide6.pack()

var7 = tk.DoubleVar()
slide7 = tk.Scale(root, label='X-scale', from_=0.01, to=2, orient=tk.HORIZONTAL, variable=var7, command=cmd,
                  length=300, resolution=0.01)
slide7.set(1)
slide7.pack()

var8 = tk.DoubleVar()
slide8 = tk.Scale(root, label='Y-scale', from_=0.01, to=1.5, orient=tk.HORIZONTAL, variable=var8, command=cmd,
                  length=300, resolution=0.01)
slide8.set(1)
slide8.pack()

# button
button = tk.Button(root, text='Get Crop', command=show_crop)
button.pack()

# display info
broadcast_text = tk.StringVar()
broadcast_text.set(' \n \n \n ')
broadcast_message = tk.Label(root, textvariable=broadcast_text)
broadcast_message.pack()


loop_active = True
while loop_active:
    all_dim = cmd()
    ret1, frame = cap.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # (x1:x2, y1:y2)
    x1 = all_dim[0]
    x2 = all_dim[1]
    y1 = all_dim[2]
    y2 = all_dim[3]
    x_offset = all_dim[4]
    y_offset = all_dim[5]
    x_scale = all_dim[6]
    y_scale = all_dim[7]
    new_dimen_dict = therm_dim(y_scale, x_scale, x_dim, y_dim)
    new_x_dim = new_dimen_dict['new_x_dim']
    new_y_dim = new_dimen_dict['new_y_dim']
    new_ratio = (int(new_x_dim), int(new_y_dim))

    if x2 > x1 and y2 > y1:
        crop_img = frame2[y1:y2, x1:x2]
    else:
        crop_img = frame2[0:rgb_y, 0:rgb_x]
        slide1.set(0)
        slide2.set(rgb_x)
        slide3.set(0)
        slide4.set(rgb_y)
        y1 = 0
        y2 = rgb_y
        x1 = 0
        x2 = rgb_x

    # src, top, bottom, left, right, value
    rgb_border = cv.copyMakeBorder(crop_img, y1, rgb_y - y2, x1, rgb_x - x2, cv.BORDER_CONSTANT, value=BLACK)

    # scale the image
    scaled = cv.resize(src=frame, dsize=new_ratio, fx=x_scale, fy=y_scale, interpolation=cv.INTER_NEAREST)
    scaled_y = scaled.shape[0]
    scaled_x = scaled.shape[1]

    if (scaled_y + y_offset) > 720 or (scaled_x + x_offset) > 1280:
        slide5.set(0)
        slide6.set(0)
        slide7.set(1)
        slide8.set(1)
    else:
        # src, top, bottom, left, right, value
        constant = cv.copyMakeBorder(scaled, y_offset, 720 - scaled_y - y_offset, x_offset, 1280 - scaled_x - x_offset,
                                     cv.BORDER_CONSTANT, value=BLACK)

    # operations on image for red mask and cropping
    red_im = thermal2red(constant)
    blend = cv.addWeighted(red_im, 0.7, rgb_border, 0.5, 0)
    cv.imshow('Blended', blend)

    # update and closing operations
    root.update()
    if cv.waitKey(1) == ord('q'):
        break


    # tk.mainloop()

root.quit()
# When everything done, release the capture
cap.release()
cap2.release()
cv.destroyAllWindows()

