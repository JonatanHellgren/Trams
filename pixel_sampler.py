import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class PixelSampler:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        # print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def sample_pixels(directory, folder, image):
    im = Image.open(f"{directory}/{folder}/{image}")
    fig, ax = plt.subplots()
    ax.imshow(im)
    line, = ax.plot([0], [0])  # empty line
    linebuilder = PixelSampler(line)
    plt.show()

    rgb = []
    xs, ys = np.round(linebuilder.xs[1:]), np.round(linebuilder.ys[1:])
    for x, y in zip(xs, ys):
        rgb.append(im.getpixel((int(x), int(y))))
    write_output(folder, rgb)

def write_output(folder, rgb):
    with open("sign_colors.csv", "a+") as f:
        for line in rgb:
            l = f"{folder},{line[0]},{line[1]},{line[2]}\n"
            f.writelines(l)


def show_images(directory, folder):
    images = os.listdir(f"{directory}/{folder}")
    for image in images:
        sample_pixels(directory, folder, image)
        

if __name__ == "__main__":
    with open("sign_colors.csv", "w") as f:
        f.writelines("sign,R,G,B\n")

    for folder in range(1,12):
        directory = f"./model2_easy"
        show_images(directory, folder)


