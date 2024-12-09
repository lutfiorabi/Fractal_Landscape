import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

'''
The diamond square algorithm.
iterates through the grid through a diamond step followed by a square step. 
With each iteration the scale is divided by scale_step to insure that closer datapoints have closer values. 
'''


def diamond_square(size, scale_step, iterations):
    # initial scale
    scale = 100.0

    # Calculate the midpoint of each square and displace it by the current scale
    def diamond_step(arr, step_size, scale):
        half_step = step_size // 2

        for x in range(half_step, size, step_size):
            for y in range(half_step, size, step_size):
                avg = (arr[x - half_step, y - half_step] +
                       arr[x - half_step, y + half_step] +
                       arr[x + half_step, y - half_step] +
                       arr[x + half_step, y + half_step]) / 4.0
                arr[x, y] = avg + (np.random.rand() - 0.5) * scale

    # Calculate the midpoint of each edge and displace it by the current scale
    def square_step(arr, step_size, scale):
        half_step = step_size // 2

        for x in range(0, size, half_step):
            for y in range((x + half_step) % step_size, size, step_size):
                avg = (arr[(x - half_step) % size, y] +
                       arr[(x + half_step) % size, y] +
                       arr[x, (y - half_step) % size] +
                       arr[x, (y + half_step) % size]) / 4.0
                arr[x, y] = avg + (np.random.rand() - 0.5) * scale

    # initialise the corners to random values
    arr = np.zeros((size, size))
    arr[0, 0] = (np.random.rand() - 0.5) * scale
    arr[0, -1] = (np.random.rand() - 0.5) * scale
    arr[-1, 0] = (np.random.rand() - 0.5) * scale
    arr[-1, -1] = (np.random.rand() - 0.5) * scale

    show_results(arr)

    step_size = size - 1

    for i in tqdm(range(iterations)):
        diamond_step(arr, step_size, scale)
        square_step(arr, step_size, scale)

        step_size //= 2
        scale /= scale_step
        show_results(arr, i + 1)

    return arr


def show_results(terrain, i=0, last=False, colormap='terrain'):
    x, y = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))

    def plot_3D(ax, terrain, angle):
        ax.plot_surface(x, y, terrain, cmap=colormap)
        ax.view_init(*angle)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if not last:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].imshow(terrain.T, cmap=colormap)
        axs[0].set_title("Top-down View")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        fig.delaxes(axs[1])
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        plot_3D(ax2, terrain, (30, -45))
        ax2.set_title('3D Side View')

        fig.suptitle("Initial terrain" if i == 0 else f"Terrain after {i} iterations", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{directory}/iteration_{i}.png")
    else:
        fig = plt.figure(figsize=(10, 10))
        angles = [(30, -45), (30, 45), (45, 0), (90, 0)]
        for j, angle in enumerate(angles, 1):
            ax = fig.add_subplot(2, 2, j, projection='3d')
            plot_3D(ax, terrain, angle)

        fig.suptitle("Final Fractal Landscape", fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.savefig(f"{directory}/final_{i}.png")


if __name__ == '__main__':

    directory = "results"

    # Parameters
    print("Enter the number of iterations: ", end="")
    while True:
        ITERATION = int(input())
        if ITERATION <= 0:
            print("ITERATION must be greater than zero.\nEnter the number of iterations: ", end="")
            continue
        break


    SIZE = 2 ** ITERATION + 1  # 2^n+1 x 2^n+1 grid
    print(f"The Grid size will be {SIZE}x{SIZE}")
    SCALE_STEP = 2.0  # How much is the scale divided by in each iteration?

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    terrain = diamond_square(SIZE, SCALE_STEP, ITERATION)

    show_results(terrain, last=True, i=0, colormap='PuRd_r')
    show_results(terrain, last=True, i=1, colormap='terrain')
