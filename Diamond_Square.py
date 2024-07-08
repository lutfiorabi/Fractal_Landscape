import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
ITERATION = int(input())  # Amount of Iterations (grid size varies depending on the iterations count)
SIZE = 2 ** ITERATION + 1  # 2^n+1 x 2^n+1 grid
print(f"The Grid size will be {SIZE}x{SIZE}")
SCALE_STEP = 2.0  # How much is the scale divided by in each iteration?

directory = "results" # the directory in which the results are saved

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
    arr[0, 0] = np.random.rand() - 0.5
    arr[0,-1] = np.random.rand() - 0.5
    arr[-1, 0] = np.random.rand() - 0.5
    arr[-1,-1] = np.random.rand() - 0.5

    show_results(arr)

    step_size = size - 1

    for i in range(iterations):

        diamond_step(arr, step_size, scale)
        square_step(arr, step_size, scale)

        step_size //= 2
        scale /= scale_step
        show_results(arr, i+1)

    return arr


'''
terrain: the generated terrain to show
i: the current iteration.
last: if last = True, then print the final terrain picture
colormap: input cmap used with plt
'''
def show_results(terrain, i=0, last=False, colormap='terrain'):
    if last == False:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # Top-down view
        axs[0].imshow(terrain.T, cmap=colormap)
        axs[0].set_title(f'Top-down View')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Bird-eye view
        fig.delaxes(axs[1])
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        X, Z = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
        ax2.plot_surface(X, Z, terrain, cmap=colormap)
        ax2.set_title(f'3D Side View')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('Y')
        ax2.set_xticks([])
        ax2.set_yticks([])

        if i == 0:
            fig.suptitle(f'Initial terrain', fontsize=20)
        else:
            fig.suptitle(f'Terrain after {i} iterations', fontsize=20)

        plt.tight_layout()
        plt.savefig(f"{directory}/iteration_{i}.png")

    else:
        fig = plt.figure(figsize=(10, 10))

        # Perspective 1
        ax1 = fig.add_subplot(221, projection='3d')
        x = np.arange(terrain.shape[0])
        y = np.arange(terrain.shape[1])
        x, y = np.meshgrid(x, y)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.view_init(30, -45)

        ax1.plot_surface(x, y, terrain, cmap=colormap)

        # Perspective 2
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.plot_surface(x, y, terrain, cmap=colormap)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.view_init(30, 45)

        # Perspective 3
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.plot_surface(x, y, terrain, cmap=colormap)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.view_init(45, 0)

        # Top-Down View (Perspective 4)
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.plot_surface(x, y, terrain, cmap=colormap)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_zticks([])
        ax4.view_init(90, 0)

        fig.suptitle('Final Fractal Landscape', fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.savefig(f"{directory}/final_{i}.png")


if __name__ == '__main__':
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    terrain = diamond_square(SIZE, SCALE_STEP, ITERATION)

    show_results(terrain, last=True, i=0, colormap='PuRd_r')
    show_results(terrain, last=True, i=1, colormap='terrain')