import os
import imageio
import glob

images = []

image_file_names = glob.glob('model_convergence_round_*')
image_file_names.sort(key=os.path.getmtime)

for filename in image_file_names:
    images.append(imageio.imread(filename))

#imageio.mimsave('movie.gif', images,  duration=0.2)

writer = imageio.get_writer('results_0.33.mp4', fps=20)

for im in image_file_names:
    writer.append_data(imageio.imread(im))
writer.close()