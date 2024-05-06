import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def save_masks_to_file(masks, save_dir, save_name, format='mat'):
    file_path = os.path.join(save_dir, f'{save_name}_masks.{format}')
    masks_np = masks.int().numpy()

    if format == 'txt':
        with open(file_path, 'w') as f:
            for i in range(masks_np.shape[0]):
                f.write(f'Mask {i+1} of {masks_np.shape[0]}, Dimensions: {masks_np[i].shape}\n')
                for x in range(masks_np[i].shape[0]):
                    for y in range(masks_np[i].shape[1]):
                        f.write(f'({x}, {y}): {masks_np[i, x, y]}\n')
                f.write('-' * 40 + '\n')

    elif format == 'mat':
        scipy.io.savemat(file_path, {'masks': masks_np})

    elif format == 'png':
        for i in range(masks_np.shape[0]):
            file_path_i = os.path.join(save_dir, f'{save_name}_mask_{i}.{format}')
            plt.imsave(file_path_i, masks_np[i], cmap='gray')
            if 'google.colab' in str(get_ipython()):
                from google.colab import files
                files.download(file_path_i)

    elif format == 'npy':
        np.save(file_path, masks_np)

    else:
        raise ValueError(f'Unsupported format: {format}')

    # If running in Google Colab, download the file
    if 'google.colab' in str(get_ipython()) and format != 'png':
        from google.colab import files
        files.download(file_path)
    else:
        print(f'Masks saved to {file_path}')
