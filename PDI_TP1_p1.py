import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

# Lectura de información de la imagen

img2 = Image.open('Imagen_con_detalles_escondidos.tif')

exifdata = img2.getexif()
for tag_id in exifdata:
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)
    print(f"{tag:25}: {data}")

# Lectura con cv2

img = cv2.imread('Imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE)  

type(img)
img.dtype                           # uint8
img.shape                           # (256, 256)

w,h = img.shape

img.min()                           # 0
img.max()                           # 228
pix_vals = np.unique(img)           # 0,1,2,3,4,5,6,7,8,9,10,11,226,227,228
N_pix_vals = len(np.unique(img))    # 15 valores distintos

plt.imshow(img, cmap='gray')
plt.show()



# Funcion que toma como parámetro el tamaño del kernel y la imagen a ecualizar

def kernel_histogram_equalization(img, M, N):

    size_exp_M = M // 2 # entero de la división
    size_exp_N = N // 2
    img_expanded = cv2.copyMakeBorder(img, size_exp_M, size_exp_M, size_exp_N, size_exp_N, borderType=cv2.BORDER_REPLICATE) # expandimos img según los parametros de entrada

    h, w = img.shape                                            # dimensiones de la img

    result_img = np.zeros_like(img)                             # clonamos img de entrada
    
    for i in range(size_exp_M, h + size_exp_M):
        for j in range(size_exp_N, w + size_exp_N):             # iteración sobre posiciones de píxeles de la img original

            kernel = img_expanded[i - size_exp_M:i + size_exp_M + 1, j - size_exp_N:j + size_exp_N + 1] # extraemos el kernel de la imagen expandida
            
            kernel_flatten = kernel.flatten()                   # aplanamos la matriz de entrada a 1D
            kernel_equalized = cv2.equalizeHist(kernel_flatten) # aplicamos la función equalizadora sobre el kernel extraído
            
            result_img[i - size_exp_M, j - size_exp_N] = kernel_equalized[kernel.size // 2] # tomamos solo el valor del pixel central del kernel para impactarlo en la img resultante
    
    plt.imshow(result_img, cmap='gray')
    plt.show()
    return

kernel_histogram_equalization(img,75,75)