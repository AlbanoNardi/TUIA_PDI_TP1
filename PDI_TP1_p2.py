import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

# Cargamos imgs

img1 = cv2.imread('examen_1.png',cv2.IMREAD_GRAYSCALE)  
img2 = cv2.imread('examen_2.png',cv2.IMREAD_GRAYSCALE)  
img3 = cv2.imread('examen_3.png',cv2.IMREAD_GRAYSCALE)  
img4 = cv2.imread('examen_4.png',cv2.IMREAD_GRAYSCALE)  
img5 = cv2.imread('examen_5.png',cv2.IMREAD_GRAYSCALE)  

lista_imgs = [img1,img2,img3,img4,img5]

for i in range(0,5):
    type(lista_imgs[i])

for i in range(0,5):
    lista_imgs[i].dtype

for i in range(0,5):
    lista_imgs[i].shape 

# Los tamaños y tipos de imgs son iguales 686x570

for i in range(0,5):
    lista_imgs[i].min()
    lista_imgs[i].max()                             # rangos de 0 a 255

for i in range(0,5):
    pix_vals = np.unique(lista_imgs[i])
    print(pix_vals)
    
for i in range(0,5):
    N_pix_vals = len(np.unique(lista_imgs[i]))
    print(N_pix_vals)                               # 236 y 237 valores distintos por img


for i in range(0,5):
    N_pix_vals = len(np.unique(lista_imgs[i]))

plt.subplot(121), plt.imshow(img1, cmap = 'gray'), plt.title('Examen1'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img2, cmap = 'gray'), plt.title('Examen2'), plt.xticks([]), plt.yticks([])
plt.show()

######################################################################################################

def umbralizar_imagen(img, thresh=128, maxval=255):
    if len(img.shape) == 3:  # Si la imagen está en color, conviértela a escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, img_umbralizada = cv2.threshold(img, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)
    return img, img_umbralizada

# Función para obtener y dibujar contornos en la imagen
def obtener_y_dibujar_contornos(img_umbralizada, img, grosor=1):
    contornos,_ = cv2.findContours(img_umbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    for i, contorno in enumerate(contornos_ordenados[:13]):  # Limitar a 13 contornos para simplificar
        color = (255, 0, 0) if i == 0 else (0, 255, 0)  # Usamos colores diferentes
        cv2.drawContours(img, contornos_ordenados, contourIdx=i, color=color, thickness=grosor)
    
    plt.imshow(img)
    plt.show()
    
    return contornos_ordenados

# Función para recortar la región de una pregunta dada su índice
def recortar_pregunta_por_indice(index_pregunta, contornos_ordenados, img_umbralizada, dicc_indices):
    if index_pregunta not in dicc_indices:
        return None
    
    contorno_pregunta = contornos_ordenados[dicc_indices[index_pregunta]]
    x, y, w, h = cv2.boundingRect(contorno_pregunta)
    roi_pregunta = img_umbralizada[y:y+h, x:x+w]  # Recortar la región de interés
    
    plt.imshow(roi_pregunta, cmap='gray')
    plt.title(f'Pregunta número {index_pregunta}')
    plt.show()
    
    return roi_pregunta

# Función para detectar la línea horizontal y recortar la región de respuesta
def detectar_linea_y_recortar_respuesta(roi_pregunta):
    contornos, _ = cv2.findContours(roi_pregunta, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    contorno_linea = contornos_ordenados[1]  # El segundo contorno es la línea horizontal
    x, y, w, h = cv2.boundingRect(contorno_linea)
    
    # Recortamos desde la línea hacia arriba
    roi_respuesta = roi_pregunta[0:y, x:x+w]

    roi_respuesta =  roi_respuesta[::-1,]

    plt.imshow(roi_respuesta, cmap='gray')
    plt.show()
    
    return roi_respuesta

# Función para recortar la parte superior de la pregunta
def recortar_parte_superior(index_pregunta, index_pregunta_superior, contornos_ordenados, img_color, dicc_indices):
    contorno_inf = contornos_ordenados[dicc_indices[index_pregunta]]
    contorno_sup = contornos_ordenados[dicc_indices[index_pregunta_superior]]
    
    _, y_inf, _, _ = cv2.boundingRect(contorno_inf)
    _, y_sup, _, _ = cv2.boundingRect(contorno_sup)
    
    borde_inferior = min(y_inf, y_sup)
    
    # Recortamos la parte superior
    roi_superior = img_color[6:borde_inferior-6, 6:564]
    
    plt.imshow(roi_superior)
    plt.show()
    
    return roi_superior



-------------------------------------------------------------------------------------------
"""#aca ver como cortar la parte que contiene la letrad



"""
-------------------------------------------------------------------------------------------

def detectar_letra(recorte_respuesta):
    # Conectar componentes
    connectivity = 8  # Conexión de 8 vecinos
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(recorte_respuesta, connectivity, cv2.CV_32S)

    # Revisar si hay más de una componente conectada (excluyendo el fondo)
    if num_labels <= 1:  # Solo hay fondo
        return 0  # No se detectó ninguna letra

    # Si hay más de una componente conectada, indicaría más de una letra o ruido
    if num_labels > 2:  # Excluyendo el fondo
        return 0  # Hay más de una letra o ruido

    # Obtener el rectángulo delimitador de la única letra
    x, y, w, h = stats[0][:4]  # stats[1] porque stats[0] es el fondo

    # Recortar la imagen de la letra desde la imagen binaria original
    letra_recortada = recorte_respuesta[y:y+h, x:x+w]  # Aquí se realiza el recorte correcto

    return letra_recortada  # Devolvemos la letra recortada


def identificador_letra(letra_recortada):
    # Definir un filtro horizontal para detectar líneas
    kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    
    # Aplicar el filtro a la imagen de la letra
    filtro_aplicado = cv2.filter2D(letra_recortada, cv2.CV_64F, kernel)

    # Obtener la magnitud del filtro
    magnitud_filtro = np.abs(filtro_aplicado)

    # Umbralizar la imagen filtrada para obtener una imagen binaria
    umbral = magnitud_filtro.max() * 0.8  # Ajusta este valor si es necesario
    imagen_binaria = magnitud_filtro >= umbral

    # Contar las filas con al menos un valor True
    lineas_horizontales = np.any(imagen_binaria, axis=1)
    cantidad_lineas = np.sum(lineas_horizontales)

    # Visualizar la imagen binaria
    plt.imshow(imagen_binaria, cmap='gray')
    plt.title(f'Líneas detectadas: {cantidad_lineas}')
    plt.axis('off')
    plt.show()

    # Clasificar la letra basada en la cantidad de líneas horizontales
    if cantidad_lineas == 1:
        return "A"
    elif cantidad_lineas == 3:
        return "B"
    elif cantidad_lineas == 2:
        return "D"
    else:
        return "C"


-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------


def correccion_examen(examen):
    # Diccionario para mapear las preguntas con los índices de contornos
    dicc_indices = {1: 6, 2: 5, 3: 4, 4: 3, 5: 12, 6: 7, 7: 11, 8: 8, 9: 9, 10: 10}

    img, img_umbralizada = umbralizar_imagen(examen)
    contornos_ordenados = obtener_y_dibujar_contornos(img_umbralizada,img,grosor=1)
    
    respuestas_correctas = {1: "C", 2: "B", 3: "A", 4: "D", 5: "B", 6: "B", 7: "A", 8: "B", 9: "D", 10: "D"}

    for i in range(1, 11):  # Iteramos por todas las preguntas
        # Paso 3: Recortar una pregunta específica
        pregunta = recortar_pregunta_por_indice(i, contornos_ordenados, img_umbralizada, dicc_indices)

        if pregunta is None:
            print(f"Pregunta {i}: No se pudo recortar.")
            continue

        respuesta = detectar_linea_y_recortar_respuesta(pregunta)

        letra_detectada = detectar_letra(respuesta)

        if isinstance(letra_detectada, int) and letra_detectada == 0:
            print(f'Pregunta {i}: MAL (No se detectó ninguna letra)')
            continue

        letra_identificada = identificador_letra(letra_detectada)

        if letra_identificada == respuestas_correctas[i]:
            print(f'Pregunta {i}: OK')
        else:
            print(f'Pregunta {i}: MAL')

    return





img_gris, img_umbralizada = umbralizar_imagen(img5,thresh=128)

contornos_ordenados = obtener_y_dibujar_contornos(img_umbralizada, img5)

dicc_indices = {1: 6, 2: 5, 3: 4, 4: 3, 5: 12, 6: 7, 7: 11, 8: 8, 9: 9, 10: 10}

pregunta = recortar_pregunta_por_indice(4, contornos_ordenados, img_umbralizada, dicc_indices)

respuesta = detectar_linea_y_recortar_respuesta(pregunta)
"""
respuesta.all(axis=0)               # any() devuelve TRUE si aunque sea un elemento es != 0. 
np.argwhere(np.logical_not(respuesta.all(axis=0)))  
# argwhere() devuelve los índices del vector de entrada donde el valor es TRUE.

respuesta.any(axis=1)               
"""

img_zeros = respuesta==0

img_row_zeros = img_zeros.any(axis=1)

img_row_zeros_idxs = np.argwhere(np.logical_not(respuesta.all(axis=1))) # Tengo los indices de los renglones

plt.figure(), plt.plot(img_row_zeros), plt.show()


xr = img_row_zeros*(respuesta.shape[1]-1)     # Generamos valores en el eje x (el eje que apunta hacia la derecha) --> dos posibles valores: 0 o el ancho de la imagen (cantidad de columnas de la imagen -1).
yr = np.arange(respuesta.shape[0])            # Generamos valores en el eje y (el eje que apunta hacia abajo) --> Vector que arranca en 0, aumenta de a 1 y llega hasta la cantidad de filas de la imagen [0, 1, ..., N_filas-1]
plt.figure(), plt.imshow(respuesta, cmap='gray'), plt.plot(xr, yr, c='r'), plt.title("Renglones"), plt.show(block=False)                    

x = np.diff(img_row_zeros)          
renglones_indxs = np.argwhere(x)    # Esta variable contendrá todos los inicios y finales de los renglones
len(renglones_indxs)  

np.argwhere(img_row_zeros)[0]

inicio_renglon = np.argwhere(np.diff(img_row_zeros))[0]

fin_renglon = np.argwhere(np.diff(img_row_zeros))[1]+1
