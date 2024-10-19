import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    # Análisis

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
    N_pix_vals = len(np.unique(lista_imgs[i]))"""

"""plt.subplot(121), plt.imshow(img1, cmap = 'gray'), plt.title('Examen1'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img2, cmap = 'gray'), plt.title('Examen2'), plt.xticks([]), plt.yticks([])
plt.show()"""

######################################################################################################

def umbralizar_imagen(img, thresh=128, maxval=255):
    if len(img.shape) == 3:  # Si la imagen está en color, conviértela a escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, img_umbralizada = cv2.threshold(img, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)
    return img, img_umbralizada

# Función para obtener y dibujar contornos en la imagen
def obtener_y_dibujar_contornos(img_umbralizada, grosor=1):

    contornos,_ = cv2.findContours(img_umbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    return contornos_ordenados

# Función para recortar la región de una pregunta dada su índice
def recortar_pregunta_por_indice(index_pregunta, contornos_ordenados, img_umbralizada, dicc_indices):
    
    contorno_pregunta = contornos_ordenados[dicc_indices[index_pregunta]]
    x, y, w, h = cv2.boundingRect(contorno_pregunta)
    pregunta = img_umbralizada[y:y+h, x:x+w]  # Recortar la región de interés

    return pregunta

# Función para detectar la línea horizontal y recortar la región de respuesta
def detectar_linea_y_recortar_respuesta(pregunta):
    
    contornos, _ = cv2.findContours(pregunta, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    contorno_linea = contornos_ordenados[1]  # El segundo contorno es la línea horizontal
    x, y, w, h = cv2.boundingRect(contorno_linea)
    
    respuesta = pregunta[0:y, x:x+w]  # Recortamos desde la línea hacia arriba

    respuesta =  respuesta[::-1,] # Rotacion 180º

    img_zeros = respuesta==0

    img_row_zeros = img_zeros.any(axis=1)

    img_row_zeros_idxs = np.argwhere(np.logical_not(respuesta.all(axis=1))) # Tengo los indices de los renglones

    if img_row_zeros_idxs.size == 0:  # Condición si es vacío no hay respuesta
        return None
    
    if img_row_zeros_idxs[0] > 10: # respuesta vacía con texto arriba
        return None

    start_end = np.diff(img_row_zeros) # inicio y final de los textos

    renglones_indxs = np.argwhere(start_end) # indices de los mismos

    start_idx = (renglones_indxs[0]).item()

    end_idx = (renglones_indxs[1]).item()

    respuesta = respuesta[start_idx:end_idx+1, :] # cortamos el sector respuesta

    respuesta =  respuesta[::-1,] # volvemos a rotarla
    
    return respuesta

def detectar_letra(recorte_respuesta):
    # Conectar componentes
    connectivity = 8  # Conexión de 8 vecinos
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(recorte_respuesta, connectivity, cv2.CV_32S)

    # Revisar si hay más de una componente conectada (excluyendo el fondo)
    if num_labels <= 1:  # Solo hay fondo
        return None # No se detectó ninguna letra

    # Si hay más de una componente conectada, indicaría más de una letra o ruido
    if num_labels > 2:  # Excluyendo el fondo
        return None  # Hay más de una letra o ruido

    # Obtener el rectángulo delimitador de la única letra
    x, y, w, h = stats[0][:4]  # stats[1] porque stats[0] es el fondo

    # Recortar la imagen de la letra desde la imagen binaria original
    letra_recortada = recorte_respuesta[y:y+h, x:x+w]  # Aquí se realiza el recorte correcto

    return letra_recortada  # Devolvemos la letra recortada


def identificador_letra(letra_recortada):
    
    img_expand = cv2.copyMakeBorder(letra_recortada, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255) # agregamos bordes

    img_inv = img_expand==0 # invertimos para que quede fondo negro

    inv_uint8 = img_inv.astype(np.uint8) # conversión para que no quede bool

    contours,_ = cv2.findContours (inv_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # buscamos contornos

    if len(contours) == 1:
        return "C"
        
    if len(contours) == 3:
        return "B"

    if len(contours) == 2:

        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])              # definimos un filtro horizontal para detectar líneas

        filtro_aplicado = cv2.filter2D(letra_recortada, cv2.CV_64F, kernel)     # Aplicar el filtro a la imagen de la letra

        magnitud_filtro = np.abs(filtro_aplicado)                               # Obtener la magnitud del filtro

 
        umbral = magnitud_filtro.max() * 0.8                                    # Umbralizar la imagen filtrada para obtener una imagen binaria
        imagen_binaria = magnitud_filtro >= umbral

        
        lineas_horizontales = np.any(imagen_binaria, axis=1)                    # Contar las filas con al menos un valor True
        cantidad_lineas = np.sum(lineas_horizontales)

        if cantidad_lineas == 1:
            return "A"

        else: 
            return "D"

# Función para recortar la parte superior de la pregunta
def recortar_parte_superior(contornos_ordenados, img_umbralizada, dicc_indices):

    idx_pregunta_sup_izq = 1

    idx_pregunta_sup_der = 6    # Ya sabemos que las pregutas superiores están en esos indices

    contorno_izq = contornos_ordenados[dicc_indices[idx_pregunta_sup_izq]]
    contorno_der = contornos_ordenados[dicc_indices[idx_pregunta_sup_der]]
    
    _, y_inf, _, _ = cv2.boundingRect(contorno_izq)
    _, y_sup, _, _ = cv2.boundingRect(contorno_der)
    
    borde_inferior = min(y_inf, y_sup)  # buscamos la mínima altura para cortar la parte superior
    
    # Recortamos la parte superior
    box_superior = img_umbralizada[6:borde_inferior-6, 6:564]
    
    return box_superior

def recortar_encabezado(idx: int, examen):

    _, img_umbralizada = umbralizar_imagen(examen,thresh=140) # Subimos el thresh porque las barras inclinadas se desconectaban con 128 y las respuestas en 140 se distorcionaban

    contornos_ordenados = obtener_y_dibujar_contornos(img_umbralizada)

    dicc_indices = {1: 6, 2: 5, 3: 4, 4: 3, 5: 12, 6: 7, 7: 11, 8: 8, 9: 9, 10: 10}

    box_superior_umb = recortar_parte_superior(contornos_ordenados, img_umbralizada, dicc_indices)

    contornos_box_sup = obtener_y_dibujar_contornos(box_superior_umb)

    contorno_box = contornos_box_sup[idx] # Iterar de 1 a 3

    x, y, w, _ = cv2.boundingRect(contorno_box)

    dato = box_superior_umb[0:y, x:x+w]  # Recortar la región de interés

    return dato



def contar_caracteres(img_palabra):

    img_zeros = img_palabra == 0

    img_row_zeros = img_zeros.any(axis=0)

    start_end = np.diff(img_row_zeros)

    renglones_indxs = np.argwhere(start_end)

    espacios = 0

    if len(renglones_indxs) == 0:
        
        return 0,espacios

    cantidad_letras = len(renglones_indxs)/2

    if len(renglones_indxs) == 2:
        
        return cantidad_letras,espacios

    for i in range(1,len(renglones_indxs)-1,2):

        if (renglones_indxs[i+1].item() - renglones_indxs[i].item()) > 5:

            espacios+=1

    return cantidad_letras,espacios

def correccion_examen(examen, verbose=True):
    # Diccionario para mapear las preguntas con los índices de contornos
    dicc_indices = {1: 6, 2: 5, 3: 4, 4: 3, 5: 12, 6: 7, 7: 11, 8: 8, 9: 9, 10: 10}

    img, img_umbralizada = umbralizar_imagen(examen)

    contornos_ordenados = obtener_y_dibujar_contornos(img_umbralizada, grosor=1)

    respuestas_correctas = {1: "C", 2: "B", 3: "A", 4: "D", 5: "B", 6: "B", 7: "A", 8: "B", 9: "D", 10: "D"}

    cantidad_correctas = 0

    for i in range(1, 11):  # Iteramos por todas las preguntas
        
        pregunta = recortar_pregunta_por_indice(i, contornos_ordenados, img_umbralizada, dicc_indices)
        
        box_respuesta = detectar_linea_y_recortar_respuesta(pregunta)

        if box_respuesta is None:
            if verbose:
                print(f'Pregunta {i}: MAL')
            continue
        
        respuesta = detectar_letra(box_respuesta)

        if respuesta is None:
            if verbose:
                print(f'Pregunta {i}: MAL')
            continue

        letra_identificada = identificador_letra(respuesta)

        if letra_identificada == respuestas_correctas[i]:
            if verbose:
                print(f'Pregunta {i}: OK')
            cantidad_correctas += 1
        else:
            if verbose:
                print(f'Pregunta {i}: MAL')
            
    return cantidad_correctas


def correccion_encabezado(examen):

    name = recortar_encabezado(1,examen)

    date = recortar_encabezado(3,examen)

    clas = recortar_encabezado(2,examen)

    name_car,name_esp = contar_caracteres(name)

    date_car,_ = contar_caracteres(date)

    clas_car,_ = contar_caracteres(clas)

    if name_esp == 1 and name_car < 26:
        print('Name: OK')

    else:
        print('Name: MAL')
        
    if date_car == 8:
        print('Date: OK')

    else:
        print('Date: MAL')
        
    if clas_car == 1:
        print('Class: OK')

    else:
        print('Class: MAL')
        
    return


####################################################################################################################################################

def save_img(exam_list):

    dict_exam = {} 

    for idx, exam in enumerate(exam_list, start=1):

        name = recortar_encabezado(1, exam)

        cantidad_correctas = correccion_examen(exam, verbose=False)

        if cantidad_correctas >= 6:
            resultado = "Aprobado"
        else:
            resultado = "Desaprobado"

        dict_exam[idx] = (name, resultado)  


    altura_crop = dict_exam[1][0].shape[0]  # altura de cada crop, tomamos el primero
    ancho_crop = dict_exam[1][0].shape[1]   # ancho del crop del nombre
    margen_vertical = 2                    # espacio entre cada crop
    
    altura_imagen = len(exam_list) * (altura_crop + margen_vertical)
    ancho_imagen = ancho_crop
    
    imagen_final = np.ones((altura_imagen, ancho_imagen, 3), dtype=np.uint8) * 255  # creamos una imagen final en blanco con los parametros antes creados
    
    y_coor = 0
    
    for idx, (crop, resultado) in dict_exam.items():

        crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) # convertimos a una imagen RGB
       
        imagen_final[y_coor:y_coor + altura_crop, :ancho_crop] = crop_color # pegamos el crop en la imagen final

        if resultado == "Aprobado":
            color_borde = (0, 255, 0)  # Verde
        else:
            color_borde = (0, 0, 255)  # Rojo

        cv2.rectangle(imagen_final, (0, y_coor), (ancho_crop, y_coor + altura_crop), color_borde, 2) # rectangulo sobre el crop

        y_coor += altura_crop + margen_vertical  # Incrementar vertical para el siguiente nombre

    cv2.imwrite('resultados_examenes.png', imagen_final)

    return

####################################################################################################################################################

def main():
    # Cargamos las imágenes de los exámenes
    exam1 = cv2.imread('examen_1.png', cv2.IMREAD_GRAYSCALE)
    exam2 = cv2.imread('examen_2.png', cv2.IMREAD_GRAYSCALE)
    exam3 = cv2.imread('examen_3.png', cv2.IMREAD_GRAYSCALE)
    exam4 = cv2.imread('examen_4.png', cv2.IMREAD_GRAYSCALE)
    exam5 = cv2.imread('examen_5.png', cv2.IMREAD_GRAYSCALE)
    
    exam_list = [exam1, exam2, exam3, exam4, exam5]

    for idx, exam in enumerate(exam_list, start=1):
        print(f'----- Corrigiendo Examen {idx} -----')
        
        # Corrección del encabezado
        print(f'Corrección del encabezado del Examen {idx}:')
        correccion_encabezado(exam)
        
        # Corrección del cuerpo del examen
        print(f'\nCorrección del cuerpo del Examen {idx}:')
        correccion_examen(exam)
        
        print(f'----- Fin de la corrección del Examen {idx} -----\n')
    
    save_img(exam_list)

if __name__ == "__main__":
    main()
