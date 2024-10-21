# TUIA_PDI_TP1

Requisitos
Antes de ejecutar los scripts, asegúrate de tener instaladas las siguientes bibliotecas de Python:

OpenCV (cv2)
NumPy
Matplotlib
Pillow (para extraer los metadatos de imágenes TIFF)
Puedes instalar estas dependencias ejecutando:

pip install opencv-python numpy matplotlib pillow

Instrucciones para correr los scripts
1. Ecualización Local del Histograma
   
Ejecución:

python PDI_TP1_p1.py
El script aplicará la ecualización local y mostrará la imagen procesada en una ventana gráfica.

Parámetros de entrada:
Imagen: El archivo Imagen_con_detalles_escondidos.tif.
Tamaño de la ventana: Puedes modificar los parámetros de la función kernel_histogram_equalization(img, M, N) en el código para cambiar el tamaño de la ventana (M x N píxeles).


2. Corrección Automática de Exámenes

Ejecución:

Ejecuta el siguiente comando en tu terminal:

python PDI_TP1_p2.py
El script:

Mostrará en la consola los resultados de la corrección de cada examen (respuestas correctas/incorrectas).
Validará los campos del encabezado (Nombre, Fecha, Clase).
Guardará una imagen resultados_examenes.png que contendrá los nombres de los alumnos aprobados y desaprobados, diferenciados visualmente con bordes verdes (aprobados) y rojos (desaprobados).
Parámetros de entrada:
Imágenes de exámenes: Archivos examen_1.png, examen_2.png, examen_3.png, examen_4.png, examen_5.png.
