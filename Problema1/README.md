# Analizador de Regiones de Color

Esta herramienta permite analizar los rangos de color en regiones específicas de imágenes, facilitando la definición de umbrales para segmentación de objetos por color.

## Requisitos Previos

- Python 3.x
- Bibliotecas: numpy, opencv-python (cv2), matplotlib, pandas, tkinter
- Imágenes de referencia del objeto/color que se quiere analizar

## Estructura de Archivos

```
Problema1/
├── color_region_analyzer.py    # Script principal
├── Imagenes/                   # Carpeta donde deben estar las imágenes de entrada
└── test/                       # Carpeta donde se guardarán los resultados
    ├── color_statistics.csv
    ├── color_stats.txt
    ├── distribucion_rgb_3d.png
    ├── distribucion_hsv_3d.png
    └── rango_hsv_visualizado_2d.png
```

## Cómo Usar

1. **Preparación**: 
   - Coloque las imágenes que desea analizar en la carpeta `Problema1/Imagenes/`
   - Si las carpetas no existen, el programa las creará automáticamente

2. **Ejecución**:
   ```
   python color_region_analyzer.py
   ```

3. **Uso de la Interfaz**:
   - Para cada imagen, se abrirá una ventana interactiva
   - **Modo Dibujo**: Haga clic para marcar puntos que formen el contorno de la región de interés
   - **Modo Zoom**: Alterne con el botón "Activar Zoom" para navegar y acercar la imagen
   - Dibuje al menos 3 puntos para formar una región cerrada
   - Use los botones:
     - **Aceptar**: Procesa la región seleccionada y pasa a la siguiente imagen
     - **Limpiar**: Elimina los puntos seleccionados para empezar de nuevo
     - **Cancelar**: Salta la imagen actual sin procesarla
     - **Cancelar Todo**: Termina el proceso completo
   - Presione 'Esc' para volver al modo dibujo desde el modo zoom

4. **Consejos**:
   - Seleccione regiones que contengan exclusivamente el color que desea analizar
   - Para resultados más precisos, analice múltiples regiones en diferentes imágenes
   - Asegúrese de que las imágenes tengan buena iluminación y representen bien el color objetivo

## Resultados

Después de procesar todas las imágenes, se generarán los siguientes archivos en la carpeta `test/`:

1. **color_statistics.csv**: 
   - Datos crudos de todos los píxeles analizados con sus valores RGB y HSV

2. **color_stats.txt**: 
   - Estadísticas detalladas de los rangos de color en formato legible
   - Incluye código listo para usar en aplicaciones de visión por computadora

3. **Visualizaciones**:
   - **rango_hsv_visualizado_2d.png**: Representación visual 2D de los rangos de color HSV
   - **distribucion_rgb_3d.png**: Gráfico 3D de los píxeles en el espacio RGB
   - **distribucion_hsv_3d.png**: Gráfico 3D de los píxeles en el espacio HSV

## Aplicación de los Resultados

Los rangos de color obtenidos pueden utilizarse directamente en aplicaciones de visión por computadora para:

- Segmentación de objetos por color
- Filtrado de imágenes
- Detección de objetos específicos

Ejemplo de uso de los valores en código OpenCV:
```python
import cv2
import numpy as np

# Valores obtenidos del análisis (ejemplo)
lower_hsv_cv = np.array([5, 100, 100])
upper_hsv_cv = np.array([15, 255, 255])

# Cargar imagen
image = cv2.imread('imagen.jpg')

# Convertir a HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Crear máscara utilizando los valores obtenidos
mask = cv2.inRange(hsv, lower_hsv_cv, upper_hsv_cv)

# Aplicar la máscara
result = cv2.bitwise_and(image, image, mask=mask)

# Mostrar resultado
cv2.imshow('Resultado', result)
cv2.waitKey(0)
```

## Limitaciones

- El análisis depende de la calidad de las regiones seleccionadas
- Los resultados pueden variar según las condiciones de iluminación de las imágenes
- Se recomienda probar los rangos obtenidos en diferentes condiciones antes de su uso definitivo
