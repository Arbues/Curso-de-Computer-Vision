
# Transformador de Imágenes Interactivo

## Descripción
Herramienta interactiva para seleccionar una región específica de una imagen y transformarla para que se ajuste a un rectángulo vertical fijo. Ideal para extraer y normalizar elementos de imágenes como documentos, carteles o cualquier objeto en perspectiva.

## Requisitos
- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib

```bash
pip install opencv-python numpy matplotlib
```

## Estructura del Proyecto
```
Problema2/
├── TransformadorInteractivo.py
├── imagenes/                    # Coloca aquí las imágenes a procesar
└── imagenes_transformadas/      # Aquí se guardan los resultados
```

## Funcionamiento

El programa permite:
1. Seleccionar una región mediante un cuadro rojo ajustable
2. Transformar esa región para que se ajuste a un rectángulo verde vertical (400x600 píxeles)
3. Aplicar diferentes tipos de transformaciones: perspectiva, afín o similaridad

## Uso

1. Coloca las imágenes que deseas transformar en la carpeta `imagenes/`
2. Ejecuta el script:
   ```bash
   python TransformadorInteractivo.py
   ```
3. Interfaz:
   - **Panel Izquierdo**: Imagen original con cuadro rojo ajustable
     - Puntos rojos: Puntos de control para seleccionar la región
     - Punto naranja: Punto con hover (resaltado)
     - Punto amarillo: Punto seleccionado
   - **Panel Central**: Rectángulo verde vertical de destino (400x600) y resultado
   - **Panel Derecho**: Controles
     - Tipo de transformación: Perspectiva, Afín, Similaridad
     - Navegación entre imágenes
     - Aplicar transformación
     - Guardar resultado
     - Resetear puntos

## Tipos de Transformación

- **Perspectiva**: Ideal para corregir distorsión de perspectiva (4 puntos)
- **Afín**: Mantiene líneas paralelas (3 puntos)
- **Similaridad**: Rotación + Escalado + Traslación (preserva ángulos)

## Características

- Cuadro rojo ajustable para seleccionar la región deseada
- Rectángulo verde fijo de destino (vertical, 400x600 píxeles)
- Feedback visual durante la selección de puntos
- Transformación en tiempo real
- Guardado con nombre descriptivo incluyendo tipo y dimensiones
- Reset de puntos a posición inicial

## Ejemplos de Uso

1. **Extraer documentos**: Selecciona un documento en perspectiva y obtén una vista frontal normalizada
2. **Normalizar carteles**: Extrae carteles o pósters desde cualquier ángulo
3. **Capturar pantallas**: Obtén capturas de pantallas o proyecciones en perspectiva

## Notas

- La región seleccionada con el cuadro rojo se transformará para ajustarse exactamente al rectángulo verde
- El resultado siempre tendrá las dimensiones del rectángulo verde (400x600 píxeles)
- Los puntos rojos inician en una región central de la imagen
- Las imágenes transformadas se guardan con información del tipo de transformación y dimensiones


Los cambios principales son:

1. **Puntos de origen (rojos)** son los que movemos para seleccionar la región
2. **Puntos de destino (verdes)** están fijos en un rectángulo vertical (400x600)
3. La transformación ajusta la región seleccionada al rectángulo fijo
4. El resultado siempre tiene las dimensiones del rectángulo vertical
5. Se muestra un preview del rectángulo de destino antes de aplicar la transformación
6. Los colores son coherentes: rojo para origen, verde para destino
7. El nombre de archivo guardado incluye las dimensiones del resultado