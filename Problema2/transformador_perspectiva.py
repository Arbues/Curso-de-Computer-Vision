import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import cv2
from pathlib import Path
import sys

class TransformadorInteractivo:
    def __init__(self):
        self.img = None
        self.src_points = []
        self.dst_points = []
        self.selected_point = None
        self.hover_point = None
        self.hover_tolerance = 30
        self.current_transform = 'perspective'
        self.transform_applied = False
        
        # Dimensiones del rectángulo de destino (vertical)
        self.output_width = 400
        self.output_height = 600
        
        # Configurar directorios
        self.img_dir = Path('Problema2/imagenes')
        self.output_dir = Path('Problema2/imagenes_transformadas')
        self.output_dir.mkdir(exist_ok=True)
        
        # Listar imágenes disponibles
        self.images = list(self.img_dir.glob('*.*'))
        self.current_image_idx = 0
        
        if not self.images:
            print("No se encontraron imágenes en la carpeta 'imagenes/'")
            sys.exit(1)
        
        # Crear la figura principal con más espacio para controles
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[4, 4, 1])
        
        self.ax_original = self.fig.add_subplot(gs[0])
        self.ax_transformed = self.fig.add_subplot(gs[1])
        
        # Panel de control en el espacio de la grilla
        self.ax_controls = self.fig.add_subplot(gs[2])
        self.ax_controls.set_visible(False)  # Ocultamos este axis
        
        # Panel de control como subplots
        self.setup_controls()
        
        # Conectar eventos del mouse
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Cargar primera imagen
        self.load_current_image()
        
    def setup_controls(self):
        """Configurar el panel de control"""
        # Ajustar la posición de los controles para que no se superpongan
        
        # Título principal
        plt.figtext(0.78, 0.95, 'Controles', ha='center', fontsize=14, weight='bold')
        
        # Radio buttons para tipo de transformación
        ax_radio = plt.axes([0.75, 0.75, 0.2, 0.15])
        self.radio_transform = RadioButtons(ax_radio, ('Perspectiva', 'Afín', 'Similaridad'))
        self.radio_transform.on_clicked(self.update_transform_type)
        ax_radio.set_title('Tipo de Transformación', fontsize=10)
        
        # Botón de Reset
        self.ax_reset = plt.axes([0.75, 0.6, 0.2, 0.05])
        self.btn_reset = Button(self.ax_reset, 'Reset Puntos')
        self.btn_reset.on_clicked(self.reset_points)
        
        # Botón de aplicar transformación
        self.ax_apply = plt.axes([0.75, 0.5, 0.2, 0.05])
        self.btn_apply = Button(self.ax_apply, 'Aplicar')
        self.btn_apply.on_clicked(self.apply_transform)
        
        # Botón de guardar
        self.ax_save = plt.axes([0.75, 0.4, 0.2, 0.05])
        self.btn_save = Button(self.ax_save, 'Guardar')
        self.btn_save.on_clicked(self.save_image)
        
        # Titulo para navegación
        plt.figtext(0.85, 0.3, 'Navegación', ha='center', fontsize=10)
        
        # Botones de navegación
        self.ax_prev = plt.axes([0.75, 0.2, 0.095, 0.05])
        self.btn_prev = Button(self.ax_prev, '← Prev')
        self.btn_prev.on_clicked(self.prev_image)
        
        self.ax_next = plt.axes([0.855, 0.2, 0.095, 0.05])
        self.btn_next = Button(self.ax_next, 'Next →')
        self.btn_next.on_clicked(self.next_image)
        
    def load_current_image(self):
        """Carga la imagen actual y reinicia los puntos"""
        try:
            self.img_path = self.images[self.current_image_idx]
            self.img = cv2.imread(str(self.img_path))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            # Reiniciar puntos al cargar nueva imagen
            self.reset_points()
            self.update_display()
        except Exception as e:
            print(f"Error al cargar imagen: {e}")
            
    def reset_points(self, event=None):
        """Reinicia los puntos a sus posiciones por defecto"""
        if self.img is not None:
            h, w = self.img.shape[:2]
            
            # Puntos de origen (rojo) - inicialmente en el centro de la imagen
            center_x, center_y = w // 2, h // 2
            region_size = min(w, h) // 3
            
            self.src_points = np.float32([
                [center_x - region_size, center_y - region_size],  # Top-left
                [center_x + region_size, center_y - region_size],  # Top-right
                [center_x + region_size, center_y + region_size],  # Bottom-right
                [center_x - region_size, center_y + region_size]   # Bottom-left
            ])
            
            # Puntos de destino (verde) - rectángulo vertical fijo
            self.dst_points = np.float32([
                [0, 0],                          # Top-left
                [self.output_width, 0],          # Top-right
                [self.output_width, self.output_height],  # Bottom-right
                [0, self.output_height]          # Bottom-left
            ])
            
            self.transform_applied = False
            self.update_display()
    
    def update_transform_type(self, label):
        """Actualiza el tipo de transformación"""
        transform_map = {
            'Perspectiva': 'perspective',
            'Afín': 'affine',
            'Similaridad': 'similarity'
        }
        self.current_transform = transform_map[label]
        if self.transform_applied:
            self.apply_transform()
    
    def find_closest_point(self, x, y):
        """Encuentra el punto más cercano a las coordenadas dadas"""
        min_dist = float('inf')
        closest_idx = None
        
        # Solo permitimos mover los puntos de origen (rojos)
        for i, point in enumerate(self.src_points):
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist < min_dist and dist < self.hover_tolerance:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def on_press(self, event):
        """Maneja el click del mouse"""
        if event.inaxes == self.ax_original:
            self.selected_point = self.find_closest_point(event.xdata, event.ydata)
    
    def on_release(self, event):
        """Maneja cuando se suelta el click"""
        self.selected_point = None
    
    def on_motion(self, event):
        """Maneja el movimiento del mouse"""
        if event.inaxes == self.ax_original:
            # Hovering
            self.hover_point = self.find_closest_point(event.xdata, event.ydata)
            
            # Arrastre
            if self.selected_point is not None:
                self.src_points[self.selected_point] = [event.xdata, event.ydata]
            
            self.update_display()
    
    def apply_transform(self, event=None):
        """Aplica la transformación actual"""
        try:
            ##################### CÓDIGO PRINCIPAL #####################
            if self.current_transform == 'perspective':
                matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                self.warped_image = cv2.warpPerspective(self.img, matrix, 
                                                      (self.output_width, self.output_height))
            elif self.current_transform == 'affine':
                matrix = cv2.getAffineTransform(self.src_points[:3], self.dst_points[:3])
                self.warped_image = cv2.warpAffine(self.img, matrix, 
                                                 (self.output_width, self.output_height))
            else:  # similarity
                matrix = cv2.getRotationMatrix2D(
                    tuple(self.dst_points[0]), 
                    cv2.minAreaRect(self.dst_points)[2], 
                    1.0
                )
                self.warped_image = cv2.warpAffine(self.img, matrix, 
                                                 (self.output_width, self.output_height))
            ##################### CÓDIGO PRINCIPAL #####################
            
            self.transform_applied = True
            self.update_display()
        except Exception as e:
            print(f"Error al aplicar transformación: {e}")
    
    def update_display(self):
        """Actualiza la visualización"""
        # Limpiar axes
        self.ax_original.clear()
        self.ax_transformed.clear()
        
        # Mostrar imagen original con puntos
        self.ax_original.imshow(self.img)
        self.ax_original.set_title(f'Original - {self.img_path.name}\nAjusta el cuadro rojo alrededor de la región a transformar')
        self.ax_original.axis('off')
        
        # Dibujar polígono de origen (rojo)
        img_with_poly = self.img.copy()
        pts = self.src_points.reshape((-1, 1, 2))
        cv2.polylines(img_with_poly, [pts.astype(np.int32)], True, (255, 0, 0), 4)
        
        # Dibujar puntos de origen (rojos)
        for i, point in enumerate(self.src_points):
            # Color del punto según estado
            if i == self.selected_point:
                color = (255, 255, 0)  # Amarillo cuando está seleccionado
            elif i == self.hover_point:
                color = (255, 165, 0)  # Naranja cuando tiene hover
            else:
                color = (255, 0, 0)    # Rojo normal
            
            cv2.circle(img_with_poly, tuple(point.astype(int)), 15, color, -1)
            
            # Etiqueta del punto
            text_pos = (int(point[0] + 20), int(point[1] + 10))
            cv2.putText(img_with_poly, str(i + 1), text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        self.ax_original.imshow(img_with_poly)
        
        # Mostrar imagen transformada o mensaje
        if self.transform_applied and hasattr(self, 'warped_image'):
            self.ax_transformed.imshow(self.warped_image)
            self.ax_transformed.set_title(f'Resultado: {self.output_width}x{self.output_height}')
        else:
            # Mostrar rectángulo vertical de destino (verde)
            dummy_img = np.zeros((self.output_height + 100, self.output_width + 100, 3), dtype=np.uint8)
            dummy_img[50:50+self.output_height, 50:50+self.output_width] = [0, 100, 0]  # Verde oscuro
            
            # Dibujar borde del rectángulo
            cv2.rectangle(dummy_img, (50, 50), 
                         (50 + self.output_width, 50 + self.output_height), 
                         (0, 255, 0), 3)
            
            self.ax_transformed.imshow(dummy_img)
            self.ax_transformed.set_title(f'Destino Vertical ({self.output_width}x{self.output_height})\nClic en "Aplicar" para transformar')
        
        self.ax_transformed.axis('off')
        plt.draw()
    
    def next_image(self, event):
        """Carga la siguiente imagen"""
        self.current_image_idx = (self.current_image_idx + 1) % len(self.images)
        self.load_current_image()
    
    def prev_image(self, event):
        """Carga la imagen anterior"""
        self.current_image_idx = (self.current_image_idx - 1) % len(self.images)
        self.load_current_image()
    
    def save_image(self, event):
        """Guarda la imagen transformada"""
        if hasattr(self, 'warped_image'):
            transform_suffix = self.current_transform
            output_filename = f"{self.img_path.stem}_{transform_suffix}_{self.output_width}x{self.output_height}{self.img_path.suffix}"
            output_path = self.output_dir / output_filename
            cv2.imwrite(str(output_path), cv2.cvtColor(self.warped_image, cv2.COLOR_RGB2BGR))
            self.ax_transformed.set_title(f'Guardada en: {output_path.name}')
            plt.draw()
    
    def run(self):
        """Ejecuta la aplicación"""
        plt.tight_layout(rect=[0, 0.03, 0.95, 1.0])  # Ajusta los márgenes
        plt.subplots_adjust(wspace=0.1)  # Reduce el espacio entre subplots
        plt.show()

if __name__ == "__main__":
    print("=== Transformador de Imágenes Interactivo ===")
    print("Instrucciones:")
    print("1. Ajusta el cuadro ROJO alrededor de la región que quieres transformar")
    print("2. El cuadro VERDE vertical (400x600) es el destino fijo")
    print("3. Selecciona el tipo de transformación en el panel de control")
    print("4. Clic en 'Aplicar' para ver el resultado")
    print("5. Usa 'Guardar' para exportar la imagen transformada")
    print("6. 'Reset Puntos' para volver a la posición inicial")
    print("==========================================")
    
    app = TransformadorInteractivo()
    app.run()