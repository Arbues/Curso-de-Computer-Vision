
##falta mejorar el output, las imagens que muestra como resultado, lo demas esta completo, tambien falta meter todo esto en una carpeta xD
import os
import sys
import numpy as np
import cv2
from tkinter import Tk, filedialog, simpledialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
import pandas as pd

class RegionSelector:
    def __init__(self):
        self.image_folder = "Problema1/Imagenes"  # Carpeta donde se buscarán las imágenes (sin tilde)
        self.output_folder = "Problema1/test"     # Carpeta donde se guardarán los resultados
        self.output_file = os.path.join(self.output_folder, "color_statistics.csv")
        self.img = None
        self.points = []
        self.current_filename = ""
        self.all_pixels_data = []  # Aquí guardaremos todos los datos de píxeles de todas las regiones
        self.fig = None
        self.ax = None
        self.polygon = None
        self.accepted = False
        self.cancel_all = False
        self.drawing_enabled = True  # Estado para controlar si se permiten dibujar puntos
    
    def find_images(self):
        """Encuentra todas las imágenes en la carpeta especificada"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Verificar si la carpeta existe
        if not os.path.exists(self.image_folder):
            print(f"La carpeta '{self.image_folder}' no existe. Creándola...")
            os.makedirs(self.image_folder)
            print(f"Por favor, coloca imágenes en la carpeta '{self.image_folder}' y ejecuta de nuevo el programa.")
            sys.exit(0)
        
        # Crear carpeta de resultados si no existe
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Creada carpeta '{self.output_folder}' para guardar resultados.")
        
        image_files = []
        for file in os.listdir(self.image_folder):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(self.image_folder, file))
        
        if not image_files:
            print(f"No se encontraron imágenes en la carpeta '{self.image_folder}'.")
            sys.exit(0)
            
        return image_files
    
    def on_click(self, event):
        """Maneja eventos de clic para seleccionar puntos"""
        if event.inaxes != self.ax or not self.drawing_enabled:
            return
        
        # Añadir el punto a la lista
        self.points.append((event.xdata, event.ydata))
        
        # Actualizar la visualización
        x, y = zip(*self.points) if self.points else ([], [])
        
        if self.polygon:
            self.polygon.remove()
        
        if len(self.points) > 0:
            self.polygon = Polygon(np.array(self.points), alpha=0.3, color='yellow')
            self.ax.add_patch(self.polygon)
        
        # Mostrar puntos
        self.ax.plot(x, y, 'ro')
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Maneja eventos de teclado para modos de navegación"""
        if event.key == 'escape':
            # Desactivar el modo de zoom y volver al modo de dibujo
            self.toggle_drawing_mode(None)

    def toggle_drawing_mode(self, event):
        """Alterna entre modo de dibujo y modo de navegación"""
        self.drawing_enabled = not self.drawing_enabled
        
        # Actualizar la interfaz para indicar el modo actual
        if self.drawing_enabled:
            self.ax.set_title(f"MODO DIBUJO: {self.current_filename}")
            self.btn_toggle.label.set_text('Activar Zoom')
            # Desactivar herramientas de navegación
            self.fig.canvas.toolbar.pan()  # Desactivar el modo pan si está activo
            self.fig.canvas.toolbar.zoom()  # Desactivar el modo zoom si está activo
        else:
            self.ax.set_title(f"MODO ZOOM: {self.current_filename}")
            self.btn_toggle.label.set_text('Activar Dibujo')
        
        self.fig.canvas.draw()

    def on_accept(self, event):
        """Maneja el evento cuando se acepta la selección"""
        if len(self.points) < 3:
            messagebox.showwarning("Advertencia", "Por favor, selecciona al menos 3 puntos para formar una región cerrada.")
            return
        
        # Crear una máscara para la región seleccionada
        mask = self.create_mask()
        
        # Extraer los valores de color de los píxeles dentro de la máscara
        self.process_region(mask)
        
        # Marcar como aceptado para pasar a la siguiente imagen
        self.accepted = True
        plt.close(self.fig)

    def on_clear(self, event):
        """Limpia todos los puntos seleccionados"""
        self.points = []
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        
        if self.drawing_enabled:
            self.ax.set_title(f"MODO DIBUJO: {self.current_filename}")
        else:
            self.ax.set_title(f"MODO ZOOM: {self.current_filename}")
            
        self.fig.canvas.draw()

    def on_cancel(self, event):
        """Cancela la selección actual y pasa a la siguiente imagen"""
        self.accepted = True  # Marcamos como aceptado para pasar a la siguiente, pero no procesamos
        plt.close(self.fig)

    def on_cancel_all(self, event):
        """Cancela todo el proceso y termina el programa"""
        if messagebox.askyesno("Cancelar todo", "¿Estás seguro de que quieres cancelar el proceso completo?"):
            self.cancel_all = True
            self.accepted = True
            plt.close(self.fig)

    def create_mask(self):
        """Crea una máscara para la región seleccionada"""
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        points_array = np.array([self.points], dtype=np.int32)
        cv2.fillPoly(mask, points_array, 255)
        return mask

    def process_region(self, mask):
        """Procesa los píxeles dentro de la región seleccionada"""
        # Obtener píxeles dentro de la máscara
        pixels = self.img[mask == 255]
        
        if len(pixels) == 0:
            messagebox.showwarning("Advertencia", "No hay píxeles en la región seleccionada.")
            return
        
        # Convertir los píxeles a RGB y HSV
        pixels_rgb = pixels[..., ::-1]  # BGR a RGB
        pixels_hsv = cv2.cvtColor(np.array([pixels]), cv2.COLOR_BGR2HSV)[0]
        
        # Normalizar valores de HSV
        h_normalized = pixels_hsv[:, 0] / 179.0  # H está en rango [0, 179] en OpenCV
        s_normalized = pixels_hsv[:, 1] / 255.0
        v_normalized = pixels_hsv[:, 2] / 255.0
        
        # Guardar información para cada píxel
        for i in range(len(pixels)):
            self.all_pixels_data.append({
                'image': self.current_filename,
                'r': pixels_rgb[i, 0],
                'g': pixels_rgb[i, 1],
                'b': pixels_rgb[i, 2],
                'h': h_normalized[i],
                's': s_normalized[i],
                'v': v_normalized[i],
                'h_raw': pixels_hsv[i, 0],
                's_raw': pixels_hsv[i, 1],
                'v_raw': pixels_hsv[i, 2]
            })
        
        print(f"Procesados {len(pixels)} píxeles de la imagen {self.current_filename}")

    def select_region(self, image_path):
        """Muestra la interfaz para seleccionar una región en la imagen"""
        self.current_filename = os.path.basename(image_path)
        self.img = cv2.imread(image_path)
        
        if self.img is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return False
        
        self.points = []
        self.accepted = False
        self.drawing_enabled = True
        
        # Crear la figura para mostrar la imagen y permitir la selección
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.ax.set_title(f"MODO DIBUJO: {self.current_filename}")
        
        # Conectar los eventos
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Añadir botones
        ax_accept = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_clear = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_cancel = plt.axes([0.59, 0.05, 0.1, 0.075])
        ax_cancel_all = plt.axes([0.48, 0.05, 0.1, 0.075])
        ax_toggle = plt.axes([0.37, 0.05, 0.1, 0.075])
        
        btn_accept = Button(ax_accept, 'Aceptar')
        btn_accept.on_clicked(self.on_accept)
        
        btn_clear = Button(ax_clear, 'Limpiar')
        btn_clear.on_clicked(self.on_clear)
        
        btn_cancel = Button(ax_cancel, 'Cancelar')
        btn_cancel.on_clicked(self.on_cancel)
        
        btn_cancel_all = Button(ax_cancel_all, 'Cancelar Todo')
        btn_cancel_all.on_clicked(self.on_cancel_all)
        
        self.btn_toggle = Button(ax_toggle, 'Activar Zoom')
        self.btn_toggle.on_clicked(self.toggle_drawing_mode)
        
        # Mostrar la figura
        plt.tight_layout()
        plt.show()
        
        if self.cancel_all:
            return None  # Señal especial para cancelar todo
        
        return self.accepted

    def analyze_colors(self):
        """Analiza los colores de todas las regiones seleccionadas"""
        if not self.all_pixels_data:
            print("No hay datos para analizar.")
            return
        
        # Convertir a DataFrame
        df = pd.DataFrame(self.all_pixels_data)
        
        # Guardar todos los datos en un CSV
        df.to_csv(self.output_file, index=False)
        print(f"Datos guardados en {self.output_file}")
        
        # Calcular estadísticas
        stats = {
            'rgb_min': [df['r'].min(), df['g'].min(), df['b'].min()],
            'rgb_max': [df['r'].max(), df['g'].max(), df['b'].max()],
            'hsv_min': [df['h'].min(), df['s'].min(), df['v'].min()],
            'hsv_max': [df['h'].max(), df['s'].max(), df['v'].max()],
            'hsv_raw_min': [df['h_raw'].min(), df['s_raw'].min(), df['v_raw'].min()],
            'hsv_raw_max': [df['h_raw'].max(), df['s_raw'].max(), df['v_raw'].max()]
        }
        
        # Mostrar estadísticas
        print("\n=== ESTADÍSTICAS DE COLOR ===")
        print(f"Total de píxeles analizados: {len(df)}")
        print(f"Total de imágenes procesadas: {df['image'].nunique()}")
        
        print("\nValores RGB (0-255):")
        print(f"  Mínimo: R={stats['rgb_min'][0]}, G={stats['rgb_min'][1]}, B={stats['rgb_min'][2]}")
        print(f"  Máximo: R={stats['rgb_max'][0]}, G={stats['rgb_max'][1]}, B={stats['rgb_max'][2]}")
        
        print("\nValores HSV (normalizados 0-1):")
        print(f"  Mínimo: H={stats['hsv_min'][0]:.4f}, S={stats['hsv_min'][1]:.4f}, V={stats['hsv_min'][2]:.4f}")
        print(f"  Máximo: H={stats['hsv_max'][0]:.4f}, S={stats['hsv_max'][1]:.4f}, V={stats['hsv_max'][2]:.4f}")
        
        print("\nValores HSV (formato OpenCV):")
        print(f"  Mínimo: H={stats['hsv_raw_min'][0]}, S={stats['hsv_raw_min'][1]}, V={stats['hsv_raw_min'][2]}")
        print(f"  Máximo: H={stats['hsv_raw_max'][0]}, S={stats['hsv_raw_max'][1]}, V={stats['hsv_raw_max'][2]}")
        
        # Mostrar resultados en una forma adecuada para usar en código
        print("\n=== CÓDIGO PARA USAR ESTOS VALORES ===")
        print("# Para formato normalizado (0-1):")
        print(f"lower_hsv = np.array([{stats['hsv_min'][0]:.4f}, {stats['hsv_min'][1]:.4f}, {stats['hsv_min'][2]:.4f}])")
        print(f"upper_hsv = np.array([{stats['hsv_max'][0]:.4f}, {stats['hsv_max'][1]:.4f}, {stats['hsv_max'][2]:.4f}])")
        
        print("\n# Para formato OpenCV:")
        print(f"lower_hsv_cv = np.array([{int(stats['hsv_raw_min'][0])}, {int(stats['hsv_raw_min'][1])}, {int(stats['hsv_raw_min'][2])}])")
        print(f"upper_hsv_cv = np.array([{int(stats['hsv_raw_max'][0])}, {int(stats['hsv_raw_max'][1])}, {int(stats['hsv_raw_max'][2])}])")
        
        # Guardar estas estadísticas en un archivo de texto
        stats_file = os.path.join(self.output_folder, "color_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("=== ESTADÍSTICAS DE COLOR ===\n")
            f.write(f"Total de píxeles analizados: {len(df)}\n")
            f.write(f"Total de imágenes procesadas: {df['image'].nunique()}\n\n")
            
            f.write("Valores RGB (0-255):\n")
            f.write(f"  Mínimo: R={stats['rgb_min'][0]}, G={stats['rgb_min'][1]}, B={stats['rgb_min'][2]}\n")
            f.write(f"  Máximo: R={stats['rgb_max'][0]}, G={stats['rgb_max'][1]}, B={stats['rgb_max'][2]}\n\n")
            
            f.write("Valores HSV (normalizados 0-1):\n")
            f.write(f"  Mínimo: H={stats['hsv_min'][0]:.4f}, S={stats['hsv_min'][1]:.4f}, V={stats['hsv_min'][2]:.4f}\n")
            f.write(f"  Máximo: H={stats['hsv_max'][0]:.4f}, S={stats['hsv_max'][1]:.4f}, V={stats['hsv_max'][2]:.4f}\n\n")
            
            f.write("Valores HSV (formato OpenCV):\n")
            f.write(f"  Mínimo: H={stats['hsv_raw_min'][0]}, S={stats['hsv_raw_min'][1]}, V={stats['hsv_raw_min'][2]}\n")
            f.write(f"  Máximo: H={stats['hsv_raw_max'][0]}, S={stats['hsv_raw_max'][1]}, V={stats['hsv_raw_max'][2]}\n\n")
            
            f.write("=== CÓDIGO PARA USAR ESTOS VALORES ===\n")
            f.write("# Para formato normalizado (0-1):\n")
            f.write(f"lower_hsv = np.array([{stats['hsv_min'][0]:.4f}, {stats['hsv_min'][1]:.4f}, {stats['hsv_min'][2]:.4f}])\n")
            f.write(f"upper_hsv = np.array([{stats['hsv_max'][0]:.4f}, {stats['hsv_max'][1]:.4f}, {stats['hsv_max'][2]:.4f}])\n\n")
            
            f.write("# Para formato OpenCV:\n")
            f.write(f"lower_hsv_cv = np.array([{int(stats['hsv_raw_min'][0])}, {int(stats['hsv_raw_min'][1])}, {int(stats['hsv_raw_min'][2])}])\n")
            f.write(f"upper_hsv_cv = np.array([{int(stats['hsv_raw_max'][0])}, {int(stats['hsv_raw_max'][1])}, {int(stats['hsv_raw_max'][2])}])\n")
        
        print(f"Estadísticas guardadas en {stats_file}")
        
        # Visualizar los resultados
        self.visualize_color_ranges(stats)
        
        return stats
    
    def visualize_color_ranges(self, stats):
        """Visualiza los rangos de color en los espacios HSV y RGB, incluida vista 3D"""
        # 1. Crear visualizaciones 2D como ya tenías
        h_range = np.linspace(stats['hsv_min'][0], stats['hsv_max'][0], 100)
        s_range = np.linspace(stats['hsv_min'][1], stats['hsv_max'][1], 100)
        
        # Crear una cuadrícula 2D para H y S
        h_grid, s_grid = np.meshgrid(h_range, s_range)
        
        # Crear arrays 3D para HSV con valor V fijo
        hsv_min = np.zeros((100, 100, 3))
        hsv_min[:,:,0] = h_grid
        hsv_min[:,:,1] = s_grid
        hsv_min[:,:,2] = stats['hsv_min'][2]
        
        hsv_max = np.zeros((100, 100, 3))
        hsv_max[:,:,0] = h_grid
        hsv_max[:,:,1] = s_grid
        hsv_max[:,:,2] = stats['hsv_max'][2]
        
        # Convertir HSV a RGB para visualización
        rgb_min = self.hsv_to_rgb(hsv_min)
        rgb_max = self.hsv_to_rgb(hsv_max)
        
        # Visualizar 2D
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(rgb_min)
        axes[0].set_title(f'Rango HSV con V mínimo ({stats["hsv_min"][2]:.2f})')
        axes[0].set_xlabel('Matiz (H)')
        axes[0].set_ylabel('Saturación (S)')
        
        axes[1].imshow(rgb_max)
        axes[1].set_title(f'Rango HSV con V máximo ({stats["hsv_max"][2]:.2f})')
        axes[1].set_xlabel('Matiz (H)')
        axes[1].set_ylabel('Saturación (S)')
        
        plt.tight_layout()
        vis_file = os.path.join(self.output_folder, 'rango_hsv_visualizado_2d.png')
        plt.savefig(vis_file)
        plt.show()
        print(f"Visualización 2D guardada en {vis_file}")
        
        # 2. Crear visualizaciones 3D de los píxeles seleccionados
        self.visualize_rgb_3d()
        self.visualize_hsv_3d()

    def visualize_rgb_3d(self):
        """Visualiza los píxeles seleccionados en el espacio de color RGB en 3D"""
        if not self.all_pixels_data:
            print("No hay datos para visualizar en 3D.")
            return
        
        # Convertir datos a DataFrame para facilitar el trabajo
        df = pd.DataFrame(self.all_pixels_data)
        
        # Crear figura 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Graficar cada punto en el espacio RGB
        scatter = ax.scatter(df['r'], df['g'], df['b'], 
                            c=df[['r', 'g', 'b']].values/255,  # Normalizar para colores
                            marker='o', s=15, alpha=0.7)
        
        # Configurar ejes
        ax.set_xlabel('Rojo (R)', fontsize=12)
        ax.set_ylabel('Verde (G)', fontsize=12)
        ax.set_zlabel('Azul (B)', fontsize=12)
        
        # Establecer límites para el cubo RGB
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        
        # Dibujar el cubo RGB (opcional)
        r = [0, 255]
        g = [0, 255]
        b = [0, 255]
        
        # Lista de 8 vértices del cubo RGB
        vertices = [(R, G, B) for R in r for G in g for B in b]
        
        # Dibujar líneas del cubo
        for i, j in [
            (0, 1), (0, 2), (1, 3), (2, 3),  # Base inferior
            (4, 5), (4, 6), (5, 7), (6, 7),  # Base superior
            (0, 4), (1, 5), (2, 6), (3, 7)   # Conectores verticales
        ]:
            ax.plot3D(*zip(vertices[i], vertices[j]), color='gray', alpha=0.5)
        
        # Añadir título
        ax.set_title('Distribución de Píxeles en Espacio de Color RGB', fontsize=14)
        
        # Añadir una nota sobre los colores
        fig.text(0.1, 0.01, 'Cada punto representa un píxel con su color correspondiente', fontsize=10)
        
        # Ajustar la vista
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        vis_file = os.path.join(self.output_folder, 'distribucion_rgb_3d.png')
        plt.savefig(vis_file)
        plt.show()
        print(f"Visualización 3D RGB guardada en {vis_file}")

    def visualize_hsv_3d(self):
        """Visualiza los píxeles seleccionados en el espacio de color HSV en 3D"""
        if not self.all_pixels_data:
            print("No hay datos para visualizar en 3D.")
            return
        
        # Convertir datos a DataFrame
        df = pd.DataFrame(self.all_pixels_data)
        
        # Crear figura 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convertir valores HSV a coordenadas cilíndricas/cónicas para mejor visualización
        # H como ángulo, S como radio, V como altura
        theta = df['h'] * 2 * np.pi  # H mapeado a [0, 2π]
        radius = df['s']             # S ya está en [0, 1]
        height = df['v']             # V ya está en [0, 1]
        
        # Convertir a coordenadas cartesianas para graficar
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height
        
        # Colores basados en HSV para cada punto
        colors = np.array([self.hsv_to_rgb_single(h, s, v) for h, s, v in zip(df['h'], df['s'], df['v'])])
        
        # Graficar puntos
        scatter = ax.scatter(x, y, z, c=colors, marker='o', s=15, alpha=0.7)
        
        # Añadir un círculo en la base para referencia
        theta_circle = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta_circle)
        y_circle = np.sin(theta_circle)
        z_circle = np.zeros_like(theta_circle)
        ax.plot(x_circle, y_circle, z_circle, color='gray', alpha=0.5)
        
        # Añadir líneas de referencia desde el origen
        ax.plot([0, 0], [0, 0], [0, 1], color='gray', alpha=0.5)  # Eje V
        
        # Configurar ejes
        ax.set_xlabel('X (basado en H y S)', fontsize=12)
        ax.set_ylabel('Y (basado en H y S)', fontsize=12)
        ax.set_zlabel('Valor (V)', fontsize=12)
        
        # Establecer límites
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(0, 1.1)
        
        # Añadir título
        ax.set_title('Distribución de Píxeles en Espacio de Color HSV', fontsize=14)
        
        # Ajustar la vista
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        vis_file = os.path.join(self.output_folder, 'distribucion_hsv_3d.png')
        plt.savefig(vis_file)
        plt.show()
        print(f"Visualización 3D HSV guardada en {vis_file}")

    def hsv_to_rgb_single(self, h, s, v):
        """Convierte un solo valor HSV a RGB"""
        # Crear un array 1x1x3 con el valor HSV
        hsv = np.zeros((1, 1, 3), dtype=np.float32)
        hsv[0, 0, 0] = h * 179  # Escalar a formato OpenCV
        hsv[0, 0, 1] = s * 255
        hsv[0, 0, 2] = v * 255
        
        # Convertir a uint8 para OpenCV
        hsv = hsv.astype(np.uint8)
        
        # Convertir a RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Normalizar a [0, 1] para matplotlib
        return rgb[0, 0] / 255.0

    def hsv_to_rgb(self, hsv):
        """Convierte HSV a RGB para visualización"""
        # Convertir de nuestro formato normalizado a formato OpenCV
        hsv_cv = hsv.copy()
        hsv_cv[:,:,0] = hsv[:,:,0] * 179
        hsv_cv[:,:,1] = hsv[:,:,1] * 255
        hsv_cv[:,:,2] = hsv[:,:,2] * 255
        
        # Convertir a enteros de 8 bits para OpenCV
        hsv_cv = hsv_cv.astype(np.uint8)
        
        # Usar OpenCV para convertir a RGB
        rgb_cv = cv2.cvtColor(hsv_cv, cv2.COLOR_HSV2RGB)
        
        return rgb_cv

    def run(self):
        """Ejecuta el programa principal"""
        print("=== SELECTOR DE REGIONES PARA ANÁLISIS DE COLOR ===")
        print(f"Buscando imágenes en la carpeta: {self.image_folder}")
        
        # Encontrar todas las imágenes
        image_files = self.find_images()
        print(f"Se encontraron {len(image_files)} imágenes.")
        print(f"Los resultados se guardarán en la carpeta: {self.output_folder}")
        print("\nInstrucciones:")
        print("- Usa el botón 'Activar Zoom' para alternar entre dibujar puntos y hacer zoom")
        print("- En modo zoom, usa las herramientas de navegación de matplotlib")
        print("- Presiona 'Esc' para volver al modo dibujo desde el modo zoom")
        print("- Dibuja al menos 3 puntos para formar una región cerrada")
        print("- Usa 'Cancelar Todo' para salir del programa en cualquier momento")
        
        # Procesar cada imagen
        for i, image_path in enumerate(image_files):
            print(f"\nProcesando imagen {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Mostrar la interfaz para seleccionar una región
            result = self.select_region(image_path)
            
            if result is None:  # Cancelar todo
                print("Proceso cancelado por el usuario.")
                return
            
            if not result:
                print(f"Saltando imagen {os.path.basename(image_path)}")
        
        if self.all_pixels_data:
            # Analizar los colores de todas las regiones seleccionadas
            self.analyze_colors()
            
            print("\n¡Análisis completado!")
            print(f"Los resultados completos están disponibles en la carpeta: {self.output_folder}")
        else:
            print("\nNo se seleccionaron regiones para analizar.")

if __name__ == "__main__":
    app = RegionSelector()
    app.run()