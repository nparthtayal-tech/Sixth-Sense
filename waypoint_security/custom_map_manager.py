"""
custom_map_manager.py
=====================
Loads a custom floor plan image, computes the real-world scale,
and extracts obstacle boundaries for spoofing collision detection.
"""

import numpy as np
import matplotlib.pyplot as plt

class CustomMapManager:
    def __init__(self, image_path: str, real_width_m: float = 175.0, real_height_m: float = 185.0):
        self.image_path = image_path
        self.real_width_m = real_width_m
        self.real_height_m = real_height_m
        
        # Load image (supports PNG, JPG via matplotlib)
        self.image_data = plt.imread(self.image_path)
        
        # Handle RGBA / RGB / Grayscale
        if self.image_data.ndim == 3:
            # Convert to grayscale using luminance
            self.gray_data = np.dot(self.image_data[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            self.gray_data = self.image_data

        self.height_px, self.width_px = self.gray_data.shape
        
        # Calculate scale
        self.meters_per_pixel_x = self.real_width_m / self.width_px
        self.meters_per_pixel_y = self.real_height_m / self.height_px
        
        # Create obstacle mask. 
        # Assuming dark lines/pixels are walls (value < 0.5 for floats 0-1, or < 128 for 0-255)
        # plt.imread usually returns 0.0-1.0 for PNGs, 0-255 for JPGs.
        if self.gray_data.max() <= 1.0:
            threshold = 0.5
        else:
            threshold = 128.0
            
        self.obstacle_mask = self.gray_data < threshold
        
    def get_extent(self):
        """Returns the [xmin, xmax, ymin, ymax] for matplotlib imshow."""
        return [0.0, self.real_width_m, 0.0, self.real_height_m]
        
    def is_collision(self, x: float, y: float) -> bool:
        """
        Check if physical coordinates (x, y) map to an obstacle.
        Origin (0,0) is bottom-left in physical space, but images are top-left origin.
        """
        # Out of bounds is considered a collision (outside the building)
        if x < 0 or x >= self.real_width_m or y < 0 or y >= self.real_height_m:
            return True
            
        px = int(x / self.meters_per_pixel_x)
        # Y is flipped because image (0,0) is top-left, but physical (0,0) is bottom-left
        py = int((self.real_height_m - y) / self.meters_per_pixel_y)
        
        # Keep bounds safe
        px = np.clip(px, 0, self.width_px - 1)
        py = np.clip(py, 0, self.height_px - 1)
        
        return bool(self.obstacle_mask[py, px])
