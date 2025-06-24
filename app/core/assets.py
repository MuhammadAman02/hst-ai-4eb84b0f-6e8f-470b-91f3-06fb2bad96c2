"""Advanced professional visual asset management system for color analysis application"""

import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class ColorAnalysisAssetManager:
    """Specialized asset manager for color analysis and fashion applications."""
    
    def __init__(self):
        self.unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY', 'demo_key')
        self.cache_dir = Path("static/images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fashion and beauty specific categories
        self.fashion_categories = [
            'fashion', 'beauty', 'portrait', 'style', 'makeup', 
            'skincare', 'model', 'cosmetics', 'styling'
        ]
        
        self.color_categories = [
            'color', 'palette', 'art', 'design', 'rainbow',
            'colorful', 'gradient', 'spectrum'
        ]
    
    def get_hero_images(self) -> Dict[str, str]:
        """Get collection of hero images for different sections."""
        return {
            'main_hero': self._fetch_contextual_image(
                ['fashion', 'beauty', 'portrait'], 1200, 600, 'main_hero'
            ),
            'color_theory': self._fetch_contextual_image(
                ['color', 'palette', 'art'], 800, 400, 'color_theory_hero'
            ),
            'skin_analysis': self._fetch_contextual_image(
                ['beauty', 'skincare', 'portrait'], 600, 400, 'skin_analysis_hero'
            )
        }
    
    def get_color_palette_examples(self) -> List[str]:
        """Get color palette demonstration images."""
        return [
            self._fetch_contextual_image(
                ['palette', 'color', 'design'], 300, 200, f'palette_example_{i}'
            ) for i in range(4)
        ]
    
    def get_skin_tone_diversity_gallery(self) -> List[str]:
        """Get diverse skin tone reference images."""
        diversity_terms = [
            'diverse+portrait', 'multicultural+beauty', 'skin+tone+diversity',
            'ethnic+beauty', 'global+faces', 'inclusive+beauty'
        ]
        
        return [
            self._fetch_contextual_image(
                [term.replace('+', ' ')], 250, 300, f'diversity_{i}'
            ) for i, term in enumerate(diversity_terms)
        ]
    
    def get_seasonal_color_references(self) -> Dict[str, str]:
        """Get seasonal color palette reference images."""
        seasons = {
            'spring': ['spring', 'fresh', 'light', 'warm'],
            'summer': ['summer', 'soft', 'cool', 'muted'],
            'autumn': ['autumn', 'warm', 'deep', 'rich'],
            'winter': ['winter', 'cool', 'clear', 'dramatic']
        }
        
        return {
            season: self._fetch_contextual_image(
                keywords, 400, 300, f'season_{season}'
            ) for season, keywords in seasons.items()
        }
    
    def get_fashion_styling_examples(self) -> List[str]:
        """Get fashion styling and outfit coordination examples."""
        styling_terms = [
            'fashion+coordination', 'outfit+styling', 'color+matching+clothes',
            'wardrobe+essentials', 'style+guide', 'fashion+colors'
        ]
        
        return [
            self._fetch_contextual_image(
                [term.replace('+', ' ')], 300, 400, f'styling_{i}'
            ) for i, term in enumerate(styling_terms)
        ]
    
    def _fetch_contextual_image(self, categories: List[str], width: int, height: int, cache_key: str) -> str:
        """Fetch contextually relevant images with comprehensive fallback strategy."""
        cache_path = self.cache_dir / f"{cache_key}_{width}x{height}.jpg"
        
        if cache_path.exists():
            return f"/static/images/{cache_path.name}"
        
        # Primary: Unsplash API
        if self.unsplash_access_key != 'demo_key':
            for category in categories:
                try:
                    url = "https://api.unsplash.com/photos/random"
                    params = {
                        'query': category,
                        'w': width,
                        'h': height,
                        'fit': 'crop',
                        'crop': 'faces' if 'portrait' in category else 'center'
                    }
                    headers = {'Authorization': f'Client-ID {self.unsplash_access_key}'}
                    
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        img_url = data['urls']['custom']
                        img_response = requests.get(img_url, timeout=15)
                        if img_response.status_code == 200:
                            with open(cache_path, 'wb') as f:
                                f.write(img_response.content)
                            return f"/static/images/{cache_path.name}"
                except Exception:
                    continue
        
        # Secondary: Lorem Picsum with category-based seeding
        try:
            seed = abs(hash(f"{cache_key}_{categories[0]}")) % 1000
            picsum_url = f"https://picsum.photos/seed/{seed}/{width}/{height}"
            response = requests.get(picsum_url, timeout=10)
            if response.status_code == 200:
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
                return f"/static/images/{cache_path.name}"
        except Exception:
            pass
        
        # Tertiary: Generate professional placeholder
        return self._generate_professional_placeholder(width, height, cache_key, categories[0])
    
    def _generate_professional_placeholder(self, width: int, height: int, cache_key: str, category: str) -> str:
        """Generate professional, category-specific placeholder images."""
        cache_path = self.cache_dir / f"{cache_key}_{width}x{height}.jpg"
        
        # Category-specific color schemes and patterns
        category_configs = {
            'fashion': {
                'colors': [(255, 182, 193), (255, 105, 180), (219, 112, 147)],
                'pattern': 'gradient',
                'text': 'FASHION'
            },
            'beauty': {
                'colors': [(255, 218, 185), (255, 160, 122), (255, 127, 80)],
                'pattern': 'radial',
                'text': 'BEAUTY'
            },
            'color': {
                'colors': [(138, 43, 226), (75, 0, 130), (148, 0, 211)],
                'pattern': 'spectrum',
                'text': 'COLOR'
            },
            'portrait': {
                'colors': [(205, 133, 63), (160, 82, 45), (139, 69, 19)],
                'pattern': 'gradient',
                'text': 'PORTRAIT'
            },
            'palette': {
                'colors': [(255, 99, 71), (255, 165, 0), (255, 215, 0)],
                'pattern': 'blocks',
                'text': 'PALETTE'
            }
        }
        
        # Default configuration
        config = category_configs.get(category, category_configs['fashion'])
        
        # Create base image
        img = Image.new('RGB', (width, height), config['colors'][0])
        draw = ImageDraw.Draw(img)
        
        # Apply pattern based on configuration
        if config['pattern'] == 'gradient':
            self._apply_gradient(img, config['colors'])
        elif config['pattern'] == 'radial':
            self._apply_radial_gradient(img, config['colors'])
        elif config['pattern'] == 'spectrum':
            self._apply_spectrum(img, config['colors'])
        elif config['pattern'] == 'blocks':
            self._apply_color_blocks(img, config['colors'])
        
        # Add subtle text overlay
        try:
            # Try to use a nice font, fall back to default if not available
            font_size = min(width, height) // 10
            font = ImageFont.load_default()
            
            text = config['text']
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Add text with semi-transparent background
            draw.rectangle([x-10, y-5, x+text_width+10, y+text_height+5], 
                         fill=(255, 255, 255, 128))
            draw.text((x, y), text, fill=(0, 0, 0, 180), font=font)
            
        except Exception:
            pass  # Skip text if there's any issue
        
        # Save the image
        img.save(cache_path, 'JPEG', quality=85)
        return f"/static/images/{cache_path.name}"
    
    def _apply_gradient(self, img: Image.Image, colors: List[tuple]):
        """Apply linear gradient to image."""
        width, height = img.size
        for y in range(height):
            ratio = y / height
            if len(colors) >= 2:
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                
                for x in range(width):
                    img.putpixel((x, y), (r, g, b))
    
    def _apply_radial_gradient(self, img: Image.Image, colors: List[tuple]):
        """Apply radial gradient to image."""
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        max_distance = ((width/2)**2 + (height/2)**2)**0.5
        
        for y in range(height):
            for x in range(width):
                distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
                ratio = min(distance / max_distance, 1.0)
                
                if len(colors) >= 2:
                    r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                    g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                    b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                    img.putpixel((x, y), (r, g, b))
    
    def _apply_spectrum(self, img: Image.Image, colors: List[tuple]):
        """Apply spectrum/rainbow effect."""
        width, height = img.size
        for x in range(width):
            ratio = x / width
            color_index = int(ratio * (len(colors) - 1))
            next_index = min(color_index + 1, len(colors) - 1)
            local_ratio = (ratio * (len(colors) - 1)) - color_index
            
            r = int(colors[color_index][0] * (1 - local_ratio) + colors[next_index][0] * local_ratio)
            g = int(colors[color_index][1] * (1 - local_ratio) + colors[next_index][1] * local_ratio)
            b = int(colors[color_index][2] * (1 - local_ratio) + colors[next_index][2] * local_ratio)
            
            for y in range(height):
                img.putpixel((x, y), (r, g, b))
    
    def _apply_color_blocks(self, img: Image.Image, colors: List[tuple]):
        """Apply color blocks pattern."""
        width, height = img.size
        block_width = width // len(colors)
        
        for i, color in enumerate(colors):
            x_start = i * block_width
            x_end = (i + 1) * block_width if i < len(colors) - 1 else width
            
            for x in range(x_start, x_end):
                for y in range(height):
                    img.putpixel((x, y), color)
    
    def generate_color_swatch(self, colors: List[str], width: int = 400, height: int = 100) -> str:
        """Generate a color swatch image from hex colors."""
        cache_key = f"swatch_{'_'.join(colors).replace('#', '')}"
        cache_path = self.cache_dir / f"{cache_key}_{width}x{height}.png"
        
        if cache_path.exists():
            return f"/static/images/{cache_path.name}"
        
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        swatch_width = width // len(colors)
        
        for i, color_hex in enumerate(colors):
            try:
                # Convert hex to RGB
                color_hex = color_hex.lstrip('#')
                rgb = tuple(int(color_hex[j:j+2], 16) for j in (0, 2, 4))
                
                x_start = i * swatch_width
                x_end = (i + 1) * swatch_width if i < len(colors) - 1 else width
                
                draw.rectangle([x_start, 0, x_end, height], fill=rgb)
                
            except ValueError:
                # Skip invalid colors
                continue
        
        img.save(cache_path, 'PNG')
        return f"/static/images/{cache_path.name}"