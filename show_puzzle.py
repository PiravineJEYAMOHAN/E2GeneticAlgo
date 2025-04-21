import json
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
from io import BytesIO
from PIL import Image, ImageDraw

# Fonction pour charger une image SVG et la convertir en PNG
def load_svg_as_png(pattern_id):
    try:
        # Convertir le fichier SVG en PNG
        png_data = cairosvg.svg2png(url=f'patterns/pattern{pattern_id}.svg')
        # Charger l'image PNG avec PIL
        return Image.open(BytesIO(png_data))
    except Exception as e:
        print(f"Warning: {e}")
        return None

# Fonction pour créer une image de pièce
def create_piece_image(piece):
    size = 100
    image = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(image)

    # Charger les motifs
    top_pattern = load_svg_as_png(piece["haut"])
    right_pattern = load_svg_as_png(piece["droite"])
    bottom_pattern = load_svg_as_png(piece["bas"])
    left_pattern = load_svg_as_png(piece["gauche"])

    # Dessiner les triangles
    if top_pattern:
        top_pattern = top_pattern.resize((size, size))
        mask = Image.new('L', (size, size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon([(0, 0), (size, 0), (size // 2, size // 2)], fill=255)
        image.paste(top_pattern, mask=mask)

    if right_pattern:
        right_pattern = right_pattern.resize((size, size))
        mask = Image.new('L', (size, size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon([(size, 0), (size, size), (size // 2, size // 2)], fill=255)
        image.paste(right_pattern, mask=mask)

    if bottom_pattern:
        bottom_pattern = bottom_pattern.resize((size, size))
        mask = Image.new('L', (size, size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon([(0, size), (size // 2, size // 2), (size, size)], fill=255)
        image.paste(bottom_pattern, mask=mask)

    if left_pattern:
        left_pattern = left_pattern.resize((size, size))
        mask = Image.new('L', (size, size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon([(0, 0), (size // 2, size // 2), (0, size)], fill=255)
        image.paste(left_pattern, mask=mask)

    return image

# Fonction pour afficher la grille
def show_puzzle(filename='best_solution.json'):
    # Charger la meilleure solution
    with open(filename, 'r') as f:
        best_solution = json.load(f)

    grid_size = 5
    piece_size = 100
    border_width = 2

    # Calculer la taille totale de l'image de la grille
    total_size = grid_size * (piece_size + border_width) + border_width
    grid_image = Image.new('RGB', (total_size, total_size), color='black')

    # Placer les pièces sur la grille
    for row in range(grid_size):
        for col in range(grid_size):
            piece = best_solution[row][col]
            x = col * (piece_size + border_width) + border_width
            y = row * (piece_size + border_width) + border_width
            piece_image = create_piece_image(piece)
            grid_image.paste(piece_image, (x, y))

    # Afficher la grille
    plt.imshow(grid_image)
    plt.axis('off')
    plt.show()

# Exécuter l'affichage de la grille
if __name__ == "__main__":
    show_puzzle()
