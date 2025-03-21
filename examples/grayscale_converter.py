def rgb_to_grayscale(image):
    grayscale = []
    for row in image:
        gray_row = []
        for r, g, b in row:
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            gray_row.append(gray)
        grayscale.append(gray_row)
    return grayscale

def main():
    image = [[(100, 150, 200) for _ in range(100)] for _ in range(100)]
    gray = rgb_to_grayscale(image)
