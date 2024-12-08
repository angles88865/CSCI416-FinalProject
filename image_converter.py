from PIL import Image
import os


def resize_images_in_directory(input_dir, output_dir, size=(200, 200)):
    """
    Resizes all images in the input directory and saves them to the output directory.

    :param input_dir: Directory containing the input images
    :param output_dir: Directory to save the resized images
    :param size: Tuple (width, height) for the new size
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process each file in the input directory
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Check if the file is an image
            try:
                with Image.open(input_path) as img:
                    # Resize and save the image
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)
                    img_rotated = img_resized.rotate(270, expand=True)
                    img_rotated.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    except Exception as e:
        print(f"Error processing directory: {e}")


# Example usage
if __name__ == "__main__":
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    for c in classes:
        input_directory = f"asl-alphabet/LiveAction/{c}"
        output_directory = f"asl-alphabet/LiveActionConverted/{c}"
        resize_images_in_directory(input_directory, output_directory)
