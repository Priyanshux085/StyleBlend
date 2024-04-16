# StyleBlend

StyleBlend is a project for artistic style transfer using the VGG19 convolutional neural network architecture. It allows users to blend the content of one image with the style of another image, creating visually appealing compositions.

## Overview

StyleBlend utilizes the powerful feature representations learned by the VGG19 model pretrained on the ImageNet dataset. By extracting and manipulating feature maps from different layers of the network, it achieves style transfer by minimizing the content and style loss between the input images.

## Features

- Content and style image selection
- Customizable style transfer parameters
- Iterative optimization for fine-tuning
- Visualize progress with output images

## Usage

1. Clone the repository.
2. Place your content and style images in the designated directories.
3. Run the `style_transfer` function with paths to your content and style images.
4. Adjust the number of iterations, content weight, and style weight for desired results.
5. Retrieve the output image with the stylized blend of content and style.

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## Example

```python
content_path = 'content_image.jpg'
style_path = 'style_image.jpg'
output_image = style_transfer(content_path, style_path)
plt.imshow(output_image)
plt.axis('off')
plt.show()
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.