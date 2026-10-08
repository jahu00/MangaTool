This is a tool for cropping, splitting and saving manga pages to a CBZ file.

- Takes a folder of images as an input.
- Checks if images have desired height (for detecting a failed dump).
- Detects padding that isn't part of the page (with some options to finetune the process)
- Optionally detects grayscale pages and saves them as single-channel images to reduce file size, with an adjustable color tolerance.
- Saves split pages as a new folder of images of a CBZ file