# PyTorch Neural Style Transfer 🎨

This project is a Python implementation of the Neural Style Transfer algorithm, originally described in the paper "A Neural Algorithm of Artistic Style" by Gatys et al. It allows you to take two images—a content image and a style image—and blend them together to create a new image that has the content of the first and the artistic style of the second.



## Features

* Uses a pre-trained **VGG19** network to extract content and style features.
* Employs the **L-BFGS optimizer**, which often produces higher-quality results.
* Includes **Total Variation Loss** as a regularization term to create smoother, more coherent images.
* All settings are centralized in a simple and easy-to-use **"Control Panel"** directly in the main script.

---
## Setup

To run this project on your local machine, follow these steps.

**1. Clone the repository:**
```bash
git clone <your-github-repo-url>
cd <your-repo-name>
