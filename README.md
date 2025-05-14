# Skin Cancer Detection
[data source](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset)
## Project Overview
This project aims to develop a Convolutional Neural Network (CNN) model to detect skin cancer from images. The project utilizes a dataset of dermatoscopic images to train and validate the model, enhancing the early detection and diagnosis of skin cancer.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/ShubhamCoder-In/skin_cancer_detection.git
    ```
2. Navigate to the project directory:
    ```sh
    cd skin_cancer_detection
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it in the `data` directory.
2. Preprocess the dataset:
    ```sh
    python preprocess.py
    ```
3. Train the model:
    ```sh
    python train.py
    ```
4. Evaluate the model:
    ```sh
    python evaluate.py
    ```

## Dataset
The dataset used in this project is the HAM10000 dataset, which contains 10,015 dermatoscopic images of different types of skin lesions.

## Model
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture includes several convolutional layers, max-pooling layers, and fully connected layers.

## Results
The trained model achieves an accuracy of XX% on the test dataset. Detailed results and performance metrics can be found in the `results` directory.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING](CONTRIBUTING.md) guidelines before submitting a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Additional Information on Licenses

#### Choosing the Best License

For a medical project, selecting the right license is crucial. Some common licenses are:

- **MIT License**: Allows broad use and minimal restrictions.
- **Apache License 2.0**: Adds patent protection and requires stating changes.
- **GNU GPL**: Ensures derivative works are also open-source.
- **CC BY-NC**: Suitable for non-software components, restricts commercial use.

Ensure you read the full text and consult legal advice if needed.

## Acknowledgments
- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- TensorFlow and Keras communities for their amazing tools and libraries.
