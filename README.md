# Fact-Checking System

This is a Fact-Checking System designed to classify claims into three categories: SUPPORTED, REFUTED, and NEI (Not Enough Information). It provides predictions based on a given context and claim, using a pre-trained multilingual BERT model.

## Overview

The system consists of three main components:

1. **API Server (`api.py`):** This component serves as the backend, providing an API endpoint for making predictions. It uses Flask for handling HTTP requests and MongoDB for storing prediction data.

2. **Streamlit Predictor (`predictor.py`):** This is the frontend component where users can input context and claims, and get predictions interactively. It utilizes Streamlit for creating a user-friendly interface.

3. **Utilities (`utilities.py`):** This module contains utility functions for preprocessing data, making predictions using the BERT model, and retrieving evidence from the context.

## Setup

### Prerequisites

- Python 3.x
- Flask
- pymongo
- requests
- Streamlit
- Transformers (from Hugging Face)

### Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/your-username/fact-checking-system.git
    ```

2. Install the required Python dependencies.

    ```bash
    cd <path/to/folder>fact-checking-system
    pip install -r requirements.txt
    ```

3. Download the pre-trained multilingual BERT model and place it in the `Model` directory. You can download the model from the Hugging Face model hub or use any other compatible model.

### Running the Application

1. Start the API server.

    ```bash
    python api.py
    ```

2. Run the Streamlit predictor.

    ```bash
    streamlit run predictor.py
    ```

3. Open your web browser and navigate to `http://localhost:8501` to access the Streamlit app.

## Usage

1. Input the context and claim in the respective fields provided by the Streamlit interface.
2. Click on the appropriate button to check the claim based on the specified label (NEI, REFUTED, or SUPPORTED).
3. View the prediction result, including the predicted label, probabilities, and evidence supporting the prediction.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project utilizes the Transformers library by Hugging Face for working with pre-trained BERT models.
- Special thanks to the creators and contributors of Flask, Streamlit, and MongoDB for providing the tools necessary for building this system.
