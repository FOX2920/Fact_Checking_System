
# Project Title

Fact-Checking Application

## Overview

This project aims to provide a tool for fact-checking claims against a given context. The application leverages machine learning models to predict whether a claim is supported, refuted, or not enough information (NEI) is available based on the provided context.

## Features

- Predict the status of a claim as Supported, Refuted, or NEI.
- Display label probabilities in a visually intuitive format.
- Allow users to input claims and contexts through a web interface.
- Save and display annotated data.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

### Setting up the environment

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/fact-checking-app.git
    cd fact-checking-app
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```sh
      .\venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

5. Download the model checkpoint file from Google Drive and place it in the `Model` folder:
    1. Go to [Google Drive](https://drive.google.com/drive/u/0/folders/1zAjAad5J3obOJgioptqEcA-Ta9l5OTul?fbclid=IwAR0Qskn-DcTTrN_LhRd6uRs1LPwjhe5fjDWJuXEay9iuW07TKeijV3lDrJU).
    2. Download the file `Checkpoint.pt`.
    3. Create a `Model` folder in the project directory if it doesn't exist:
       ```sh
       mkdir Model
       ```
    4. Move the downloaded `Checkpoint.pt` file into the `Model` folder.

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run predictor.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to use the application.

## Files Description

- **utilities.py**: Contains utility functions for setting the seed, loading the model, and predicting the status of claims based on the context.
- **predictor.py**: Implements the Streamlit web application for user interaction.

## Functions

### utilities.py

- **set_seed(seed)**: Sets the seed for reproducibility.
- **predict(context, claim)**: Predicts the status of a claim against the given context.

### predictor.py

- **result_form(result, user_label)**: Displays the prediction results in a styled dataframe.
- **create_expander_with_check_button(label, title, context, predict_func)**: Creates an expander with a check button for user input.
- **predictor_app()**: Main function to run the Streamlit app.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face Transformers
- Streamlit
