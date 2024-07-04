# Fact-Checking Data Collection Application

## Overview

This project is designed to facilitate the collection of data for adversarial fact-checking. The application allows users to input claims and contexts, and then uses machine learning models to predict whether a claim is supported, refuted, or if there is not enough information (NEI). 

## Features

- Predict the status of a claim as Supported, Refuted, or NEI.
- Display label probabilities in a visually intuitive format.
- Allow users to input claims and contexts through a web interface.
- Save and display annotated data.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

### Setting up the environment

1. Clone the repository:
    ```sh
    git clone https://github.com/FOX2920/Fact_Checking_System.git
    cd path/to/Fact_Checking_System-main
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```sh
      .\\venv\\Scripts\\activate
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

## Using the Application

### Step-by-Step Guide

1. **Upload CSV File**
   - Upload a CSV file containing the contexts through the sidebar. The CSV file must have the following columns: `Summary`, `ID`, `Title`, `URL`.

2. **Mission Tab**
   - Read the instructions on creating claims. This section provides guidelines on how to interpret the context and classify claims.

3. **Annotate Tab**
   - **Context and Details**
     - View the current context and its details (File name, ID, Title, and URL) at the top of the page.
   - **Creating Claims**
     - For each category (SUPPORTED, REFUTED, NEI), expand the section and enter your claim.
     - The application will check if a similar claim already exists to prevent duplicates.
     - After entering a claim, the application will predict its classification. If the prediction probability is low or the predicted label differs, you will be prompted to modify the claim or provide additional evidence.
     - Enter evidence from the context to support or refute the claim.
   - **Navigating Between Contexts**
     - Use the "Previous" and "Next" buttons to navigate between different contexts. Ensure you have entered at least three claims for each label before moving to the next context.

4. **Save Tab**
   - **Saving Data**
     - Ensure all claims and evidence are entered before saving. The "Save" button will save the annotated data.
   - **Viewing Saved Annotations**
     - The saved annotations can be viewed in this tab. If no data has been saved yet, an informative message will be displayed.

### Example of Using the Annotation Interface

1. **Upload CSV File**
   - Click on "Upload CSV file" in the sidebar and select your CSV file.

2. **Annotate Claims**
   - Under the "Annotate" tab, read the context provided.
   - Expand the "SUPPORTED" section and enter a claim that is supported by the context.
   - If prompted, enter evidence from the context.
   - Repeat for "REFUTED" and "NEI" sections.

3. **Save Annotations**
   - Once all claims and evidence are entered, click on the "Save" button.

4. **Navigate Between Contexts**
   - Use the "Previous" and "Next" buttons to switch between different contexts.

## Guidelines for Creating Claims

- **SUPPORTED**: Claims that are fully supported by the provided context.
- **REFUTED**: Claims that are clearly contradicted by the provided context.
- **NEI**: Claims that cannot be fully supported or refuted based on the provided context.

## Important Notes

- Ensure that each context has at least three claims for each label before moving to the next context.
- Avoid entering duplicate claims for the same context and label.
- Save your work frequently to prevent data loss.

## Troubleshooting

- **Missing Required Columns**: Ensure your CSV file includes `Summary`, `ID`, `Title`, and `URL` columns.
- **Prediction Issues**: If the prediction probability is low or the predicted label differs, modify the claim or provide additional evidence.

## Additional Resources

- [Detailed Annotation Guide](https://docs.google.com/document/d/121GHPAOFa4_fhmXDGJFYCrmsStcXYc7H/edit)
- [Download Contexts](https://drive.google.com/drive/folders/1bbW7qiglBZHvGs5oNF-s_eac09t5oWOW)


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

