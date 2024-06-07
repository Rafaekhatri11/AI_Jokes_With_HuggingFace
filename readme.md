# AI Joke Generator

This repository contains a Streamlit application that generates humorous jokes based on news headlines using a Hugging Face model. The application allows users to input a news article and receive a joke in response. Additionally, users can provide feedback, and the system saves the prompts and responses locally.

## Installation

**Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit application:**

    ```sh
    streamlit run streamlit.py
    ```

2. **Interact with the application:**

    - Enter the content of a news article in the provided text area.
    - Click the "Run" button to generate a joke.
    - Review the joke and provide feedback in the feedback text area.
    - Click the "Submit" button to save the prompt, response, and feedback locally.

## Fine-Tuning the Model

This repository also includes scripts for fine-tuning the model using PEFT (Parameter-Efficient Fine-Tuning).

1. **Load and preprocess the dataset:**

    The dataset is loaded from a local JSON file (`responses.json`) that contains prompts and their corresponding responses.

2. **Initialize the model and tokenizer:**

    The `google/flan-t5-base` model is used as the base model.

3. **Apply PEFT and train the model:**

    The model is fine-tuned using the PEFT configuration.

4. **Save the fine-tuned model:**

    The fine-tuned model is saved locally for further use.

## Folder Structure

- `streamlit.py`: Contains the Streamlit application code.
- `app.py`: Contains the core logic for generating jokes using the Hugging Face model.
- `responses.json`: Stores the user prompts and responses.
- `requirements.txt`: Lists the required Python packages.
- `README.md`: This file.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the pre-trained models and libraries.
- [Streamlit](https://streamlit.io/) for the interactive web application framework.

## Author

- Your Name - [@yourusername](https://github.com/yourusername)
