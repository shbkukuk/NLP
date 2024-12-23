# NLP Homework Projects

This repository contains three homework projects for a Natural Language Processing (NLP) course. Each homework is organized in its own directory and includes Jupyter notebooks and Python scripts.

## Project Structure

```
.gitignore
HW1/
    hw1.ipynb
HW2/
    __pycache__/
    helper.py
    inputs/
        test.txt
        train.txt
    main.py
    ngrams.py
    outputs/
        testperplexity_1.txt
        testperplexity_2.txt
        testperplexity_3.txt
        testperplexity.txt
HW3/
    hw3.ipynb
    wiki.tr.txt
```

## HW1

- **File:** `HW1/hw1.ipynb`
- **Description:** This Jupyter notebook contains the first homework assignment. The notebook is currently empty and needs to be filled with the relevant code and analysis.

## HW2

- **Files:**
  - `HW2/helper.py`
  - `HW2/main.py`
  - `HW2/ngrams.py`
  - `HW2/inputs/`
  - `HW2/outputs/`
- **Description:** This homework focuses on n-gram language models. The main script (`main.py`) runs the preprocessing and n-gram calculations.

### Key Components

- **Preprocessing:**
  - Implemented in [`HW2/helper.py`](HW2/helper.py) within the `Preprocessing` class.
  - Reads and processes text data, splits it into training and test sets, and removes stopwords.
  - Example usage:
    ```python
    preprocessing = Preprocessing(dump='HW2/archive/wiki_00')
    preprocessing.creata_train_test()
    ```

- **N-grams:**
  - Implemented in [`HW2/ngrams.py`](HW2/ngrams.py) within the `Ngrams` class.
  - Calculates unigrams, bigrams, and trigrams, applies Good-Turing smoothing, and computes test perplexity.
  - Example usage:
    ```python
    unigrams = Ngrams(1)
    unigrams.read_files()
    unigrams.calculate_Ngrams()
    unigrams.create_count_table()
    unigrams.gt_smoothing()
    unigrams.calculate_test_perplexity()
    unigrams.generate_random()
    ```

## HW3

- **Files:**
  - `HW3/hw3.ipynb`
  - `HW3/wiki.tr.txt`
- **Description:** This homework involves text classification using transformers. The notebook (`hw3.ipynb`) contains the code for preprocessing, training, and evaluating a text classification model using the Hugging Face Transformers library.

### Key Components

- **Data Preprocessing:**
  - Reads and preprocesses text data from `wiki.tr.txt`.
  - Tokenizes the text using a pre-trained tokenizer from the Hugging Face library.

- **Model Training:**
  - Defines and trains a text classification model using TensorFlow and the Hugging Face Transformers library.
  - Evaluates the model on a validation set and plots training/validation accuracy.

- **Example Usage:**
  ```python
  from transformers import AutoTokenizer, TFDistilBertForSequenceClassification, create_optimizer
  import tensorflow as tf

  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

  # Define optimizer and compile model
  optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
  model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=tf.metrics.BinaryAccuracy(threshold=0.5))

  # Train model
  history = model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=10, callbacks=callbacks)
  ```

## How to Run

1. **HW1:**
   - Open `HW1/hw1.ipynb` in Jupyter Notebook and run the cells.

2. **HW2:**
   - Run `HW2/main.py` to execute the preprocessing and n-gram calculations.

3. **HW3:**
   - Open `HW3/hw3.ipynb` in Jupyter Notebook and run the cells to preprocess data, train the model, and evaluate its performance.

## Requirements

- Python 3.7+
- Jupyter Notebook
- TensorFlow
- Hugging Face Transformers
- Other dependencies as specified in the notebooks and scripts

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the course instructors and TAs for their guidance and support.
- The Hugging Face team for providing the Transformers library.

Feel free to reach out if you have any questions or need further assistance. Happy coding!