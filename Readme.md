# Stack Overflow Tag Predictor

This project is an implementation of an AI model that predicts tags for Stack Overflow questions. It uses machine learning techniques to analyze the text of a question and predict the most relevant tags.

## Project Structure

The project is structured into three main directories:

- [`predict`](https://github.com/MandalorianY/Poc_to_Prod/tree/main/predict "predict"): Contains the main application code for the prediction model. The entry point is [`app.py`](https://github.com/MandalorianY/Poc_to_Prod/tree/main/predict/app.py "predict\predict\app.py").
- [`preprocessing`](https://github.com/MandalorianY/Poc_to_Prod/tree/main/preprocessing "preprocessing"): Contains the code for preprocessing the data. This includes the `BaseTextCategorizationDataset` and `LocalTextCategorizationDataset` classes, as well as utility functions such as `_get_label_list`, `_get_num_train_batches`, `_get_num_train_samples`, `filter_tag_position`, `filter_tags_with_less_than_x_samples`, `get_label_to_index_map`, and `get_train_batch`.

- [`train`](https://github.com/MandalorianY/Poc_to_Prod/tree/main/train "train"): Contains the code for training the model. The `add_timestamp` function is used to timestamp the training process.

## How to Run

Install the required Python packages listed in [`requirements.txt`] using pip:

```sh
pip install -r requirements.txt
```


## Testing

Unit tests are located in the `tests` subdirectories of each main directory. You can run them using your preferred testing framework.

## Contributing

Contributions are welcome. Please submit a pull request with your changes.

