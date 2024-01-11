import unittest
from unittest.mock import MagicMock
import pandas as pd
from train.train import run
from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils
import tempfile


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestPredict(unittest.TestCase):
    def test_predict(self):
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(
            return_value=load_dataset_mock())

        params = {
            'batch_size': 1,
            'epochs': 4,
            'dense_dim': 64,
            'min_samples_per_label': 10,
            'verbose': 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            _, path = run.train(
                "fake_path",
                train_conf=params,
                model_path=model_dir,
                add_timestamp=False
            )
            model = TextPredictionModel.from_artefacts(path)

            predictions = model.predict(
                "Is it possible to execute the procedure of a function in the scope of the caller ?",
                top_k=1
            )
            self.assertEqual(predictions, ['php'])
