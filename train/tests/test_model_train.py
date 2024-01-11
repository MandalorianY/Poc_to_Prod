import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd
from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock() -> pd.DataFrame:
    """ Creates a mock dataset for testing purposes with titles and tags

    Returns:
        pd.DataFrame: a dataframe with titles and tags
    """
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


class TestTrain(unittest.TestCase):
    def test_train(self):
        # load the mocked dataset
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(
            return_value=load_dataset_mock())
        # define the training parameters for the model , we'll use training conf for the run
        params = {
            'batch_size': 1,
            'epochs': 4,
            'dense_dim': 64,
            'min_samples_per_label': 10,
            'verbose': 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train(
                "fake_path",
                train_conf=params,
                model_path=model_dir,
                add_timestamp=True
            )

        # assert that accuracy is equal to 1.0
        self.assertEqual(accuracy, 1.0)
