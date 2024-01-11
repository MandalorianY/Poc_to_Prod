import unittest
import pandas as pd
from unittest.mock import MagicMock
from unittest.mock import patch
from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label1', 'label2'])
        expected_map = {0: 'label1', 1: 'label2'}
        self.assertEqual(base.get_index_to_label_map(), expected_map)

    def test_index_to_label_and_label_to_index_are_identity(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base.label_to_index = {'label1': 0, 'label2': 1}
        base.index_to_label = {0: 'label1', 1: 'label2'}
        labels = ['label1', 'label2']
        indexes = [base.label_to_index[label] for label in labels]
        converted_labels = [base.index_to_label[index] for index in indexes]
        self.assertEqual(labels, converted_labels)

    def test_to_indexes(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label1', 'label2'])
        labels = ['label1', 'label2']
        expected_indexes = [0, 1]
        self.assertEqual(list(base.to_indexes(labels)), expected_indexes)


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset_mock = utils.LocalTextCategorizationDataset.load_dataset(
            "fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })

        print(dataset_mock)
        pd.testing.assert_frame_equal(dataset_mock, expected)


def test__get_num_samples_is_correct(self):
    with patch('pandas.read_csv', return_value=pd.DataFrame({
        'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
        'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b', 'tag_c', 'tag_c'],
        'tag_id': [1, 2, 3, 4, 5, 6],
        'tag_position': [0, 0, 0, 0, 0, 0],
        'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
    })):
        dataset_mock = utils.LocalTextCategorizationDataset(
            "fake_path", 2, 0.6, 1)
        self.assertEqual(dataset_mock._get_num_samples(), 6)

    def test_get_train_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b', 'tag_c', 'tag_c'],
            'tag_id': [1, 2, 3, 4, 5, 6],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        dataset_mock = utils.LocalTextCategorizationDataset(
            "fake_path", 2, 0.6, 1)

        batch_size = 2
        dataset_mock.batch_size = batch_size
        dataset_mock.train_batch_index = 0
        x, y = dataset_mock.get_train_batch()
        self.assertEqual(x.shape, (batch_size,))
        self.assertEqual(y.shape, (batch_size, 2))


def test_get_test_batch_returns_expected_shape(self):
    with patch('pandas.read_csv', return_value=pd.DataFrame({
        'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
        'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b', 'tag_c', 'tag_c'],
        'tag_id': [1, 2, 3, 4, 5, 6],
        'tag_position': [0, 0, 0, 0, 0, 0],
        'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
    })):
        dataset_mock = utils.LocalTextCategorizationDataset(
            "fake_path", 2, 0.6, 1)

        batch_size = 2
        dataset_mock.batch_size = batch_size
        dataset_mock.test_batch_index = 0
        x, y = dataset_mock.get_test_batch()
        self.assertEqual(x.shape, (batch_size,))

    def test_get_train_batch_raises_assertion_error(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b', 'tag_c', 'tag_c'],
            'tag_id': [1, 2, 3, 4, 5, 6],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))

        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset(
                "fake_path", 20, 0.7, 1)