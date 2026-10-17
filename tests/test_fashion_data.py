"""Fashion-MNIST handling: IDX parsing, verification, the validation split, no network at import/load."""
from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request

import numpy as np
import pytest

from shared import fashion


def _write_idx(path, arr: np.ndarray):
    if arr.ndim == 3:
        header = struct.pack(">IIII", 2051, *arr.shape)
    else:
        header = struct.pack(">II", 2049, arr.shape[0])
    with gzip.open(path, "wb") as fh:
        fh.write(header + arr.tobytes())


def test_class_names_are_the_ten_official_labels_in_order():
    assert fashion.CLASS_NAMES == ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    assert fashion.N_CLASSES == 10 and fashion.IMG_SIZE == 28


def test_idx_reader_roundtrips_images_and_labels(tmp_path):
    imgs = np.random.default_rng(0).integers(0, 255, (5, 28, 28), dtype=np.uint8)
    labels = np.array([0, 3, 9, 1, 2], dtype=np.uint8)
    _write_idx(tmp_path / "i.gz", imgs)
    _write_idx(tmp_path / "l.gz", labels)
    assert np.array_equal(fashion._read_idx(tmp_path / "i.gz"), imgs)
    assert np.array_equal(fashion._read_idx(tmp_path / "l.gz"), labels)


def test_missing_files_give_an_actionable_error_and_never_download(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("fashion.load must never touch the network")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(fashion.DatasetUnavailable, match="fetch_fashion_mnist"):
        fashion.load(tmp_path)


def test_a_corrupted_file_fails_the_published_md5(tmp_path):
    for name in fashion.FILES:
        (tmp_path / name).write_bytes(b"not the real file")
    with pytest.raises(fashion.DatasetUnavailable, match="MD5"):
        fashion.load(tmp_path)


def test_include_test_false_never_opens_the_test_files(tmp_path):
    # Only the training files exist; skipping verification isolates the behaviour under test.
    _write_idx(tmp_path / "train-images-idx3-ubyte.gz",
               np.zeros((fashion.N_TRAIN_OFFICIAL, 28, 28), dtype=np.uint8))
    _write_idx(tmp_path / "train-labels-idx1-ubyte.gz", np.zeros(fashion.N_TRAIN_OFFICIAL, dtype=np.uint8))
    data = fashion.load(tmp_path, include_test=False, verify=False)
    assert data.x_test is None and data.y_test is None and len(data.y_train) == 60_000
    assert set(data.file_sha256) == {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"}


def test_stratified_split_is_reproducible_disjoint_and_balanced():
    y = np.repeat(np.arange(10), 600).astype(np.uint8)
    tr, va = fashion.split_train_val(y, 1000, split_seed=3)
    tr2, va2 = fashion.split_train_val(y, 1000, split_seed=3)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
    assert not set(tr) & set(va) and len(tr) + len(va) == len(y)
    assert np.bincount(y[va], minlength=10).tolist() == [100] * 10
    assert (np.diff(tr) > 0).all() and (np.diff(va) > 0).all()


def test_split_seed_changes_validation_images_only_by_seed():
    y = np.repeat(np.arange(10), 600).astype(np.uint8)
    _, a = fashion.split_train_val(y, 1000, 1)
    _, b = fashion.split_train_val(y, 1000, 2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("val", [0, 6000])
def test_degenerate_validation_sizes_are_rejected(val):
    with pytest.raises(ValueError):
        fashion.split_train_val(np.repeat(np.arange(10), 600).astype(np.uint8), val if val < 6000 else 6000, 0)


def test_identity_records_source_files_split_and_normalization():
    from tests.fashion_fixture import tiny_fashion

    data = tiny_fashion()
    tr, va = fashion.split_train_val(data.y_train, 50, 4)
    ident = fashion.dataset_identity(data, tr, va, 4, 50)
    assert ident["name"] == "fashion-mnist" and ident["class_names"] == fashion.CLASS_NAMES
    assert ident["split_seed"] == 4 and ident["n_train"] == len(tr) and ident["n_val"] == 50
    assert ident["train_index_sha256"] == fashion.index_sha256(tr)
    assert ident["normalization"]["mean"] == fashion.NORM_MEAN
    assert ident["official_train_images"] == len(data.y_train)


REAL = fashion.data_dir()
@pytest.mark.skipif(not (REAL / "train-images-idx3-ubyte.gz").is_file(),
                    reason="official Fashion-MNIST files not downloaded (python tools/fetch_fashion_mnist.py)")
class TestRealFiles:
    """Checks against the downloaded official files; skipped on machines without them."""

    def test_official_shapes_hashes_and_normalization_constants(self):
        d = fashion.load()
        assert d.x_train.shape == (60_000, 28, 28) and d.x_test.shape == (10_000, 28, 28)
        assert np.bincount(d.y_train).tolist() == [6000] * 10
        assert np.bincount(d.y_test).tolist() == [1000] * 10
        assert d.file_sha256["train-images-idx3-ubyte.gz"] == \
            "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"
        mean, std = fashion.measure_norm_stats(d.x_train)
        assert abs(mean - fashion.NORM_MEAN) < 5e-4 and abs(std - fashion.NORM_STD) < 5e-4

    def test_default_validation_split_is_6000_balanced_images_from_train_only(self):
        d = fashion.load(include_test=False)
        tr, va = fashion.split_train_val(d.y_train, 6000, 0)
        assert (len(tr), len(va)) == (54_000, 6_000)
        assert np.bincount(d.y_train[va]).tolist() == [600] * 10
        assert hashlib.sha256(np.sort(va).astype(np.int64).tobytes()).hexdigest() == fashion.index_sha256(va)
