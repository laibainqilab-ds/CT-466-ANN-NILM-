"""
Tests for Stage 7 Step 5A six-experiment runner (CT-466 Phase 2).

Run with:  python -m unittest tests.test_stage7_run_experiments -v
(from the project root, using the project's .venv interpreter)

Deliberately never calls the real stage7_train.train_model (that would
launch real training) or the real stage7_train.build_datasets more than
is needed -- train_fn is injected as a stub throughout, and
build_datasets is patched for pure-orchestration tests. No real
checkpoint is ever written: a temporary checkpoint_dir is always used,
and the stub train_fn never calls torch.save.
"""
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage7_run_experiments as runner  # noqa: E402
import stage7_train as st  # noqa: E402


class _FakeDataset:
    """Identity-only stand-in -- these tests check object identity across
    all six experiment calls, not real dataset behavior (that is covered
    separately by TestBuildDatasetsExcludesTestSplit)."""
    pass


class _Sentinel:
    """Marker object embedded in each stub result; used to detect whether
    the runner retains a reference to a completed experiment's result
    after moving on to the next one."""
    pass


def _fake_build_datasets():
    return _FakeDataset(), _FakeDataset(), "FAKE_LABELS_TRAIN_DF", {"note": "fake"}


def _make_stub_train_fn(calls, sentinels):
    def stub_train_fn(config, train_ds=None, val_ds=None, labels_train_df=None,
                       preprocessing_reference=None, log_fn=None):
        checkpoint_path = str(config.checkpoint_dir / f"{config.model_name}_{config.class_weight_mode}_best.pt")
        calls.append({
            "model_name": config.model_name,
            "class_weight_mode": config.class_weight_mode,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "labels_train_df": labels_train_df,
            "preprocessing_reference": preprocessing_reference,
        })
        sentinel = _Sentinel()
        sentinels.append(weakref.ref(sentinel))
        return {
            "best_val_loss": 1.23,
            "best_epoch": 0,
            "epochs_run": 1,
            "checkpoint_path": checkpoint_path,
            "history": [{
                "epoch": 0,
                "train": {"cls_loss": 0.9, "reg_loss": 0.8, "total_loss": 1.7, "n_batches": 3},
                "val": {"cls_loss": 0.5, "reg_loss": 0.73, "total_loss": 1.23, "n_batches": 2},
                "duration_s": 0.01,
                "memory_mb": 123.4,
            }],
            "_sentinel": sentinel,
        }
    return stub_train_fn


def _make_mixed_stub_train_fn(calls, sentinels, fail_on):
    """Like _make_stub_train_fn, but raises RuntimeError for any
    (model_name, class_weight_mode) combination present in `fail_on`
    (a set of 2-tuples) instead of returning a result."""
    def stub_train_fn(config, train_ds=None, val_ds=None, labels_train_df=None,
                       preprocessing_reference=None, log_fn=None):
        calls.append((config.model_name, config.class_weight_mode))
        if (config.model_name, config.class_weight_mode) in fail_on:
            raise RuntimeError(f"synthetic failure for {config.model_name}/{config.class_weight_mode}")
        checkpoint_path = str(config.checkpoint_dir / f"{config.model_name}_{config.class_weight_mode}_best.pt")
        sentinel = _Sentinel()
        sentinels.append(weakref.ref(sentinel))
        return {
            "best_val_loss": 1.23,
            "best_epoch": 0,
            "epochs_run": 1,
            "checkpoint_path": checkpoint_path,
            "history": [{
                "epoch": 0,
                "train": {"cls_loss": 0.9, "reg_loss": 0.8, "total_loss": 1.7, "n_batches": 3},
                "val": {"cls_loss": 0.5, "reg_loss": 0.73, "total_loss": 1.23, "n_batches": 2},
                "duration_s": 0.01,
                "memory_mb": 123.4,
            }],
            "_sentinel": sentinel,
        }
    return stub_train_fn


class TestBuildExperimentConfigs(unittest.TestCase):
    def test_six_unique_model_mode_combinations(self):
        configs = runner.build_experiment_configs()
        combos = [(c.model_name, c.class_weight_mode) for c in configs]
        self.assertEqual(len(combos), 6)
        self.assertEqual(len(set(combos)), 6)
        self.assertEqual(set(combos), {
            ("MLP", "unweighted"), ("MLP", "cap20"), ("MLP", "cap50"),
            ("CNN1D", "unweighted"), ("CNN1D", "cap20"), ("CNN1D", "cap50"),
        })

    def test_six_unique_checkpoint_filenames(self):
        configs = runner.build_experiment_configs()
        filenames = [f"{c.model_name}_{c.class_weight_mode}_best.pt" for c in configs]
        self.assertEqual(len(filenames), len(set(filenames)))

    def test_batch_size_and_num_workers_explicit(self):
        for c in runner.build_experiment_configs():
            self.assertEqual(c.batch_size, 256)
            self.assertEqual(c.num_workers, 0)

    def test_checkpoint_dir_override_applied_to_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = runner.build_experiment_configs(checkpoint_dir=Path(tmpdir))
            for c in configs:
                self.assertEqual(c.checkpoint_dir, Path(tmpdir))

    def test_checkpoint_dir_defaults_to_root_level_models_when_not_overridden(self):
        for c in runner.build_experiment_configs():
            self.assertEqual(c.checkpoint_dir, st.DEFAULT_CHECKPOINT_DIR)
            self.assertNotIn("src", c.checkpoint_dir.parts)


class TestBestEpochValMetricsExtraction(unittest.TestCase):
    def test_normal_case_extracts_correct_values(self):
        result = {"best_epoch": 1, "history": [
            {"val": {"cls_loss": 9.0, "reg_loss": 9.0}},
            {"val": {"cls_loss": 0.3, "reg_loss": 0.4}},
        ]}
        cls_loss, reg_loss = runner._best_epoch_val_metrics(result)
        self.assertEqual(cls_loss, 0.3)
        self.assertEqual(reg_loss, 0.4)

    def test_negative_best_epoch_returns_none(self):
        cls_loss, reg_loss = runner._best_epoch_val_metrics({"best_epoch": -1, "history": []})
        self.assertIsNone(cls_loss)
        self.assertIsNone(reg_loss)

    def test_out_of_range_best_epoch_returns_none(self):
        result = {"best_epoch": 5, "history": [{"val": {"cls_loss": 1.0, "reg_loss": 1.0}}]}
        cls_loss, reg_loss = runner._best_epoch_val_metrics(result)
        self.assertIsNone(cls_loss)
        self.assertIsNone(reg_loss)


class TestRunAllExperimentsDatasetReuse(unittest.TestCase):
    def test_build_datasets_called_exactly_once_and_reused_across_all_six(self):
        calls, sentinels = [], []
        stub_train_fn = _make_stub_train_fn(calls, sentinels)

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()) as mocked:
            with tempfile.TemporaryDirectory() as tmpdir:
                runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=Path(tmpdir) / "summary.json", checkpoint_dir=Path(tmpdir),
                )
            self.assertEqual(mocked.call_count, 1)

        self.assertEqual(len(calls), 6)
        first = calls[0]
        for call in calls[1:]:
            self.assertIs(call["train_ds"], first["train_ds"])
            self.assertIs(call["val_ds"], first["val_ds"])
            self.assertIs(call["labels_train_df"], first["labels_train_df"])
            self.assertIs(call["preprocessing_reference"], first["preprocessing_reference"])


class TestRunAllExperimentsSequentialCleanup(unittest.TestCase):
    def test_per_experiment_result_sentinel_is_collected_after_run_completes(self):
        calls, sentinels = [], []
        stub_train_fn = _make_stub_train_fn(calls, sentinels)

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=Path(tmpdir) / "summary.json", checkpoint_dir=Path(tmpdir),
                )

        self.assertEqual(len(sentinels), 6)
        for ref in sentinels:
            self.assertIsNone(
                ref(), "a per-experiment result sentinel was still reachable after "
                       "run_all_experiments completed -- runner is retaining unnecessary references"
            )


class TestRunAllExperimentsSummaryOutput(unittest.TestCase):
    def test_summary_json_matches_stub_results_and_is_written_to_disk(self):
        calls, sentinels = [], []
        stub_train_fn = _make_stub_train_fn(calls, sentinels)

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                summary_path = Path(tmpdir) / "summary.json"
                returned_summary = runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=summary_path, checkpoint_dir=Path(tmpdir),
                )
                self.assertTrue(summary_path.exists())
                with open(summary_path) as f:
                    on_disk = json.load(f)

        self.assertEqual(on_disk, {"experiments": returned_summary})
        self.assertEqual(len(returned_summary), 6)

        expected_combos = {(e["model_name"], e["class_weight_mode"]) for e in runner.EXPERIMENTS}
        actual_combos = {(e["model_name"], e["class_weight_mode"]) for e in returned_summary}
        self.assertEqual(actual_combos, expected_combos)

        for entry in returned_summary:
            self.assertEqual(entry["status"], "completed")
            self.assertIsNone(entry["error_type"])
            self.assertIsNone(entry["error_message"])
            self.assertEqual(entry["best_epoch"], 0)
            self.assertAlmostEqual(entry["best_val_cls_loss"], 0.5)
            self.assertAlmostEqual(entry["best_val_reg_loss"], 0.73)
            self.assertAlmostEqual(entry["best_val_total_loss"], 1.23)
            self.assertEqual(entry["epochs_run"], 1)
            self.assertIn(f"{entry['model_name']}_{entry['class_weight_mode']}_best.pt", entry["checkpoint_path"])

        checkpoint_paths = [e["checkpoint_path"] for e in returned_summary]
        self.assertEqual(len(checkpoint_paths), len(set(checkpoint_paths)))


class TestAllExperimentsSucceeding(unittest.TestCase):
    """Step 5C requirement: when every experiment succeeds, all six
    entries must carry status='completed' with real recorded values."""

    def test_all_six_entries_marked_completed(self):
        calls, sentinels = [], []
        stub_train_fn = _make_stub_train_fn(calls, sentinels)

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                summary = runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=Path(tmpdir) / "summary.json", checkpoint_dir=Path(tmpdir),
                )

        self.assertEqual(len(summary), 6)
        self.assertTrue(all(e["status"] == "completed" for e in summary))
        self.assertTrue(all(e["error_type"] is None for e in summary))
        self.assertTrue(all(e["error_message"] is None for e in summary))


class TestOneExperimentFailingContinues(unittest.TestCase):
    """Step 5C requirement: one experiment raising must not stop the
    remaining experiments from running."""

    def test_failure_on_second_experiment_does_not_block_remaining_four(self):
        calls, sentinels = [], []
        # EXPERIMENTS[1] == ("MLP", "cap20") -- fail the second configuration.
        stub_train_fn = _make_mixed_stub_train_fn(calls, sentinels, fail_on={("MLP", "cap20")})

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                summary = runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=Path(tmpdir) / "summary.json", checkpoint_dir=Path(tmpdir),
                )

        # All six configurations were actually attempted, in order.
        self.assertEqual(calls, [
            ("MLP", "unweighted"), ("MLP", "cap20"), ("MLP", "cap50"),
            ("CNN1D", "unweighted"), ("CNN1D", "cap20"), ("CNN1D", "cap50"),
        ])
        self.assertEqual(len(summary), 6)

    def test_failure_in_the_middle_still_yields_five_completed_and_one_failed(self):
        calls, sentinels = [], []
        stub_train_fn = _make_mixed_stub_train_fn(calls, sentinels, fail_on={("CNN1D", "unweighted")})

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                summary = runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=Path(tmpdir) / "summary.json", checkpoint_dir=Path(tmpdir),
                )

        statuses = [e["status"] for e in summary]
        self.assertEqual(statuses.count("completed"), 5)
        self.assertEqual(statuses.count("failed"), 1)


class TestFailedExperimentStatusAndNoFabrication(unittest.TestCase):
    """Step 5C requirement: a failed entry must carry status='failed',
    a useful exception type/message, and MUST NOT contain fabricated
    metrics or a fabricated checkpoint path."""

    def setUp(self):
        self.calls, self.sentinels = [], []
        self.stub_train_fn = _make_mixed_stub_train_fn(self.calls, self.sentinels, fail_on={("CNN1D", "cap50")})

    def _run(self):
        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                return runner.run_all_experiments(
                    train_fn=self.stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=Path(tmpdir) / "summary.json", checkpoint_dir=Path(tmpdir),
                )

    def test_failed_entry_has_failed_status_and_exception_details(self):
        summary = self._run()
        failed = [e for e in summary if e["model_name"] == "CNN1D" and e["class_weight_mode"] == "cap50"]
        self.assertEqual(len(failed), 1)
        entry = failed[0]
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["error_type"], "RuntimeError")
        self.assertIn("CNN1D/cap50", entry["error_message"])

    def test_failed_entry_has_no_fabricated_metrics_or_checkpoint_path(self):
        summary = self._run()
        failed = [e for e in summary if e["model_name"] == "CNN1D" and e["class_weight_mode"] == "cap50"][0]
        self.assertIsNone(failed["best_epoch"])
        self.assertIsNone(failed["best_val_cls_loss"])
        self.assertIsNone(failed["best_val_reg_loss"])
        self.assertIsNone(failed["best_val_total_loss"])
        self.assertIsNone(failed["checkpoint_path"])
        self.assertIsNone(failed["epochs_run"])

    def test_successful_entries_remain_unaffected_by_the_one_failure(self):
        summary = self._run()
        succeeded = [e for e in summary if not (e["model_name"] == "CNN1D" and e["class_weight_mode"] == "cap50")]
        self.assertEqual(len(succeeded), 5)
        for entry in succeeded:
            self.assertEqual(entry["status"], "completed")
            self.assertIsNone(entry["error_type"])
            self.assertIsNone(entry["error_message"])
            self.assertIsNotNone(entry["checkpoint_path"])
            self.assertEqual(entry["best_epoch"], 0)


class TestSummaryPersistedIncrementally(unittest.TestCase):
    """Step 5C requirement: the summary file must be written to disk
    after EVERY experiment, not only once at the end."""

    def test_summary_file_already_has_n_entries_before_the_nth_plus_one_call(self):
        observed_counts_before_call = []

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"

            def stub_train_fn(config, train_ds=None, val_ds=None, labels_train_df=None,
                               preprocessing_reference=None, log_fn=None):
                if summary_path.exists():
                    with open(summary_path) as f:
                        observed_counts_before_call.append(len(json.load(f)["experiments"]))
                else:
                    observed_counts_before_call.append(0)
                checkpoint_path = str(config.checkpoint_dir / f"{config.model_name}_{config.class_weight_mode}_best.pt")
                return {
                    "best_val_loss": 1.23, "best_epoch": 0, "epochs_run": 1,
                    "checkpoint_path": checkpoint_path,
                    "history": [{
                        "epoch": 0, "train": {}, "duration_s": 0.0, "memory_mb": None,
                        "val": {"cls_loss": 0.5, "reg_loss": 0.73, "total_loss": 1.23, "n_batches": 2},
                    }],
                }

            with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
                runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=summary_path, checkpoint_dir=Path(tmpdir),
                )

        self.assertEqual(observed_counts_before_call, [0, 1, 2, 3, 4, 5])

    def test_file_contains_all_six_entries_after_completion(self):
        calls, sentinels = [], []
        stub_train_fn = _make_stub_train_fn(calls, sentinels)

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                summary_path = Path(tmpdir) / "summary.json"
                runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=summary_path, checkpoint_dir=Path(tmpdir),
                )
                with open(summary_path) as f:
                    on_disk = json.load(f)

        self.assertEqual(len(on_disk["experiments"]), 6)


class TestPreviousResultsSurviveLaterFailure(unittest.TestCase):
    """Step 5C requirement: results from previously completed experiments
    must remain in the on-disk summary after a later experiment fails."""

    def test_earlier_completed_results_remain_on_disk_after_a_later_failure(self):
        calls, sentinels = [], []
        # Fail the LAST configuration -- the five before it must already
        # be persisted to disk by the time this one raises.
        stub_train_fn = _make_mixed_stub_train_fn(calls, sentinels, fail_on={("CNN1D", "cap50")})

        with mock.patch.object(st, "build_datasets", return_value=_fake_build_datasets()):
            with tempfile.TemporaryDirectory() as tmpdir:
                summary_path = Path(tmpdir) / "summary.json"
                returned_summary = runner.run_all_experiments(
                    train_fn=stub_train_fn, log_fn=lambda *a, **k: None,
                    summary_path=summary_path, checkpoint_dir=Path(tmpdir),
                )
                with open(summary_path) as f:
                    on_disk = json.load(f)

        self.assertEqual(on_disk, {"experiments": returned_summary})

        completed_on_disk = [e for e in on_disk["experiments"] if e["status"] == "completed"]
        failed_on_disk = [e for e in on_disk["experiments"] if e["status"] == "failed"]
        self.assertEqual(len(completed_on_disk), 5)
        self.assertEqual(len(failed_on_disk), 1)

        expected_completed_combos = {
            ("MLP", "unweighted"), ("MLP", "cap20"), ("MLP", "cap50"), ("CNN1D", "unweighted"), ("CNN1D", "cap20"),
        }
        actual_completed_combos = {(e["model_name"], e["class_weight_mode"]) for e in completed_on_disk}
        self.assertEqual(actual_completed_combos, expected_completed_combos)
        for entry in completed_on_disk:
            self.assertIsNotNone(entry["checkpoint_path"])
            self.assertIsNotNone(entry["best_val_total_loss"])


class TestDefaultSummaryPathLocation(unittest.TestCase):
    def test_default_summary_path_is_under_data_meta(self):
        self.assertEqual(
            runner.DEFAULT_SUMMARY_PATH,
            runner.PROJECT_ROOT / "data" / "meta" / "stage7_experiment_summary.json",
        )


class TestRunnerNeverReferencesTestSplit(unittest.TestCase):
    def test_source_never_references_test_split_label_path(self):
        source = Path(runner.__file__).read_text()
        self.assertNotIn('LABELS_PATHS["test"]', source)
        self.assertNotIn("LABELS_PATHS['test']", source)
        self.assertNotIn('"test"', source)
        self.assertNotIn("'test'", source)


if __name__ == "__main__":
    unittest.main()
