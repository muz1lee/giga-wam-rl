import io
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from giga_wam_rl import workspace
except ModuleNotFoundError:
    workspace = None
    UnsafeWorkspacePath = ValueError
    validate_output_root = None
    load_registry = None
    inspect_assets = None
    run_check = None
    main = None
else:
    UnsafeWorkspacePath = workspace.UnsafeWorkspacePath
    validate_output_root = workspace.validate_output_root
    load_registry = getattr(workspace, "load_registry", None)
    inspect_assets = getattr(workspace, "inspect_assets", None)
    run_check = getattr(workspace, "run_check", None)
    main = getattr(workspace, "main", None)


class ValidateOutputRootTests(unittest.TestCase):
    def test_accepts_path_nested_under_artifact_root(self) -> None:
        self.assertIsNotNone(validate_output_root)

        result = validate_output_root(
            Path("/mnt/nas/wenqian/giga-wam-rl/runs/experiment-1"),
            artifact_root=Path("/mnt/nas/wenqian/giga-wam-rl"),
            protected_roots=(Path("/home/wjh"), Path("/mnt/data/wjh")),
        )

        self.assertEqual(
            result, Path("/mnt/nas/wenqian/giga-wam-rl/runs/experiment-1")
        )

    def test_rejects_artifact_root_nested_under_protected_root(self) -> None:
        with self.assertRaises(UnsafeWorkspacePath):
            validate_output_root(
                Path("/home/wjh/our-runs/experiment-1"),
                artifact_root=Path("/home/wjh/our-runs"),
                protected_roots=(Path("/home/wjh"), Path("/mnt/data/wjh")),
            )

    def test_rejects_protected_root_reached_through_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            protected_root = temporary_root / "student-data"
            protected_root.mkdir()
            alias = temporary_root / "apparent-output"
            alias.symlink_to(protected_root, target_is_directory=True)

            with self.assertRaises(UnsafeWorkspacePath):
                validate_output_root(
                    alias / "experiment-1",
                    artifact_root=alias,
                    protected_roots=(protected_root,),
                )


class InspectAssetsTests(unittest.TestCase):
    def test_missing_asset_is_reported_without_being_created(self) -> None:
        self.assertIsNotNone(load_registry)
        self.assertIsNotNone(inspect_assets)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            missing_asset = temporary_root / "student" / "missing-dataset"
            registry_path = temporary_root / "assets.toml"
            registry_path.write_text(
                "\n".join(
                    (
                        "[workspace]",
                        'artifact_root = "/mnt/nas/wenqian/giga-wam-rl"',
                        'protected_roots = ["/home/wjh", "/mnt/data/wjh"]',
                        "",
                        "[[assets]]",
                        'name = "missing_demo"',
                        'kind = "dataset"',
                        f'path = "{missing_asset}"',
                        'owner = "student"',
                        "read_only = true",
                    )
                ),
                encoding="utf-8",
            )

            registry = load_registry(registry_path)
            statuses = inspect_assets(registry)

            self.assertEqual(len(statuses), 1)
            self.assertEqual(statuses[0].name, "missing_demo")
            self.assertFalse(statuses[0].exists)
            self.assertFalse(missing_asset.exists())


class ProjectRegistryTests(unittest.TestCase):
    def test_server_registry_uses_our_nas_root_and_read_only_student_assets(
        self,
    ) -> None:
        registry_path = PROJECT_ROOT / "configs" / "assets.server.toml"
        self.assertTrue(registry_path.is_file())

        registry = load_registry(registry_path)

        self.assertEqual(
            registry["workspace"]["artifact_root"],
            "/mnt/nas/wenqian/giga-wam-rl",
        )
        self.assertIn("/home/wjh", registry["workspace"]["protected_roots"])
        self.assertIn("/mnt/data/wjh", registry["workspace"]["protected_roots"])
        student_assets = [
            asset for asset in registry["assets"] if asset["owner"] == "student"
        ]
        self.assertGreater(len(student_assets), 0)
        self.assertTrue(all(asset["read_only"] is True for asset in student_assets))

    def test_gwp05_upstreams_are_pinned(self) -> None:
        upstream_path = PROJECT_ROOT / "configs" / "upstreams.toml"
        self.assertTrue(upstream_path.is_file())

        with upstream_path.open("rb") as upstream_file:
            upstream = tomllib.load(upstream_file)

        self.assertEqual(
            upstream["code"]["giga_world_policy_0_5"]["revision"],
            "5d55073a6508de7354c83679d9028f4010ff6cb2",
        )
        self.assertEqual(
            upstream["models"]["giga_world_policy_0_5_transformer"]["revision"],
            "4b68e90c0833fec96df456426be344bab64e01a3",
        )
        self.assertEqual(
            upstream["models"]["wan_2_2_ti2v_5b_diffusers"]["revision"],
            "b8fff7315c768468a5333511427288870b2e9635",
        )


class RunCheckTests(unittest.TestCase):
    def test_safe_registry_returns_success(self) -> None:
        self.assertIsNotNone(run_check)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            artifact_root = temporary_root / "our-artifacts"
            protected_root = temporary_root / "student"
            registry_path = temporary_root / "assets.toml"
            registry_path.write_text(
                "\n".join(
                    (
                        "[workspace]",
                        f'artifact_root = "{artifact_root}"',
                        f'protected_roots = ["{protected_root}"]',
                        "",
                        "[[assets]]",
                        'name = "demo"',
                        f'path = "{protected_root / "demo"}"',
                        "read_only = true",
                    )
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            exit_code = run_check(registry_path, output=output)

            self.assertEqual(exit_code, 0)
            self.assertIn("workspace status=safe", output.getvalue())
            self.assertFalse(artifact_root.exists())

    def test_unsafe_registry_returns_error(self) -> None:
        self.assertIsNotNone(run_check)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            protected_root = temporary_root / "student"
            artifact_root = protected_root / "our-artifacts"
            registry_path = temporary_root / "assets.toml"
            registry_path.write_text(
                "\n".join(
                    (
                        "[workspace]",
                        f'artifact_root = "{artifact_root}"',
                        f'protected_roots = ["{protected_root}"]',
                    )
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            exit_code = run_check(registry_path, output=output)

            self.assertEqual(exit_code, 2)
            self.assertIn("workspace status=unsafe", output.getvalue())
            self.assertFalse(artifact_root.exists())

    def test_student_asset_must_be_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            registry_path = temporary_root / "assets.toml"
            registry_path.write_text(
                "\n".join(
                    (
                        "[workspace]",
                        f'artifact_root = "{temporary_root / "our-artifacts"}"',
                        f'protected_roots = ["{temporary_root / "student"}"]',
                        "",
                        "[[assets]]",
                        'name = "student_demo"',
                        'path = "/mnt/data/wjh/demo"',
                        'owner = "student"',
                        "read_only = false",
                    )
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            exit_code = run_check(registry_path, output=output)

            self.assertEqual(exit_code, 2)
            self.assertIn("workspace status=invalid", output.getvalue())

    def test_main_runs_check_subcommand(self) -> None:
        self.assertIsNotNone(main)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            registry_path = temporary_root / "assets.toml"
            registry_path.write_text(
                "\n".join(
                    (
                        "[workspace]",
                        f'artifact_root = "{temporary_root / "our-artifacts"}"',
                        f'protected_roots = ["{temporary_root / "student"}"]',
                    )
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            exit_code = main(
                ["check", "--config", str(registry_path)], output=output
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("workspace status=safe", output.getvalue())


if __name__ == "__main__":
    unittest.main()
