"""CPU checks for the cluster package's launch and measurement contracts."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cluster import devices, main, worker_command
from cluster_data import load_workload, normalize
from cluster_load import arrivals, request, summarize


class ClusterTests(unittest.TestCase):
    def test_bench_reserves_disjoint_ports_and_rejects_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workload.json").write_text("{}", encoding="utf-8")
            output = root / "output"
            argv = [
                "cluster.py", "bench", "--gpus", "0,1,2,3", "--workload", str(root),
                "--output", str(output), "--split", "1", "--split", "3",
                "--rps", "80", "--repetitions", "2", "--base-port",
            ]
            wave = AsyncMock(side_effect=lambda *args: {
                "throughput_rps": 42, "client_dropped": 0,
            })
            aiohttp = SimpleNamespace(ClientError=OSError)
            with (
                patch("cluster.sys.platform", "linux"),
                patch.dict("cluster.sys.modules", {"aiohttp": aiohttp}),
                patch.dict("cluster.os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
                patch("cluster.shutil.which", return_value="vllm"),
                patch("cluster.importlib.metadata.version", return_value="0.22.0+cu129"),
                patch("cluster.load_workload", return_value=[]),
                patch("cluster.run_wave", wave),
                patch("cluster.signal.signal"),
                patch("cluster.sys.argv", [*argv, "65521"]),
                patch("sys.stderr", new_callable=StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("fit all 16 worker ports", stderr.getvalue())
                wave.assert_not_awaited()
                self.assertFalse(output.exists())
                with patch("cluster.sys.argv", [*argv, "65520"]), redirect_stdout(StringIO()):
                    main()

            self.assertEqual(
                [(call.args[1].base_port, call.args[3]) for call in wave.await_args_list],
                [(65520, 1), (65524, 3), (65528, 3), (65532, 1)],
            )

    def test_worker_roles_share_cache_and_preserve_gpu_allocation(self):
        args = SimpleNamespace(
            workload=Path("."), model="fixture", context=1024, max_images=5, e_memory=0.01,
            pd_memory=0.8, max_seqs=8, encoder_only=False, e_batch_tokens=8192,
            model_revision="fixture-revision",
        )
        self.assertEqual(devices("2,7", "1,2,7"), ["2", "7"])
        with self.assertRaises(ValueError):
            devices("0,7", "1,2,7")
        for producer, role in [(True, "ec_producer"), (False, "ec_consumer")]:
            launch = worker_command("vllm", args, 18080, Path("cache"), producer)
            config = json.loads(launch[launch.index("--ec-transfer-config") + 1])
            self.assertEqual(config["ec_role"], role)
            self.assertEqual(config["ec_connector_extra_config"]["shared_storage_path"], "cache")
            self.assertIn("--no-enable-prefix-caching", launch)

    def test_dataset_schema_and_local_path_boundary(self):
        question, images = normalize("mmmu", {
            "question": "<image 2> and <image 1> versus <image 2>?",
            "image_1": "first", "image_2": "second", "options": "['same', 'different']",
        })
        self.assertEqual(images, ["second", "first"])
        self.assertEqual(question.splitlines()[0], "[Image 1] and [Image 2] versus [Image 1]?")
        self.assertIn("B. different", question)
        for name in ("vqav2", "textvqa"):
            self.assertEqual(normalize(name, {"question": "Which?", "image": "one"})[1], ["one"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "workload.json"
            (root / "local.png").write_bytes(b"fixture")
            for image, allowed in [("local.png", True), ("../local.png", False)]:
                manifest.write_text(json.dumps({"requests": [{"text": "Q", "images": [image]}]}))
                if allowed:
                    self.assertEqual(load_workload(root)[0][1][0], root / image)
                else:
                    with self.assertRaises(ValueError):
                        load_workload(root)

    def test_window_counts_and_slo_denominator(self):
        rows = [
            {"scheduled_at": 0.5, "completed_at": 1.5, "status": "completed"},
            {"scheduled_at": 1.2, "completed_at": 1.7, "status": "completed",
             "ttft_ms": 30, "tpot_ms": 8},
            {"scheduled_at": 1.4, "status": "unfinished"},
            {"scheduled_at": 1.5, "status": "client_dropped"},
            {"scheduled_at": 1.6, "status": "failed"},
        ]
        result = summarize(rows, 1, 2, 50, 10)
        self.assertEqual(result["throughput_rps"], 2)
        self.assertEqual(result["offered"], 4)
        self.assertEqual(result["within_both_fraction"], 0.25)
        self.assertEqual(list(arrivals(30, 1, 17)), list(arrivals(30, 1, 17)))


class ReplayTests(unittest.TestCase):
    def test_forward_same_image_ids_and_require_complete_stream(self):
        class Response:
            def __init__(self, lines=()):
                self.lines = lines
                self.content = self.read_lines()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                await self.content.aclose()

            async def read_lines(self):
                for line in self.lines:
                    yield line

            async def read(self):
                return b"{}"

            def raise_for_status(self):
                return None

        class Session:
            def __init__(self, *responses):
                self.responses = iter(responses)
                self.requests = []

            def post(self, url, *, json):
                self.requests.append((url, json))
                return next(self.responses)

        def replay(session, args):
            # The in-memory transport never suspends for I/O.
            coroutine = request(session, *args)
            try:
                with self.assertRaises(StopIteration) as done:
                    coroutine.send(None)
                return done.exception.value
            finally:
                coroutine.close()

        events = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            b'data: {"usage":{"completion_tokens":2}}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ("http://e", "http://pd", "fixture", ("prompt", [root / "image.png"]),
                    4, "fixture", root)
            with self.assertRaisesRegex(ValueError, "cache entry"):
                replay(Session(Response()), args)
            (root / "fixture-0").mkdir()
            (root / "fixture-0" / "encoder_cache.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "incomplete generation"):
                replay(Session(Response(), Response(events)), args)
            session = Session(Response(), Response([*events, b"data: [DONE]"]))
            result = replay(session, args)
            messages = [body["messages"][0]["content"] for _, body in session.requests]
            self.assertEqual(messages[0][0], messages[1][0])
            self.assertEqual(result["tokens"], 2)
