# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import typing

from vllm import envs
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.registry import COMMANDS
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def maybe_exec_rust_bench(argv: list[str]) -> None:
    if argv[:1] != ["serve"] or not envs.VLLM_USE_RUST_BENCH:
        return

    rust_cli = envs.VLLM_RUST_FRONTEND_PATH
    if rust_cli is None:
        raise RuntimeError(
            "VLLM_USE_RUST_BENCH=1 requires VLLM_RUST_FRONTEND_PATH "
            "to resolve to the vllm-rs binary."
        )

    logger.info("Delegating `vllm bench serve` to Rust binary at %s.", rust_cli)
    os.execv(rust_cli, [rust_cli, "bench", *argv])


def _import_bench_subcommand_modules() -> None:
    # Imported lazily so `BenchmarkSubcommandBase` subclasses register only
    # when `vllm bench` is actually invoked.
    import vllm.entrypoints.cli.benchmark.forward  # noqa: F401
    import vllm.entrypoints.cli.benchmark.latency  # noqa: F401
    import vllm.entrypoints.cli.benchmark.mm_processor  # noqa: F401
    import vllm.entrypoints.cli.benchmark.serve  # noqa: F401
    import vllm.entrypoints.cli.benchmark.startup  # noqa: F401
    import vllm.entrypoints.cli.benchmark.sweep  # noqa: F401
    import vllm.entrypoints.cli.benchmark.throughput  # noqa: F401


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = COMMANDS[name].help

    def prepare(self, argv: list[str]) -> None:
        super().prepare(argv)
        maybe_exec_rust_bench(argv)
        # Offline benchmarks can build their CLI on a machine without an accelerator.
        from vllm import platforms

        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"vllm {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        _import_bench_subcommand_modules()
        for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"vllm {self.name} {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
                subcmd=f"{self.name} {cmd_cls.name}"
            )
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
