# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-level CLI registration. Loading a command imports only its module."""

import importlib
import importlib.metadata
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from vllm.entrypoints.cli.types import CLISubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser


class CommandSpec(NamedTuple):
    module: str
    help: str


COMMANDS = {
    "chat": CommandSpec(
        "openai", "Generate chat completions via the running API server."
    ),
    "complete": CommandSpec(
        "openai",
        "Generate text completions based on the given prompt "
        "via the running API server.",
    ),
    "serve": CommandSpec(
        "serve",
        "Launch a local OpenAI-compatible API server to serve "
        "LLM completions via HTTP.",
    ),
    "launch": CommandSpec("launch", "Launch individual vLLM components."),
    "bench": CommandSpec("benchmark.main", "vLLM bench subcommand."),
    "collect-env": CommandSpec(
        "collect_env", "Start collecting environment information."
    ),
    "download-kernels": CommandSpec(
        "download_kernels", "Install precompiled kernels for faster startup."
    ),
    "preload": CommandSpec(
        "preload", "Launch weight cache daemons (one per GPU) for fast engine restarts."
    ),
    "run-batch": CommandSpec(
        "run_batch", "Run batch prompts and write results to file."
    ),
    "snapshot": CommandSpec(
        "snapshot", "Create, inspect, or restore an initialized vLLM snapshot."
    ),
}


def load_command(name: str) -> "CLISubcommand":
    module = importlib.import_module(f".{COMMANDS[name].module}", __package__)
    return next(cmd for cmd in module.cmd_init() if cmd.name == name)


def create_parser(command: "CLISubcommand | None" = None) -> "FlexibleArgumentParser":
    from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v", "--version", action="version", version=importlib.metadata.version("vllm")
    )
    subparsers = parser.add_subparsers(dest="subparser")
    for name, spec in COMMANDS.items():
        if command is not None and command.name == name:
            command.subparser_init(subparsers).set_defaults(
                dispatch_function=command.cmd
            )
        else:
            subparsers.add_parser(name, help=spec.help)
    return parser
