# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class CLISubcommand:
    """Base class for CLI argument handlers."""

    name: str

    @property
    def help(self) -> str:
        from vllm.entrypoints.cli.registry import COMMANDS

        return COMMANDS[self.name].help

    def prepare(self, argv: list[str]) -> None:
        """Set up the process before building the selected command's parser."""
        from vllm.entrypoints.common.startup import cli_env_setup

        cli_env_setup()

    def parse_args(self, argv: list[str]) -> argparse.Namespace:
        from vllm.entrypoints.cli.registry import create_parser
        from vllm.logger import configure_logging_from_args

        args = create_parser(self).parse_args([self.name, *argv])
        configure_logging_from_args(args)
        self.validate(args)
        return args

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        # No validation by default
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        raise NotImplementedError("Subclasses should implement this method")
