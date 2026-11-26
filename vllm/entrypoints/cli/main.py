# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch the vLLM CLI to the selected command."""

import sys
from importlib.util import find_spec

from vllm.entrypoints.cli.registry import COMMANDS, create_parser, load_command
from vllm.logger import init_logger

logger = init_logger(__name__)


def maybe_run_omni() -> bool:
    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" not in sys.argv:
        return False

    # NOTE: Check the spec instead of importing directly here, since things could
    # fail with ImportError due to mismatched versions if things are moved around.
    spec = find_spec("vllm_omni")
    if spec is None:
        logger.error(
            "--omni flag requires a valid instance of vllm-omni to be installed."
        )
        sys.exit(1)

    from vllm_omni.entrypoints.cli.main import main as omni_main

    logger.info("Delegating entrypoint handling to vllm-omni")
    omni_main()

    return True


def main():
    if maybe_run_omni():
        return

    argv = sys.argv[1:]
    if argv and argv[0] in COMMANDS:
        command = load_command(argv[0])
        command.prepare(argv[1:])
        args = command.parse_args(argv[1:])
        args.dispatch_function(args)
    else:
        parser = create_parser()
        parser.parse_args(argv)
        parser.print_help()


if __name__ == "__main__":
    main()
