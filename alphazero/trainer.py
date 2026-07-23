#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Common training utilities for AlphaZero game training.

Provides reusable helpers for logging setup, CLI argument parsing,
and training orchestration.  Game-specific trainers (e.g.
``gomoku_9_9/trainer.py``) add their own arguments, build the game
config, and call :func:`run_training`.
"""

import argparse
import logging
import os
import sys
from dataclasses import MISSING, fields

# Dataclass field types that map directly to argparse ``type`` converters.
_ARGPARSE_TYPES = {int, float, str}


def _parse_conv_kernel(value):
    try:
        kernel = tuple(int(part) for part in value.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError("conv_kernel must be H,W") from e
    if len(kernel) != 2 or any(size <= 0 or size % 2 == 0 for size in kernel):
        raise argparse.ArgumentTypeError(
            "conv_kernel must contain two positive odd integers"
        )
    return kernel


def setup_logging(logpath, console=False):
    """Configure the root logger with file and optional console handlers."""
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_dir = os.path.dirname(logpath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    absolute_logpath = os.path.abspath(logpath)
    has_file_handler = any(
        getattr(handler, "_alphazero_log_path", None) == absolute_logpath
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(logpath)
        file_handler._alphazero_log_path = absolute_logpath
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    has_console_handler = any(
        getattr(handler, "_alphazero_console_handler", False)
        for handler in root_logger.handlers
    )
    if console and not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler._alphazero_console_handler = True
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def add_config_args(parser, config_class):
    """Auto-register CLI arguments for every field of a config dataclass.

    Reads field names, types, and defaults via :func:`dataclasses.fields`
    so that adding a new field to the config automatically makes it
    available on the command line — no manual argument registration needed.

    Args:
        parser: An :class:`argparse.ArgumentParser` instance.
        config_class: A :func:`dataclasses.dataclass` whose fields define
            the CLI arguments (e.g. :class:`GomokuConfig`).
    """
    for f in fields(config_class):
        kwargs = {}
        if f.default is not MISSING:
            kwargs["default"] = f.default
        elif f.default_factory is not MISSING:
            kwargs["default"] = f.default_factory()
        if f.name == "conv_kernel":
            kwargs["type"] = _parse_conv_kernel
        elif f.type in _ARGPARSE_TYPES:
            kwargs["type"] = f.type
        parser.add_argument(f"-{f.name}", **kwargs)


def build_config_from_args(config_class, cli_args):
    """Build a *config_class* instance from parsed CLI arguments.

    Only attributes whose names match a field in *config_class* are
    forwarded, so extra CLI flags (e.g. ``-logpath``) are silently
    ignored.

    Args:
        config_class: A :func:`dataclasses.dataclass` to instantiate.
        cli_args: The :class:`argparse.Namespace` returned by
            ``parser.parse_args()``.

    Returns:
        An instance of *config_class* populated from *cli_args*.
    """
    field_names = {f.name for f in fields(config_class)}
    kwargs = {k: v for k, v in vars(cli_args).items() if k in field_names}
    return config_class(**kwargs)


def run_training(module, game_class, config):
    """Create a trainer via the DI module, load checkpoint, and start training.

    This is the standard training entry-point shared by all games.
    """
    logging.info(config)
    trainer = module.create_trainer(game_class, config)
    trainer.nnet.load_checkpoint(config.save_checkpoint_path)
    trainer.start()
