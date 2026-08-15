"""Tests for adaptivekv.cli — command line interface commands."""

from __future__ import annotations

import pytest

from adaptivekv.cli import main


class TestCLI:
    """Test CLI commands."""

    def test_cli_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "AdaptiveKV" in captured.out

    def test_cli_inspect(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["inspect"])
        captured = capsys.readouterr()
        assert "AdaptiveKV" in captured.out
        assert "Supported Hugging Face Architectures" in captured.out
        assert "llama" in captured.out

    def test_cli_compare(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["compare", "--heads", "2", "--seq-len", "32", "--dim", "64"])
        captured = capsys.readouterr()
        assert "Quantization Comparison" in captured.out
        assert "4-bit" in captured.out
        assert "3-bit" in captured.out
        assert "2-bit" in captured.out

    def test_cli_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info"])
        captured = capsys.readouterr()
        assert "AdaptiveKV Library Information" in captured.out
        assert "Package Version:" in captured.out
        assert "Kernel Backend:" in captured.out

