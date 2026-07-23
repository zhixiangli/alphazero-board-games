import pytest

from alphazero import stdio_play
from gomoku_9_9.config import GomokuConfig
from gomoku_9_9.game import ChessType, GomokuGame


@pytest.mark.unit
def test_stdio_game_exits_when_checkpoint_does_not_load(monkeypatch):
    monkeypatch.setattr(stdio_play.AlphaZeroNNet, "load_checkpoint", lambda *_: False)
    monkeypatch.setattr("sys.argv", ["stdio-play"])

    with pytest.raises(SystemExit):
        stdio_play.run_stdio_game(
            config_class=GomokuConfig,
            game_class=GomokuGame,
            chess_type=ChessType,
            title="Test",
            description="Test game",
            parse_move=lambda *_: None,
            format_action=lambda *_: "",
            print_board=lambda *_: None,
            help_message="",
            invalid_move_message="",
        )
