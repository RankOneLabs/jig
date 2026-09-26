"""Lock the supported jig.jev submodule surface."""

import jig.jev as jev

EXPECTED = {
    "JevClient", "NoulQuestion", "ChoiceQuestion", "ScoreQuestion",
    "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "JevResult", "JevUsage",
    "JevError", "JevErrorKind", "to_jig_usage",
}


def test_jev_public_exports():
    assert set(jev.__all__) == EXPECTED
    for name in EXPECTED:
        assert hasattr(jev, name)


def test_documented_imports():
    from jig.jev import JevClient, NoulAnswer, NoulQuestion, ChoiceQuestion
    assert JevClient is jev.JevClient
    assert NoulAnswer is jev.NoulAnswer
    assert NoulQuestion is jev.NoulQuestion
    assert ChoiceQuestion is jev.ChoiceQuestion
