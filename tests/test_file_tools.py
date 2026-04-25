"""
Tests for the new file-tool behavior:
- read_file: line-numbered output, mtime tracking, outline mode, multi-file
- edit_file: stale-read guard, near-miss hint, replace_all, batch atomic, anchor_line
- write_file: probe-required guard, backup capture
- check_syntax: py_compile + JSON
- undo_last_edit: restore + delete-new-file
- find_files: glob + ignored dirs
- go_to_definition / find_references: AST scanning
"""

from __future__ import annotations

import json
import os
import time

import pytest

from tools.file_tools import _state
from tools.file_tools import check, edit, read, undo, write  # noqa
from tools.file_tools.read import read_file, file_info, search_file
from tools.file_tools.edit import edit_file
from tools.file_tools.write import write_file
from tools.file_tools.check import check_syntax, _check_content
from tools.file_tools.undo import undo_last_edit
from tools.find import find_files
from tools.code_intel import go_to_definition, find_references


@pytest.fixture(autouse=True)
def _reset_and_autoconfirm(monkeypatch):
    _state.reset_session_state()
    monkeypatch.setattr("tools.file_tools._helpers.confirm", lambda _p: True)
    monkeypatch.setattr("tools.file_tools.write.confirm", lambda _p: True)
    monkeypatch.setattr("tools.file_tools.edit.confirm", lambda _p: True)
    monkeypatch.setattr("tools.file_tools.undo.confirm", lambda _p: True)
    yield
    _state.reset_session_state()


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_line_numbers(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def foo():\n    return 1\n")
    res = json.loads(read_file(path=str(p)))
    assert res["meta"]["line_numbers_shown"] is True
    assert "1 | def foo():" in res["content"]
    assert "2 |     return 1" in res["content"]


def test_read_file_line_range_with_numbers(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    res = json.loads(read_file(path=str(p), start_line=4, end_line=6))
    assert "4 | line4" in res["content"]
    assert "6 | line6" in res["content"]
    assert "line7" not in res["content"]
    assert res["meta"]["starting_line_number"] == 4


def test_read_file_records_mtime(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    read_file(path=str(p))
    assert _state.is_known(str(p))
    stale, _ = _state.is_stale(str(p))
    assert stale is False


def test_read_file_outline_python(tmp_path):
    src = (
        "import os\n"
        "\n"
        "CONST = 1\n"
        "\n"
        "def foo(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Bar:\n"
        "    def method(self):\n"
        "        return 1\n"
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    out = json.loads(read_file(path=str(p), outline=True))
    kinds = {(s["kind"], s["name"]) for s in out["symbols"]}
    assert ("function", "foo") in kinds
    assert ("class", "Bar") in kinds
    assert ("constant", "CONST") in kinds


def test_read_file_multi_file(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    res = json.loads(read_file(paths=[str(a), str(b)]))
    assert len(res["files"]) == 2
    paths = {f["meta"]["path"] for f in res["files"]}
    assert paths == {str(a), str(b)}


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_requires_prior_read(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    res = json.loads(edit_file(path=str(p), old_string="x = 1", new_string="x = 2"))
    assert "error" in res
    assert "not been read" in res["error"]


def test_edit_file_basic_after_read(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    read_file(path=str(p))
    res = json.loads(edit_file(path=str(p), old_string="x = 1", new_string="x = 2"))
    assert res["success"] is True
    assert p.read_text() == "x = 2\n"
    assert "post_edit" in res


def test_edit_file_stale_guard(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    read_file(path=str(p))
    time.sleep(0.1)  # > _MTIME_TOLERANCE_S
    p.write_text("x = 99\n")  # external change
    res = json.loads(edit_file(path=str(p), old_string="x = 99", new_string="x = 100"))
    assert "error" in res
    assert "Stale" in res["error"]


def test_edit_file_near_miss_hint(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def calculate_total():\n    pass\n")
    read_file(path=str(p))
    res = json.loads(
        edit_file(
            path=str(p), old_string="def calculatetotal():", new_string="def x():"
        )
    )
    assert "error" in res
    assert "Closest matches" in res["error"]
    assert "calculate_total" in res["error"]


def test_edit_file_replace_all(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("foo\nfoo\nfoo\n")
    read_file(path=str(p))
    res = json.loads(
        edit_file(path=str(p), old_string="foo", new_string="bar", replace_all=True)
    )
    assert res["success"]
    assert p.read_text() == "bar\nbar\nbar\n"


def test_edit_file_anchor_line(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("return None\n# anchor\nreturn None\n")
    read_file(path=str(p))
    res = json.loads(
        edit_file(
            path=str(p), old_string="return None", new_string="return 0", anchor_line=3
        )
    )
    assert res["success"]
    assert p.read_text() == "return None\n# anchor\nreturn 0\n"


def test_edit_file_batch_atomic_rollback(tmp_path):
    p = tmp_path / "a.py"
    original = "alpha\nbeta\n"
    p.write_text(original)
    read_file(path=str(p))
    # Second edit fails (substring missing) -> whole batch rolls back
    res = json.loads(
        edit_file(
            path=str(p),
            edits=[
                {"old": "alpha", "new": "ALPHA"},
                {"old": "MISSING_THING", "new": "X"},
            ],
        )
    )
    assert "error" in res
    assert p.read_text() == original  # unchanged


def test_edit_file_batch_success(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("alpha\nbeta\n")
    read_file(path=str(p))
    res = json.loads(
        edit_file(
            path=str(p),
            edits=[
                {"old": "alpha", "new": "ALPHA"},
                {"old": "beta", "new": "BETA"},
            ],
        )
    )
    assert res["success"]
    assert res["edits_applied"] == 2
    assert p.read_text() == "ALPHA\nBETA\n"


def test_edit_file_post_edit_context_present(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("a\nb\nc\nd\ne\n")
    read_file(path=str(p))
    res = json.loads(edit_file(path=str(p), old_string="c", new_string="C"))
    assert res["success"]
    assert "preview_lines" in res["post_edit"]


def test_edit_file_runs_syntax_check(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    read_file(path=str(p))
    res = json.loads(edit_file(path=str(p), old_string="x = 1", new_string="x ="))
    # syntactically broken -> py_compile reports error, but edit still applied
    assert res["success"]
    assert "py_compile" in res["checks"]
    assert res["checks"]["py_compile"] != "ok"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_creates_new(tmp_path):
    p = tmp_path / "new.py"
    res = json.loads(write_file(path=str(p), content="x = 1\n"))
    assert res["success"] and res["created"]
    assert p.read_text() == "x = 1\n"


def test_write_file_refuses_overwrite_unread(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("old\n")
    res = json.loads(write_file(path=str(p), content="new\n"))
    assert "error" in res
    assert "has not been read" in res["error"]


def test_write_file_overwrite_with_flag(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("old\n")
    res = json.loads(write_file(path=str(p), content="new\n", overwrite=True))
    assert res["success"]
    assert p.read_text() == "new\n"


def test_write_file_overwrite_after_read(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("old\n")
    read_file(path=str(p))
    res = json.loads(write_file(path=str(p), content="new\n"))
    assert res["success"]


# ---------------------------------------------------------------------------
# check_syntax
# ---------------------------------------------------------------------------


def test_check_syntax_python_ok(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    res = json.loads(check_syntax(path=str(p)))
    assert res["checks"]["py_compile"] == "ok"


def test_check_syntax_python_err(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def x(:\n")
    res = json.loads(check_syntax(path=str(p)))
    assert res["checks"]["py_compile"] != "ok"


def test_check_syntax_json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"a": 1}')
    res = json.loads(check_syntax(path=str(p)))
    assert res["checks"]["json"] == "ok"


def test_check_content_in_memory():
    out = _check_content("foo.py", "x = ")
    assert "py_compile" in out and out["py_compile"] != "ok"


# ---------------------------------------------------------------------------
# undo_last_edit
# ---------------------------------------------------------------------------


def test_undo_restores_overwrite(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("v1\n")
    read_file(path=str(p))
    write_file(path=str(p), content="v2\n")
    assert p.read_text() == "v2\n"
    res = json.loads(undo_last_edit())
    assert res["success"] and res["action"] == "restored"
    assert p.read_text() == "v1\n"


def test_undo_deletes_new_file(tmp_path):
    p = tmp_path / "fresh.py"
    write_file(path=str(p), content="x = 1\n")
    assert p.exists()
    res = json.loads(undo_last_edit())
    assert res["success"] and res["action"] == "deleted"
    assert not p.exists()


def test_undo_empty_ledger():
    res = json.loads(undo_last_edit())
    assert "error" in res


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------


def test_find_files_basic(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    res = json.loads(find_files(pattern="*.py", root=str(tmp_path)))
    paths = {m["path"] for m in res["matches"]}
    assert any(p.endswith("a.py") for p in paths)
    assert any(p.endswith("c.py") for p in paths)
    assert not any(p.endswith("b.txt") for p in paths)


def test_find_files_skips_heavy_dirs(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "junk.py").write_text("")
    (tmp_path / "real.py").write_text("")
    res = json.loads(find_files(pattern="*.py", root=str(tmp_path)))
    paths = [m["path"] for m in res["matches"]]
    assert any(p.endswith("real.py") for p in paths)
    assert not any("node_modules" in p for p in paths)


# ---------------------------------------------------------------------------
# code intelligence
# ---------------------------------------------------------------------------


def test_go_to_definition_finds_function(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("def special_thing():\n    pass\n\nclass Other:\n    pass\n")
    res = json.loads(go_to_definition(name="special_thing", path=str(p)))
    assert len(res["results"]) == 1
    assert res["results"][0]["kind"] == "function"
    assert res["results"][0]["line"] == 1


def test_find_references_picks_up_calls(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(
        "def helper():\n    return 1\n\n"
        "def main():\n    x = helper()\n    return helper() + 2\n"
    )
    res = json.loads(find_references(name="helper", path=str(p)))
    lines = [r["line"] for r in res["results"]]
    # The def on line 1 + two call references on lines 5 and 6
    assert 5 in lines and 6 in lines


def test_find_references_attribute_access(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("class C:\n    val = 1\n\nc = C()\nprint(c.val)\n")
    res = json.loads(find_references(name="val", path=str(p)))
    assert any(r.get("attribute") for r in res["results"])


# ---------------------------------------------------------------------------
# context_window: surgical_clear keys re-read pointers by path
# ---------------------------------------------------------------------------


def test_surgical_clear_keeps_readfile_path():
    from context_window import _surgical_clear

    big_payload = json.dumps(
        {"meta": {"path": "/tmp/some/file.py", "size_bytes": 10}, "content": "X" * 1000}
    )
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": big_payload},
        # Padding so the keep_recent tail keeps the bulky msg out of the safe zone
    ] + [{"role": "user", "content": "u"} for _ in range(10)]
    cleared = _surgical_clear(msgs, keep_recent=8)
    assert cleared
    assert "Re-run read_file(path='/tmp/some/file.py'" in msgs[1]["content"]
