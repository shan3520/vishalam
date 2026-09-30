"""HubSync: checkpoint uploads squash history (keeps HF private storage bounded); errors never kill training."""
from vishalam.train.hub import HubSync


class FakeApi:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def upload_file(self, **kw):
        if self.fail:
            raise RuntimeError("network down")
        self.calls.append(("upload", kw["path_in_repo"]))

    def super_squash_history(self, repo_id, **kw):
        self.calls.append(("squash", repo_id, kw.get("repo_type")))


def make(fail=False) -> HubSync:
    h = HubSync.__new__(HubSync)  # skip __init__ (it would contact the Hub)
    h.repo_id, h.repo_type, h.api = "me/model", "model", FakeApi(fail)
    h._thread, h.last_error = None, None
    return h


def test_push_async_squashes_after_upload(tmp_path):
    h = make()
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"x")
    h.push_async([(f, "m/latest/ckpt.pt"), (f, "m/latest/log.csv")], "step 1", squash=True)
    h.wait()
    assert h.api.calls == [("upload", "m/latest/ckpt.pt"), ("upload", "m/latest/log.csv"),
                           ("squash", "me/model", "model")]


def test_no_squash_by_default_and_errors_are_reported_not_raised(tmp_path, capsys):
    h = make()
    f = tmp_path / "a"
    f.write_bytes(b"x")
    h.push_async([(f, "a")], "m")
    h.wait()
    assert ("squash", "me/model", "model") not in h.api.calls
    bad = make(fail=True)
    bad.push_async([(f, "a")], "m", squash=True)
    bad.wait()  # must not raise
    assert "previous upload failed" in capsys.readouterr().out
    assert not any(c[0] == "squash" for c in bad.api.calls)
