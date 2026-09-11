import json
from dataclasses import asdict
import pytest
from hirp import train_phase25 as runner


def test_completed_external_arm_never_creates_optimizer(tmp_path,monkeypatch):
    config=json.loads(json.dumps(asdict(runner.TrainConfig())))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=config)))
    summary=tmp_path/'summary.json';summary.write_text(json.dumps(dict(actual_optimizer_steps=2000,training_rows=2000)))
    finished=tmp_path/'finished.json';finished.write_text(json.dumps(dict(completed=True)))
    (tmp_path/'C2_concurrent_owner.json').write_text(json.dumps(dict(pid=1,summary_path=str(summary),finished_path=str(finished))))
    monkeypatch.setattr(runner,'configure',lambda:None)
    def unexpected(*a,**k):raise AssertionError('must not start optimizer or access data')
    monkeypatch.setattr(runner,'RealData',unexpected)
    monkeypatch.setattr('sys.argv',['runner','--run',str(tmp_path),'--arm','C2','--stop-after','2000'])
    runner.main()
    finished.write_text(json.dumps(dict(completed=False)))
    with pytest.raises(RuntimeError,match='external training failed'):runner.main()
