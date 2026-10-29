from alayarenderer_flash.demo.logcap import prune_hang_dumps, rotate_if_needed, sweep


def test_rotation_keeps_backups_and_appending_writer(tmp_path):
    log = tmp_path / "worker.log"
    writer = open(log, "ab", buffering=0)  # supervisors append through O_APPEND
    writer.write(b"x" * 100)
    assert not rotate_if_needed(log, max_bytes=200, backups=2)
    for round_ in range(3):
        writer.write(bytes([65 + round_]) * 300)
        assert rotate_if_needed(log, max_bytes=200, backups=2)
    writer.write(b"tail")
    writer.close()
    assert log.read_bytes() == b"tail"
    assert (tmp_path / "worker.log.1").read_bytes() == b"C" * 300
    assert (tmp_path / "worker.log.2").read_bytes() == b"B" * 300
    assert not (tmp_path / "worker.log.3").exists()


def test_sweep_and_hang_dump_pruning(tmp_path):
    (tmp_path / "bridge.log").write_bytes(b"y" * 50)
    (tmp_path / "stk.log").write_bytes(b"z" * 5)
    for i in range(5):
        (tmp_path / f"worker_hang_2026102{i}T000000Z.txt").write_text("dump")
    assert sweep(tmp_path, max_bytes=10, backups=3, keep_hang_dumps=2) == [tmp_path / "bridge.log"]
    assert sorted(p.name for p in tmp_path.glob("worker_hang_*")) == [
        "worker_hang_20261023T000000Z.txt",
        "worker_hang_20261024T000000Z.txt",
    ]
    assert prune_hang_dumps(tmp_path, keep=2) == 0
