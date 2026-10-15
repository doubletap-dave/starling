from starling.hf_progress import make_tqdm_class
from starling.progress import ProgressStore, progress_store


def test_progress_percent():
    store = ProgressStore()
    job = store.start("j1", "whisper-tiny", "download", message="start")
    assert job.percent is None
    store.update("j1", bytes_total=1000)
    store.add_bytes("j1", 250)
    assert store.get("j1").percent == 25.0
    store.finish("j1")
    assert store.get("j1").status == "done"
    assert store.get("j1").percent == 100.0


def test_job_tqdm_set_description_str():
    progress_store.start("jd", "whisper-tiny", "download")
    Tqdm = make_tqdm_class("jd")
    bar = Tqdm(total=100, desc="model.safetensors")
    bar.set_description_str("config.json")
    assert bar.desc == "config.json"
    job = progress_store.get("jd")
    assert job is not None
    assert job.file_name == "config.json"
    progress_store.finish("jd")


def test_jobs_endpoint(client):
    r = client.get("/v1/jobs")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_download_async_already_cached(client):
    # Fixture marks everything downloaded
    r = client.post("/v1/models/whisper-tiny?wait=false")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "already_downloaded"
