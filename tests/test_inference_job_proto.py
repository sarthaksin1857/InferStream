from inferstream.grpc import InferenceJob


def test_inference_job_roundtrip():
    job = InferenceJob(name="batch-7")
    assert job.name == "batch-7"
    data = job.SerializeToString()
    restored = InferenceJob()
    restored.ParseFromString(data)
    assert restored.name == "batch-7"
