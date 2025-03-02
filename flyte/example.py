from flytekit import task


@task
def test() -> str:
    return "Hello, World!"
