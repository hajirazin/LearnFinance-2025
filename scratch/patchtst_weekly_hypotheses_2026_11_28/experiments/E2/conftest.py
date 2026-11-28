"""Apply the repository I/O guard to synthetic research tests."""

pytest_plugins = ["tests.offline"]

# Archived run sources are evidence; exercise the current tests above data/.
collect_ignore = ["data"]
