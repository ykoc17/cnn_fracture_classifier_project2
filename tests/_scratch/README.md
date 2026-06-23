# Test scratch space

Unit tests write temporary NPZ and checkpoint files here because the Windows
sandbox does not permit writes to the system temporary directory. Generated
files are removed by the tests and ignored by Git.
