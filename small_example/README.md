# muChirality

Project to simulate chiral metamaterials using muSpectre. It uses [muSpectre](https://gitlab.com/muspectre/muspectre), [numpy](https://numpy.org/) and [pytest](https://pytest.org).

## Tests

Before being able to run tests, you need to execute
```python
pip install -e .[test]
```
to editably install the code.

## Examples

The folder small_example contains the torsion of a beam with a square cross section as a small example. It uses the following softwares:
python (version 3.12.3), muSpectre (version 0.27.0 - commit 2016a4bc) and numpy (version 2.2.5).

The parallel version also uses PFFT (commit e4cfcf9) and NuMPI (version 0.7.3).