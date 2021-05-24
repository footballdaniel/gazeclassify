# Uploading to PyPI

## Abstract 
The aim is to build a python package and distribute it to the PyPI server.

## Requirements
- Package `build`
- Package `twine`
- Account on PyPI test server ([Link](https://test.pypi.org/account/register/)) for testing
- Account on [pypi.org](https://pypi.org/)

## Documentation of the worklfow
### Build package
When the basic structure of the package is available, build it:

```
python -m pip install --upgrade build
python -m build # Build will be placed in /dist folder
```

### Upload to test server
- Create an account on PyPI test server ([Link](https://test.pypi.org/account/register/)) and verify the email address.
- Create an API token to upload the project.
  - Create a generic token [here](https://test.pypi.org/manage/account/#api-tokens), don't limit the scope.
  - The token should look something like: 
	```
	 pypi-AgENdGVzdC5weXBpLm9yZwIkNTNhYTMyZGYtZjc5NS00MTQ3LWE1NmUtMzFlOTNjNjVjM2M3AAIleyJwZXJtaXNzaW9ucyI6ICJ1c2VyIiwgInZlcnNpb24iOiAxfQAABiCVwABy7sbG2hJUWkieMCfrQfm6WNl-VQ7RDL-vbJ61Bw
	 ```
  - Copy the token when it appears.
- Upload the package to the test server using `twine`:
	```
	python -m pip install --user --upgrade twine
    python setup.py develop # Any dependencies in install_requires() have to be developed!
	python -m twine upload --repository testpypi dist/*
	```
- Use uploaded test package locally. Note that any packages defined in `install_requires` will need the extr-index-url.
  The test server only searches for packages on the test server, and most of the packages are directly on the pypi server.
 	```
    pip install -i https://test.pypi.org/simple/ gazeclassify --extra-index-url https://pypi.org/simple/
	```
  - When prompted to use a username, type `__token__`
  - For the password, paste the token value from above and hit enter

### Iterative upload for continuous integration
- To upload a newer version to the test server, use the two commands sequentially:
  ```
  python -m build
  python -m twine upload --repository testpypi dist/*
  ```

### View in test server
- The package is now available under `https://test.pypi.org/`
- To install the package locally, use
	```
	pip install -i https://test.pypi.org/simple/ gazeclassify
	```
- To use locally in a python script, do
	```
	import example_pkg
	```
- Note that the import refers to the folder, not the package name. its best practice to have the folder with the code having the same name as the package

### Upload package to PyPI server
- Create another user account on [pypi.org](https://pypi.org/)
- Use `python -m build` to build the package
- Use `twine upload dist/*` to send the package to the server

### Save token in Home/.pypirc
```
[pypi]
  username = __token__
  password = tokengoeshere
```