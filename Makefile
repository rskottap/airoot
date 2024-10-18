PKG = airoot
PYTHON = python3
PIP = pip3

build: build-deps
	$(PYTHON) -m build

install: build
	$(PIP) install dist/*.whl

develop:
	$(PIP) install -e .[dev]

check:
	pytest -v tests

uninstall:
	$(PIP) uninstall $(PKG)

clean:
	rm -rvf dist/ build/ src/*.egg-info

push-test:
	$(PYTHON) -m twine upload --repository testpypi dist/*.whl

pull-test:
	$(PIP) install -i https://test.pypi.org/simple/ $(PKG)

push-prod:
	$(PYTHON) -m twine upload dist/*.whl

pull-prod:
	$(PIP) install $(PKG)

build-deps:
	@$(PYTHON) -c 'import build' &>/dev/null || $(PIP) install build
