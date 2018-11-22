install:
	python3 setup.py install

build:
	python3 setup.py build_ext --inplace

uml:
	pyreverse -o eps pynteractome/*.py

.PHONY: install build uml
