install:
	python3 setup.py install

build:
	python3 setup.py build_ext --inplace

uml:
	pyreverse -ASmy -o eps pynteractome/*.py pynteractome/**/*.py pynteractome/**/**/*.py

.PHONY: install build uml
