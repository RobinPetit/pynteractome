REMOTE_ORIGIN=git@github.com:RobinPetit/pynteractome.git
DOCS_DIR =../../pynteractome-doc/

install:
	make -C src/ install

doc:
	make -C src/ build
	export SPHINX_APIDOC_OPTIONS=members,private-members,show-inheritance,ignore-module-all && \
	sphinx-apidoc -Mef -o doc/source/pynteractome/ src/pynteractome/ src/pynteractome/setup.py
	make -C doc/ html

pushdoc:
	cd ${DOCS_DIR}/html && \
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi && \
	git add . && \
	git commit -m "Build the doc" && \
	git push -f origin HEAD:gh-pages

.PHONY: doc install pushdoc
