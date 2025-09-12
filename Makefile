default:
	python3 -m pip install ../mermake/ --user

clean:
	rm -fr build
	rm -fr dist
	rm -fr mermake.egg-info
	python3 -m pip uninstall mermake -y
egg:
	pip install -e .
build:
	python3 -m build
	python3 -m twine upload dist/*

test:
	pytest tests/ -v
