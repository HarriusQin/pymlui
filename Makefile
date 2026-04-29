.PHONY: install install-all install-qt build-qt clean test

install:
	uv pip install -e .

install-all:
	uv pip install -e .

install-qt:
	uv pip install -e .

build-qt:
	rm -rf dist/pymlui-qt
	uv run pyinstaller qt.spec --clean

clean:
	rm -rf build dist *.egg-info
	rm -rf core/*.egg-info gui/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	uv run python -c "from core import train_model_with_config; print('Core imports OK')"
	uv run python -c "from gui.qt_app import main; print('Qt app imports OK')"

run:
	uv run python mlui.py
