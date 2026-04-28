.PHONY: install install-all install-tkinter install-qt install-web-flask install-web-streamlit build-tkinter build-qt build-web-flask build-streamlit clean test

install:
	uv pip install -e .

install-all:
	uv pip install -e ".[all]"

install-tkinter:
	uv pip install -e .

install-qt:
	uv pip install -e ".[qt]"

install-web-flask:
	uv pip install -e ".[web-flask]"

install-web-streamlit:
	uv pip install -e ".[web-streamlit]"

build-tkinter:
	pyinstaller mlui.spec --clean

build-qt:
	pyinstaller qt.spec --clean

build-web-flask:
	@echo "Flask app can be run directly with: flask --app gui.flask_app run"

build-streamlit:
	@echo "Streamlit app can be run directly with: streamlit run gui/streamlit_app.py"

clean:
	rm -rf build dist *.egg-info
	rm -rf core/*.egg-info gui/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	uv run python -c "from core import train_model_with_config; print('Core imports OK')"
	uv run python -c "from gui.tkinter_app import main; print('Tkinter app imports OK')"

run-tkinter:
	uv run python mlui.py

run-flask:
	uv run flask --app gui.flask_app run

run-streamlit:
	uv run streamlit run gui/streamlit_app.py

run-qt:
	uv run python -m gui.qt_app
