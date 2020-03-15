@echo off

:: https://stackoverflow.com/questions/50612169/pylint-not-recognizing-cv2-members
call pylint --extension-pkg-whitelist=cv2 symbols

call pytest --cov-report term-missing --cov=symbols symbols
