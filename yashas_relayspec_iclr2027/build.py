"""Compile the paper and run BibTeX once citations have been added."""

from pathlib import Path
import os
import shutil
import subprocess


root = Path(__file__).resolve().parent
build = root / "build"
build.mkdir(exist_ok=True)
output = root / "output" / "pdf"
output.mkdir(parents=True, exist_ok=True)


def latex():
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory=build", "main.tex"],
        cwd=root, check=True,
    )


latex()
aux = (build / "main.aux").read_text()
if "\\citation{" in aux:
    env = os.environ.copy()
    for variable in ["BIBINPUTS", "BSTINPUTS"]:
        env[variable] = str(root) + os.pathsep + env.get(variable, "")
    subprocess.run(["bibtex", "main"], cwd=build, env=env, check=True)
    latex()
latex()
destination = output / "paper.pdf"
shutil.copy2(build / "main.pdf", destination)
print(f"PDF: {destination}")
