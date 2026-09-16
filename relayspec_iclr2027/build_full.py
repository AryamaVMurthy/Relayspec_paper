"""Compile the placeholder-included variant of the paper (main_full.tex)."""

from pathlib import Path
import os
import shutil
import subprocess


root = Path(__file__).resolve().parent
build = root / "build_full"
build.mkdir(exist_ok=True)
output = root / "output" / "pdf"
output.mkdir(parents=True, exist_ok=True)


def latex():
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory=build_full", "main_full.tex"],
        cwd=root, check=True,
    )


latex()
aux = (build / "main_full.aux").read_text()
if "\\citation{" in aux:
    env = os.environ.copy()
    for variable in ["BIBINPUTS", "BSTINPUTS"]:
        env[variable] = str(root) + os.pathsep + env.get(variable, "")
    subprocess.run(["bibtex", "main_full"], cwd=build, env=env, check=True)
    latex()
latex()
destination = output / "paper_full.pdf"
shutil.copy2(build / "main_full.pdf", destination)
print(f"PDF: {destination}")
