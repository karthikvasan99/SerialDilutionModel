#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Compiling LaTeX manuscript..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Build complete: main.pdf generated successfully."
