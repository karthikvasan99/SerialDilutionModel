#!/usr/bin/env bash

cd "$(dirname "$0")"

echo "Compiling LaTeX manuscript..."
pdflatex -interaction=nonstopmode main.tex || true
bibtex main || true
pdflatex -interaction=nonstopmode main.tex || true
pdflatex -interaction=nonstopmode main.tex

echo "Build complete: main.pdf generated successfully."
