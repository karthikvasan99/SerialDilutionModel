#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Compiling Scientific Poster..."
pdflatex -interaction=nonstopmode poster.tex
pdflatex -interaction=nonstopmode poster.tex

echo "Build complete: poster.pdf generated successfully."
