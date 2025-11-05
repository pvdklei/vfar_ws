# .latexmkrc - LaTeX build configuration
# This file configures latexmk to store all build artifacts in a build directory

# Use pdf output
$pdf_mode = 1;

# Put *all* build artifacts in ./build
$out_dir = 'build';
$aux_dir = 'build';

# Nice pdflatex command with synctex for editor integration
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';

# Use bibtex for bibliography
$bibtex_use = 2;

# Continuous preview mode settings
$preview_continuous_mode = 1;
$pdf_previewer = 'open -a Preview %S';

# Clean up auxiliary files
$clean_ext = 'aux bbl blg log out toc lof lot nav snm vrb synctex.gz fls fdb_latexmk';
