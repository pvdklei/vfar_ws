# Lab 1 Report: Line Detection and Following by UGV Rover

This directory contains the LaTeX source files for the Lab 1 assignment report.

## Document Structure

The report is split into multiple files for better organization:

- **main.tex** - Main document file that includes all sections
- **introduction.tex** - Introduction section with objectives and background
- **session1.tex** - Session 1: Line Detection implementation and results
- **session2.tex** - Session 2: Line Following implementation and results
- **conclusion.tex** - Conclusion with summary, findings, and future work

## Building the Report

### Prerequisites

Make sure you have a LaTeX distribution installed:
- **macOS**: MacTeX (`brew install --cask mactex`)
- **Linux**: TeX Live (`sudo apt-get install texlive-full`)
- **Windows**: MiKTeX or TeX Live

### Build Instructions

#### Using latexmk (Recommended)

The easiest way to build the report is using `latexmk`, which handles all dependencies automatically:

```bash
cd report
latexmk -pdf main.tex
```

This will:
- Compile the LaTeX document
- Run BibTeX for references
- Store all build artifacts in the `build/` directory
- Generate `build/main.pdf`

#### Continuous Build Mode

For continuous compilation while editing:

```bash
latexmk -pdf -pvc main.tex
```

This will automatically recompile whenever you save changes to any `.tex` file.

#### Manual Build

If you prefer to build manually:

```bash
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

### Viewing the PDF

After building, the PDF will be located at:
```
build/main.pdf
```

On macOS, you can open it with:
```bash
open build/main.pdf
```

### Cleaning Build Files

To clean all build artifacts:

```bash
latexmk -C
```

Or manually:
```bash
rm -rf build/
```

## Adding Content

### Adding Figures

1. Create a `figures/` directory in the report folder
2. Place your images there (PNG, JPG, or PDF format recommended)
3. Include them in your LaTeX files:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/your_image.png}
    \caption{Your caption here}
    \label{fig:your_label}
\end{figure}
```

### Adding Tables

Example table structure:

```latex
\begin{table}[H]
\centering
\caption{Your table caption}
\label{tab:your_label}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\ \midrule
Row 1 & Value 1 & Value 2 \\
Row 2 & Value 3 & Value 4 \\ \bottomrule
\end{tabular}
\end{table}
```

### Adding Code

For inline code: `\texttt{your_code_here}`

For code blocks:

```latex
\begin{lstlisting}[language=Python, caption={Your caption}]
def your_function():
    # Your code here
    pass
\end{lstlisting}
```

### Adding References

Add references to the bibliography section in `main.tex`:

```latex
\bibitem{your_ref}
Author Name,
\textit{Title of Work},
Publisher, Year.
```

Cite in text with: `\cite{your_ref}`

## Report Guidelines Checklist

Make sure your report follows these requirements:

- [ ] Maximum 10 pages (single-column, including tables and figures)
- [ ] Includes introduction (Section 1)
- [ ] Includes conclusion (Section 4)
- [ ] Answers all questions from the assignment (marked with green boxes)
- [ ] Briefly describes all implementations
- [ ] Analyzes and discusses findings (why algorithm A works better than B)
- [ ] All tables have: numbers, titles, units of variables
- [ ] All figures have: numbers, captions, axis labels with units, legends

## Tips for Writing

1. **Replace all PLACEHOLDER comments** with your actual content
2. **Add your images** to a `figures/` directory
3. **Fill in your actual measurements** in tables
4. **Update student names and IDs** in main.tex
5. **Ensure all cross-references work** (use `\ref{label}` for figures, tables, sections)
6. **Check for LaTeX errors** by running the build
7. **Proofread** the final PDF for formatting issues

## File Organization

```
report/
├── README.md              # This file
├── .latexmkrc            # Build configuration
├── .gitignore            # Git ignore rules
├── main.tex              # Main document
├── introduction.tex      # Introduction section
├── session1.tex          # Session 1 content
├── session2.tex          # Session 2 content
├── conclusion.tex        # Conclusion section
├── figures/              # (Create this) Image files
│   ├── pipeline.png
│   ├── result1.png
│   └── ...
└── build/                # (Generated) Build artifacts
    └── main.pdf          # Final PDF output
```

## Common Issues

### Build Errors

If you encounter build errors:
1. Check that all `\input{}` files exist
2. Ensure all `\includegraphics{}` paths are correct
3. Verify all `\ref{}` and `\label{}` are properly paired
4. Make sure all packages are installed

### Missing Figures

If figures don't appear:
1. Check the image path is correct
2. Ensure the image file exists
3. Verify the image format is supported (PNG, JPG, PDF)

### Cross-Reference Warnings

If you see "Reference ... undefined" warnings:
- Run the build twice (or use latexmk which handles this automatically)

## Contact

For questions about the report structure or LaTeX issues, consult your team members or course instructors.
