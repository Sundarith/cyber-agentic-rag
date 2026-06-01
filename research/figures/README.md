# Paper Figures

## Agentic RAG Architecture

`agentic_rag_architecture.tex` is an Overleaf-ready standalone TikZ figure.

Compile by itself:

```bash
pdflatex agentic_rag_architecture.tex
```

Use inside a paper by copying the TikZ preamble pieces into the main LaTeX file and
placing the `tikzpicture` inside a normal `figure` environment, or by compiling this
file to PDF and including it with:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/agentic_rag_architecture.pdf}
  \caption{Proposed agentic CTI-RAG architecture.}
  \label{fig:agentic-rag-architecture}
\end{figure}
```
