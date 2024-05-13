# Building the ThermoNTFA Python documentation

To build the documentation:

1. Install ThermoNTFA (`thermontfa` pip package). It must be possible to import
   the module ``thermontfa``.
2. Run in this directory:
 
       make html

## Processing demo/example programs

Python demo programs are written in Python, with MyST-flavoured
Markdown syntax for comments.

1. `jupytext` reads the Python demo code, then converts to and writes a
   Markdown file.
2. `myst_parser` allows Sphinx to process the Markdown file.
