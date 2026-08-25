IMPORTANT:
RUN "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/strip_alpha_fix.py" AS SOON AS A NEW .png FILE IS CREATED!!!!

PDF COMPILATION USING TERMINAL:
cd /Users/marcel/PycharmProjects/Master_Thesis_Pavia/thesis
rm -f *.aux *.bbl *.bcf *.blg *.run.xml *.out *.toc *.lof *.lot *.equ *.log
pdflatex 01_main.tex
biber 01_main
pdflatex 01_main.tex
pdflatex 01_main.tex