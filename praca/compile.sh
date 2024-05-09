#!/bin/bash

if [ -d _tex ]; then
  rm -rf tex;
fi

mkdir -p _tex

if ! [ -d _tex ]; then
	echo "Brak katalogu \`_tex' !"
	exit 1
fi

echo -ne "\033]0;▶LaTEX compilation ("$PWD")\007"

for x in *.tex
do
	if cat $x | grep -q "documentclass"
	then
		main=$x
		echo $x
	  cat $x > _tex/$x
  else
	  cat $x | sed 's/ [ ]*/ /g;s/ \([a-zA-Z0-9]\) / \1~ /g' > _tex/$x
	fi

done

for x in *.bib
do
	cat $x | sed 's/@misc/@online/g' > _tex/$x

done

cd _others
for x in *
do
  echo $x
  if [ -d $x ]; then
    cp $x ../_tex/
  fi
done

cd ..


if ! [ -e "$main" ]
then
	echo -e "Brak pliku \`*.tex' !\033[0;37m"
	exit 2
fi

cp ./*.sty ./_tex/
cd _tex;
ln -s ../images images
ln -s ../Dictionaries Dictionaries

main=$(basename $main .tex)
pdflatex $main && biber $main && pdflatex $main && pdflatex $main;

cp Dyplom.pdf ../Dyplom.pdf
cd ..

rm -rf _tex