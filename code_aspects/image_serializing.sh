#!/bin/sh
num=1
for file in *.jpg; do
       mv "$file" "$(printf "%0.4d" $num).jpg"
       let num=$num+1
done
