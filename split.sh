#!/bin/bash

N=$1
shift

## grab data from stdin
cat | shuf > temp.$$.txt

## all but last N
head -n -${N} temp.$$.txt > temp.$$.train.txt

## only last N
tail -n ${N} temp.$$.txt > temp.$$.test.txt

## run command with split data appended
$@ temp.$$.train.txt temp.$$.test.txt

## remove temp files
rm temp.$$.*
