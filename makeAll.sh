#!/bin/bash

for file in $(ls): do
    if [ -d "$file" ] ; then
        cd $(file)
        make
        cd ..
    fi
done