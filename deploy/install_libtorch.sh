#!/usr/bin/env bash

ZIPFILE=libtorch-shared-with-deps-latest.zip
[ -f ${ZIPFILE} ] || wget https://download.pytorch.org/libtorch/nightly/cpu/${ZIPFILE}
unzip -qu ${ZIPFILE}
