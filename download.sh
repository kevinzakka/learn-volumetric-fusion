#!/usr/bin/env bash
set -e
set -x

ID="1-rt9q0FY5p1MsKMPrdsfD9OAYpqQbMnF"
gdown https://drive.google.com/uc?id=$ID
unzip -qq data.zip