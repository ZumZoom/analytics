#!/bin/bash
set -euo pipefail

die () {
    echo -e >&2 "$@"
    exit 1
}

[ "$#" -eq 1 ] || die "Please provide link to your analytics repository.\nUsage: setup-dist.sh git@github.com:%USERNAME%/analytics.git"

cd "$(dirname "$0")"/..
git clone -b gh-pages "$1" dist
