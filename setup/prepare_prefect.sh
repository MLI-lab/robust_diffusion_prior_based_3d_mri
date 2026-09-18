#!/bin/bash
# Point the Prefect CLI/SDK at the server the flows use and persist task results (needed for caching).
prefect config set PREFECT_API_URL=${PREFECT_API_URL:-http://localhost:4200/api}
prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true
