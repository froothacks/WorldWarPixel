#!/usr/bin/env bash
# Build the image with Cloud Build and deploy one always-on Cloud Run instance, with the timelapse
# bucket mounted at /data (SQLite backups + rendered videos land straight in GCS). Idempotent; run
# locally with gcloud authenticated or from .github/workflows/deploy.yml.
#
#   deploy/cloudrun.sh                 # build + deploy
#   APP_ENV_FILE=app.env deploy/cloudrun.sh   # also set app settings (KEY=value lines)
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-world-war-pixel-514101}"
REGION="${GCP_REGION:-us-west1}"
SERVICE="${CLOUD_RUN_SERVICE:-worldwarpixel}"
BUCKET="${TIMELAPSE_BUCKET:-world-war-pixel-timelapses-514101}"
REPO="${ARTIFACT_REPO:-worldwarpixel}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/web:$(git rev-parse --short HEAD)"

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --location="$REGION" --repository-format=docker
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

gcloud builds submit --tag "$IMAGE" --quiet .

args=(
  --image "$IMAGE" --region "$REGION" --platform managed --allow-unauthenticated
  --port 8080 --cpu 1 --memory 2Gi --min-instances 1 --max-instances 1 --no-cpu-throttling
  --concurrency 250 --timeout 3600 --session-affinity --execution-environment gen2
  --add-volume "name=data,type=cloud-storage,bucket=$BUCKET"
  --add-volume-mount volume=data,mount-path=/data
)
if [ -n "${APP_ENV_FILE:-}" ]; then
  # KEY=value lines -> the YAML gcloud expects; keeps the values out of the command line.
  yaml="$(mktemp)"
  grep -v '^\s*\(#\|$\)' "$APP_ENV_FILE" | sed -E 's/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/\1: "\2"/' > "$yaml"
  args+=(--env-vars-file "$yaml")
fi
gcloud run deploy "$SERVICE" "${args[@]}" --quiet
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
