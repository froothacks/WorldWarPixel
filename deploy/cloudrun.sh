#!/usr/bin/env bash
# Build the image with Cloud Build and deploy one always-on Cloud Run instance, with the timelapse
# bucket mounted at /data (SQLite backups + rendered videos land straight in GCS). Idempotent; run
# locally with gcloud authenticated or from .github/workflows/deploy.yml.
#
#   APP_ENV_FILE=/path/outside/repo/app.env deploy/cloudrun.sh
#
# APP_ENV_FILE holds the app settings (KEY=value lines, same as the Modal secret). Keep the file
# outside the repo so `gcloud builds submit` never uploads it with the build context. GOD_TOKEN is
# optional: without it god mode is open to anyone with the URL, same as the Modal deploy.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-world-war-pixel-514101}"
REGION="${GCP_REGION:-us-west1}"
SERVICE="${CLOUD_RUN_SERVICE:-worldwarpixel}"
BUCKET="${TIMELAPSE_BUCKET:-world-war-pixel-timelapses-514101}"
REPO="${ARTIFACT_REPO:-worldwarpixel}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/web:$(git rev-parse --short HEAD)"
RUNTIME_SA="${CLOUD_RUN_SA:-worldwarpixel-run}@$PROJECT.iam.gserviceaccount.com"

if [ -z "${APP_ENV_FILE:-}" ]; then
  echo "APP_ENV_FILE must point to a KEY=value file with the app settings" >&2
  exit 1
fi
case "$(realpath "$APP_ENV_FILE")" in "$PWD"/*) echo "APP_ENV_FILE must be outside the repo" >&2; exit 1;; esac

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --location="$REGION" --repository-format=docker
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

# The app runs as its own service account with access to the data bucket only.
gcloud iam service-accounts describe "$RUNTIME_SA" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${RUNTIME_SA%%@*}" --display-name "WorldWarPixel Cloud Run"
# Cloud Storage FUSE reads as the Cloud Storage service agent; objectAdmin covers it anyway.
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectAdmin >/dev/null

# Submit async and poll: `gcloud builds submit`'s log streaming needs project Viewer, which the
# deployer SA need not have; the build itself only needs Cloud Build Editor.
BUILD_ID="$(gcloud builds submit --tag "$IMAGE" --quiet --async --format="value(id)" .)"
echo "Cloud Build $BUILD_ID"
while :; do
  STATUS="$(gcloud builds describe "$BUILD_ID" --format="value(status)")"
  case "$STATUS" in
    SUCCESS) break ;;
    FAILURE|CANCELLED|TIMEOUT|INTERNAL_ERROR) echo "Cloud Build $BUILD_ID ended $STATUS" >&2; exit 1 ;;
  esac
  sleep 15
done

args=(
  --image "$IMAGE" --region "$REGION" --platform managed --allow-unauthenticated
  --port 8080 --cpu 1 --memory 2Gi --min-instances 1 --max-instances 1 --no-cpu-throttling
  --concurrency 250 --timeout 3600 --session-affinity --execution-environment gen2
  --add-volume "name=data,type=cloud-storage,bucket=$BUCKET"
  --add-volume-mount volume=data,mount-path=/data --service-account "$RUNTIME_SA"
)
# KEY=value lines -> the YAML gcloud expects; keeps the values out of the command line.
yaml="$(mktemp)"
trap 'rm -f "$yaml"' EXIT
grep -v '^\s*\(#\|$\)' "$APP_ENV_FILE" | sed -E 's/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/\1: "\2"/' > "$yaml"
args+=(--env-vars-file "$yaml")
gcloud run deploy "$SERVICE" "${args[@]}" --quiet
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
