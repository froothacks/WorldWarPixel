#!/usr/bin/env bash
# Build the image with Cloud Build and deploy one always-on Cloud Run instance, with the timelapse
# bucket mounted at /data (SQLite backups + rendered videos land straight in GCS). Idempotent; run
# locally with gcloud authenticated or from .github/workflows/deploy.yml.
#
#   APP_ENV_FILE=/path/outside/repo/app.env deploy/cloudrun.sh
#
# Only the app secrets allowlisted below are read from Secret Manager by name and exposed as env
# vars at version latest. Adding one requires an APP_SECRETS entry here and a Secret Manager secret.
# APP_ENV_FILE is optional and only for non-secret settings (KEY=value lines). Keep it outside the
# repo so `gcloud builds submit` never uploads it with the build context.
# One-time IAM prerequisites (granted by a project owner; this script does not grant them):
#   runtime SA worldwarpixel-run@ needs roles/secretmanager.secretAccessor on the project
#   deployer SA github-deployer@ needs roles/secretmanager.viewer on the project
# GOD_TOKEN is optional: without it god mode is open to anyone with the URL, same as the Modal deploy.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-world-war-pixel-514101}"
REGION="${GCP_REGION:-us-west1}"
SERVICE="${CLOUD_RUN_SERVICE:-worldwarpixel}"
BUCKET="${TIMELAPSE_BUCKET:-world-war-pixel-timelapses-514101}"
REPO="${ARTIFACT_REPO:-worldwarpixel}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/web:$(git rev-parse --short HEAD)"
CACHE_REPO="$REGION-docker.pkg.dev/$PROJECT/$REPO/cache"
RUNTIME_SA="${CLOUD_RUN_SA:-worldwarpixel-run}@$PROJECT.iam.gserviceaccount.com"
# Cloud SQL instance the app connects to via DATABASE_URL (Unix socket at /cloudsql/...);
# set CLOUDSQL_INSTANCE="" to deploy without it (SQLite event log).
CLOUDSQL_INSTANCE="${CLOUDSQL_INSTANCE-worldwarpixel-pg}"

# App secrets Cloud Run reads from Secret Manager (secret name = env var name). Anything else in
# the project stays out. Model endpoint secrets must match models.py (tests/test_deploy_secrets.py).
APP_SECRETS=(
  OPENAI_API_KEY
  TTS_API_KEY
  MODAL_KEY MODAL_SECRET
  CHRYZTOPH_MODAL_KEY CHRYZTOPH_MODAL_SECRET
  MEGANWCHUN_MODAL_KEY MEGANWCHUN_MODAL_SECRET
  GOD_TOKEN DATABASE_URL DEVICE_TOKENS
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
)

if [ -n "${APP_ENV_FILE:-}" ]; then
  case "$(realpath "$APP_ENV_FILE")" in "$PWD"/*) echo "APP_ENV_FILE must be outside the repo" >&2; exit 1;; esac
fi

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
if ! secret_names="$(gcloud secrets list --project "$PROJECT" --format='value(name.basename())')"; then
  echo "Unable to list Secret Manager secrets in $PROJECT; the deployer SA needs roles/secretmanager.viewer on the project." >&2
  exit 1
fi
declare -A existing_secrets=()
while IFS= read -r secret_name; do
  if [[ -n "$secret_name" ]]; then
    existing_secrets["$secret_name"]=1
  fi
done <<< "$secret_names"
secret_args=""
secret_display=""
for secret_name in "${APP_SECRETS[@]}"; do
  if [ "$secret_name" = DATABASE_URL ] && [ -z "$CLOUDSQL_INSTANCE" ]; then
    echo "CLOUDSQL_INSTANCE is empty; DATABASE_URL not mounted (SQLite event log)"
    continue
  fi
  if [[ -z "${existing_secrets[$secret_name]+x}" ]]; then
    echo "Secret Manager has no $secret_name; skipped"
    continue
  fi
  if ! enabled_version="$(gcloud secrets versions list "$secret_name" \
    --project "$PROJECT" --filter='state=ENABLED' --limit 1 --format='value(name)')"; then
    echo "Unable to list Secret Manager secrets in $PROJECT; the deployer SA needs roles/secretmanager.viewer on the project." >&2
    exit 1
  fi
  if [ -z "$enabled_version" ]; then
    echo "$secret_name exists in Secret Manager but has no enabled version; add a version or delete the secret" >&2
    exit 1
  fi
  secret_args+="${secret_args:+,}${secret_name}=${secret_name}:latest"
  secret_display+="${secret_display:+, }${secret_name}"
done
if [ -n "$CLOUDSQL_INSTANCE" ] && [[ ",$secret_args," != *",DATABASE_URL=DATABASE_URL:latest,"* ]]; then
  echo "CLOUDSQL_INSTANCE=$CLOUDSQL_INSTANCE needs DATABASE_URL in Secret Manager (or set CLOUDSQL_INSTANCE=\"\" for SQLite)" >&2
  exit 1
fi

gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --location="$REGION" --repository-format=docker
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

# The app runs as its own service account; bucket access is granted below.
gcloud iam service-accounts describe "$RUNTIME_SA" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${RUNTIME_SA%%@*}" --display-name "WorldWarPixel Cloud Run"
# Cloud Storage FUSE reads as the Cloud Storage service agent; objectAdmin covers it anyway.
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectAdmin >/dev/null

# Submit async and poll: `gcloud builds submit`'s log streaming needs project Viewer, which the
# deployer SA need not have; the build itself only needs Cloud Build Editor.
BUILD_ID="$(gcloud builds submit --config cloudbuild.yaml \
  --substitutions="_IMAGE=$IMAGE,_CACHE_REPO=$CACHE_REPO" \
  --quiet --async --format="value(id)" .)"
echo "Cloud Build $BUILD_ID"
while :; do
  STATUS="$(gcloud builds describe "$BUILD_ID" --format="value(status)")"
  case "$STATUS" in
    SUCCESS) break ;;
    FAILURE|CANCELLED|TIMEOUT|INTERNAL_ERROR) echo "Cloud Build $BUILD_ID ended $STATUS" >&2; exit 1 ;;
  esac
  sleep 5
done

DATA_VOLUME="name=data,type=cloud-storage,bucket=$BUCKET"
DATA_VOLUME_MOUNT="volume=data,mount-path=/data"
args=(
  --image "$IMAGE" --region "$REGION" --platform managed --allow-unauthenticated
  --port 8080 --cpu 4 --memory 8Gi --min-instances 1 --max-instances 1 --no-cpu-throttling
  --concurrency 1000 --timeout 3600 --session-affinity --execution-environment gen2 --cpu-boost
  --add-volume "$DATA_VOLUME"
  --add-volume-mount "$DATA_VOLUME_MOUNT" --service-account "$RUNTIME_SA"
)
if [ -n "$CLOUDSQL_INSTANCE" ]; then
  args+=(--add-cloudsql-instances "$PROJECT:$REGION:$CLOUDSQL_INSTANCE")
fi
# KEY=value lines -> the YAML gcloud expects; Secret Manager wins for same-named keys.
app_secret_names="$(printf '%s\n' "${APP_SECRETS[@]}")"
yaml="$(mktemp)"
filtered_env="$(mktemp)"
warnings="$(mktemp)"
trap 'rm -f "$yaml" "$filtered_env" "$warnings"' EXIT
if [ -n "$secret_args" ]; then
  args+=(--set-secrets "$secret_args")
  echo "Secrets from Secret Manager: $secret_display"
else
  echo "Secrets from Secret Manager: (none)"
fi
if [ -n "${APP_ENV_FILE:-}" ]; then
  awk -v app_secret_names="$app_secret_names" -v secret_names="$secret_names" \
    -v warnings_file="$warnings" '
    BEGIN {
      count = split(app_secret_names, app_names, "\n")
      for (i = 1; i <= count; i++) {
        if (app_names[i] != "") app_secret_keys[app_names[i]] = 1
      }
      count = split(secret_names, names, "\n")
      for (i = 1; i <= count; i++) {
        if (names[i] != "") existing_secret_keys[names[i]] = 1
      }
    }
    !/^[[:space:]]*(#|$)/ {
      if ($0 ~ /^[A-Za-z_][A-Za-z0-9_]*=/) {
        key = $0
        sub(/=.*/, "", key)
        if (key in app_secret_keys || key in existing_secret_keys) {
          print key >> warnings_file
          next
        }
      }
      print
    }
  ' "$APP_ENV_FILE" > "$filtered_env"
  while IFS= read -r key; do
    if [[ -n "$key" ]]; then
      echo "::warning::$key from APP_ENV_FILE ignored; secrets come from Secret Manager"
    fi
  done < "$warnings"
  sed -E 's/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/\1: "\2"/' "$filtered_env" > "$yaml"
fi
if [ -s "$yaml" ]; then
  args+=(--env-vars-file "$yaml")
else
  args+=(--clear-env-vars)
fi
gcloud run deploy "$SERVICE" "${args[@]}" --quiet
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
