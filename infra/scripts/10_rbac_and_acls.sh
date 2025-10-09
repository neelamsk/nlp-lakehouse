#!/usr/bin/env bash
set -euo pipefail
# See README header in this script for usage.

: "${SUB_ID:?}"
: "${RG:?}"
: "${ST:?}"
: "${FS:?}"
: "${ACC_RG:?}"
: "${ACC_NAME:?}"
USER_ROLE="${USER_ROLE:-Storage Blob Data Owner}"

az account set --subscription "$SUB_ID"
SCOPE="$(az storage account show -n "$ST" -g "$RG" --query id -o tsv)"
ME="$(az ad signed-in-user show --query id -o tsv)"
az role assignment create --assignee "$ME" --role "$USER_ROLE" --scope "$SCOPE" 1>/dev/null || true

CONNECTOR_OID="$(az resource show -g "$ACC_RG" -n "$ACC_NAME"   --resource-type "Microsoft.Databricks/accessConnectors"   --query identity.principalId -o tsv)"
az role assignment create   --assignee-object-id "$CONNECTOR_OID"   --assignee-principal-type ServicePrincipal   --role "Storage Blob Data Contributor"   --scope "$SCOPE" 1>/dev/null || true

sleep 45

az storage fs directory create --account-name "$ST" --file-system "$FS" --name uc --auth-mode login 1>/dev/null || true
az storage fs directory create --account-name "$ST" --file-system "$FS" --name uc/nlp_dev --auth-mode login 1>/dev/null || true

az storage fs access set --account-name "$ST" --file-system "$FS"   --acl "user:${ME}:rwx,group::r-x,other::---" --auth-mode login 1>/dev/null || true
az storage fs access set --account-name "$ST" --file-system "$FS"   --acl "default:user:${ME}:rwx,default:group::r-x,default:other::---" --auth-mode login 1>/dev/null || true

for p in "" "uc" "uc/nlp_dev"; do
  az storage fs access set --account-name "$ST" --file-system "$FS"     --path "$p"     --acl "user:${CONNECTOR_OID}:rwx,group::r-x,other::---" --auth-mode login 1>/dev/null || true
  az storage fs access set --account-name "$ST" --file-system "$FS"     --path "$p"     --acl "default:user:${CONNECTOR_OID}:rwx,default:group::r-x,default:other::---" --auth-mode login 1>/dev/null || true
done

echo "OK. Connector OID: $CONNECTOR_OID"
