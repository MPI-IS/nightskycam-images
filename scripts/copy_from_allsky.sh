#!/usr/bin/env bash
#
# Copy pre-2026 date directories from /is/projects/allsky/images
# to /data1/nightskycam/data/ that are missing from the destination.
#
# CRITICAL: this script only COPIES, never moves or deletes.
# The source (/is/projects/allsky/images) is never modified.
#
# Usage:
#   ./copy_from_allsky.sh           # dry run
#   ./copy_from_allsky.sh --execute # actually copy

set -euo pipefail

SRC="/is/projects/allsky/images"
DST="/data1/nightskycam/data"
CUTOFF="2026_01"  # first year_month to EXCLUDE (copy everything before 2026)

DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

is_before_cutoff() {
    local date_name="$1"
    local year_month="${date_name:0:7}"
    [[ "$year_month" < "$CUTOFF" ]]
}

echo "=== Copy pre-2026 images from allsky to data1 ==="
echo "  source:      $SRC  (READ ONLY — nothing will be modified here)"
echo "  destination: $DST"
if $DRY_RUN; then
    echo "  mode:        DRY RUN (pass --execute to apply)"
else
    echo "  mode:        EXECUTE (copy only, source is never modified)"
fi
echo ""

count=0
skipped=0
for system_dir in "$SRC"/*/; do
    [[ -d "$system_dir" ]] || continue
    system=$(basename "$system_dir")
    for date_dir in "$system_dir"*/; do
        [[ -d "$date_dir" ]] || continue
        date_name=$(basename "$date_dir")
        [[ "$date_name" =~ ^[0-9]{4}_[0-9]{2}_[0-9]{2}$ ]] || continue

        if ! is_before_cutoff "$date_name"; then
            continue
        fi

        dst_dir="$DST/$system/$date_name"
        if [[ -d "$dst_dir" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        if $DRY_RUN; then
            echo "  [dry run] would copy $date_dir -> $dst_dir"
        else
            echo "  copying $date_dir -> $dst_dir ..."
            mkdir -p "$DST/$system"
            cp -a "$date_dir" "$dst_dir"
        fi
        count=$((count + 1))
    done
done

echo ""
echo "=== Done. Copied: $count directories, already present: $skipped ==="
