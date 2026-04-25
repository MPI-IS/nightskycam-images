#!/usr/bin/env bash
#
# Reorganize nightskycam date directories between two roots:
#   - /data1/nightskycam/data/  : dates up to and including CUTOFF
#   - /data2/nightskycam/data/  : dates after CUTOFF
#
# Directories are moved to their correct root per the cutoff rule.
# If the target directory already exists (conflict), files are merged:
# each file is moved individually, and duplicates (same relative path)
# are deleted from the source since they are copies of the same data.
# The source directory is removed once empty.
#
# Phase 2 runs first (data2 -> data1) to free space on data2 before
# Phase 1 (data1 -> data2) fills it.
#
# Usage:
#   ./reorganize_roots.sh           # dry run (shows what would happen)
#   ./reorganize_roots.sh --execute # actually move/merge directories

set -euo pipefail

DATA1="/data1/nightskycam/data"
DATA2="/data2/nightskycam/data"
CUTOFF="2025_03"  # last year_month that belongs on data1

DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

# --- Helpers ---

is_after_cutoff() {
    local date_dir="$1"
    local year_month="${date_dir:0:7}"  # YYYY_MM
    [[ "$year_month" > "$CUTOFF" ]]
}

# Move an entire directory (used when no conflict).
move_dir() {
    local src="$1"
    local dst="$2"
    if $DRY_RUN; then
        echo "  [dry run] would move $src -> $dst"
    else
        echo "  moving $src -> $dst ..."
        mkdir -p "$(dirname "$dst")"
        mv "$src" "$dst"
    fi
}

# Merge src_dir into dst_dir (used when dst_dir already exists).
# - Files only in src: moved to dst.
# - Files in both (duplicates): source copy deleted.
# - After all files handled: src_dir is removed.
merge_dir() {
    local src_dir="$1"
    local dst_dir="$2"
    local moved=0
    local duplicates=0

    # Process top-level entries
    for entry in "$src_dir"/*; do
        [[ -e "$entry" ]] || continue
        local name
        name=$(basename "$entry")

        if [[ -d "$entry" ]]; then
            # Subdirectory (e.g. thumbnails/)
            if $DRY_RUN; then
                echo "    [dry run] would merge subdir $name/"
            else
                mkdir -p "$dst_dir/$name"
            fi
            for subfile in "$entry"/*; do
                [[ -f "$subfile" ]] || continue
                local subname
                subname=$(basename "$subfile")
                if [[ -e "$dst_dir/$name/$subname" ]]; then
                    if $DRY_RUN; then
                        echo "    [dry run] duplicate, would delete source: $name/$subname"
                    else
                        rm "$subfile"
                    fi
                    duplicates=$((duplicates + 1))
                else
                    if $DRY_RUN; then
                        echo "    [dry run] would move $name/$subname"
                    else
                        mv "$subfile" "$dst_dir/$name/$subname"
                    fi
                    moved=$((moved + 1))
                fi
            done
            if ! $DRY_RUN; then
                rmdir "$entry" || { echo "    ERROR: $entry not empty after merge"; exit 1; }
            fi
            continue
        fi

        # Regular file
        if [[ -e "$dst_dir/$name" ]]; then
            if $DRY_RUN; then
                echo "    [dry run] duplicate, would delete source: $name"
            else
                rm "$entry"
            fi
            duplicates=$((duplicates + 1))
        else
            if $DRY_RUN; then
                echo "    [dry run] would move $name"
            else
                mv "$entry" "$dst_dir/$name"
            fi
            moved=$((moved + 1))
        fi
    done

    # Source dir must now be empty
    if ! $DRY_RUN; then
        rmdir "$src_dir" || { echo "    ERROR: $src_dir not empty after merge"; exit 1; }
    fi

    echo "    merged: $moved moved, $duplicates duplicates removed"
}

# Move or merge a date directory to its destination.
move_or_merge() {
    local src="$1"
    local dst="$2"
    if [[ -d "$dst" ]]; then
        echo "  merging $src -> $dst ..."
        merge_dir "$src" "$dst"
    else
        move_dir "$src" "$dst"
    fi
}

# --- Main ---

echo "=== Reorganize nightskycam roots ==="
echo "  data1: $DATA1  (keeps <= $CUTOFF)"
echo "  data2: $DATA2  (keeps >  $CUTOFF)"
if $DRY_RUN; then
    echo "  mode:  DRY RUN (pass --execute to apply)"
else
    echo "  mode:  EXECUTE"
fi
echo ""

# --- Phase 2 first: data2 -> data1 (frees space on data2) ---
echo "--- Phase 2: move pre-cutoff dirs from data2 to data1 ---"
count2=0
merges2=0
for system_dir in "$DATA2"/*/; do
    [[ -d "$system_dir" ]] || continue
    system=$(basename "$system_dir")
    for date_dir in "$system_dir"*/; do
        [[ -d "$date_dir" ]] || continue
        date_name=$(basename "$date_dir")
        [[ "$date_name" =~ ^[0-9]{4}_[0-9]{2}_[0-9]{2}$ ]] || continue
        if ! is_after_cutoff "$date_name"; then
            dst="$DATA1/$system/$date_name"
            if [[ -d "$dst" ]]; then
                merges2=$((merges2 + 1))
            fi
            move_or_merge "$date_dir" "$dst"
            count2=$((count2 + 1))
        fi
    done
done
echo "  Phase 2 total: $count2 directories ($merges2 merged)"
echo ""

# --- Phase 1: data1 -> data2 (data2 now has free space) ---
echo "--- Phase 1: move post-cutoff dirs from data1 to data2 ---"
count1=0
merges1=0
for system_dir in "$DATA1"/*/; do
    [[ -d "$system_dir" ]] || continue
    system=$(basename "$system_dir")
    for date_dir in "$system_dir"*/; do
        [[ -d "$date_dir" ]] || continue
        date_name=$(basename "$date_dir")
        [[ "$date_name" =~ ^[0-9]{4}_[0-9]{2}_[0-9]{2}$ ]] || continue
        if is_after_cutoff "$date_name"; then
            dst="$DATA2/$system/$date_name"
            if [[ -d "$dst" ]]; then
                merges1=$((merges1 + 1))
            fi
            move_or_merge "$date_dir" "$dst"
            count1=$((count1 + 1))
        fi
    done
done
echo "  Phase 1 total: $count1 directories ($merges1 merged)"
echo ""

echo "=== Done. Phase 2 (data2->data1): $count2 ($merges2 merged), Phase 1 (data1->data2): $count1 ($merges1 merged) ==="
