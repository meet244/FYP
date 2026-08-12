#!/usr/bin/env python3

import os
import re
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("/Volumes/Meet_SSD/HDD/photos/fam")

# Files/folders beginning with dumpYYYY are ignored during scanning
DUMP_PATTERN = re.compile(r"^dump\d{4}$", re.IGNORECASE)

# Media extensions to process
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".webp", ".gif", ".tif", ".tiff",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv",
    ".3gp", ".mts", ".m2ts"
}


# ============================================================
# DATE PARSING
# ============================================================

def normalize_datetime(dt):
    """
    Convert a datetime to timezone-naive local time.

    This prevents:
        can't compare offset-naive and offset-aware datetimes

    from occurring when comparing dates obtained from different
    metadata sources.
    """
    if dt is None:
        return None

    if dt.tzinfo is not None:
        # Convert timezone-aware datetime to local time,
        # then remove timezone information.
        dt = dt.astimezone().replace(tzinfo=None)

    return dt


def parse_date(value):
    """
    Convert an ExifTool date string into a datetime.

    Handles examples such as:
        2019:05:21 08:32:26
        2019:05:21 08:32:26+05:30
        2019:05:21 08:32:26-04:00
        2019-05-21 08:32:26
        2019-05-21T08:32:26+05:30
    """

    if not value:
        return None

    if isinstance(value, datetime):
        return normalize_datetime(value)

    value = str(value).strip()

    if not value:
        return None

    # Remove fractional seconds if necessary while preserving timezone
    # Python's fromisoformat can handle them.
    candidates = [
        value,
        value.replace(" ", "T"),
        value.replace(":", "-", 2),
    ]

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return normalize_datetime(dt)
        except (ValueError, TypeError):
            pass

    # ExifTool's common format:
    # YYYY:MM:DD HH:MM:SS
    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y:%m:%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]

    for fmt in formats:
        try:
            return normalize_datetime(
                datetime.strptime(value, fmt)
            )
        except ValueError:
            pass

    # Sometimes ExifTool has a timezone suffix that strptime
    # doesn't handle with the formats above.
    timezone_match = re.match(
        r"^(.*?)([+-]\d{2}:\d{2})$",
        value
    )

    if timezone_match:
        base = timezone_match.group(1).strip()
        timezone = timezone_match.group(2)

        try:
            dt = datetime.fromisoformat(
                base.replace(" ", "T") + timezone
            )
            return normalize_datetime(dt)
        except ValueError:
            pass

    return None


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_media_files(source_dir):
    """
    Recursively find all supported media files.

    dumpYYYY directories are skipped.
    """

    files = []

    for root, dirs, filenames in os.walk(source_dir):

        # Do not scan dumpYYYY folders
        dirs[:] = [
            d for d in dirs
            if not DUMP_PATTERN.match(d)
        ]

        for filename in filenames:

            path = Path(root) / filename

            if path.suffix.lower() in MEDIA_EXTENSIONS:
                files.append(path)

    return files


# ============================================================
# EXIFTOOL
# ============================================================

def get_all_metadata(files):
    """
    Run one ExifTool process for all files.

    Returns:
        {
            "/path/to/file.jpg": {...metadata...}
        }
    """

    if not files:
        return {}

    command = [
        "exiftool",
        "-json",
        "-time:all",
        "-s",
        "-G",
    ]

    command.extend(str(f) for f in files)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
    except FileNotFoundError:
        print("ERROR: ExifTool is not installed or not in PATH.")
        return {}

    if result.returncode != 0 and not result.stdout.strip():
        print("ERROR running ExifTool:")
        print(result.stderr)
        return {}

    try:
        metadata_list = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("ERROR: Could not parse ExifTool output.")
        return {}

    metadata = {}

    for item in metadata_list:

        source_file = item.get("SourceFile")

        if source_file:
            metadata[str(Path(source_file))] = item

    return metadata


# ============================================================
# FIND OLDEST DATE
# ============================================================

def get_oldest_date(file_path, metadata):
    """
    Find the oldest useful date for a file.

    Priority is determined from metadata dates. If no usable
    metadata date exists, fall back to filesystem dates.
    """

    dates = []

    # --------------------------------------------------------
    # Metadata date fields
    # --------------------------------------------------------

    date_fields = [
        "EXIF:DateTimeOriginal",
        "EXIF:CreateDate",
        "EXIF:ModifyDate",

        "QuickTime:CreateDate",
        "QuickTime:ModifyDate",
        "QuickTime:CreationDate",

        "XMP:CreateDate",
        "XMP:ModifyDate",
        "XMP:DateCreated",

        "IPTC:DateCreated",
        "IPTC:DigitalCreationDate",
        "IPTC:TimeCreated",

        "Composite:SubSecDateTimeOriginal",
        "Composite:DateTimeOriginal",

        "File:FileModifyDate",
        "File:FileCreateDate",
    ]

    for field in date_fields:

        value = metadata.get(field)

        if value is None:
            continue

        # ExifTool can occasionally return lists
        if isinstance(value, list):
            values = value
        else:
            values = [value]

        for item in values:

            dt = parse_date(item)

            if dt is not None:
                # EXTRA SAFETY:
                # Always normalize before putting the datetime
                # into the comparison list.
                dt = normalize_datetime(dt)

                dates.append(dt)

    # --------------------------------------------------------
    # Filesystem fallback
    # --------------------------------------------------------

    try:
        stat = file_path.stat()

        # macOS creation time
        if hasattr(stat, "st_birthtime"):
            birth_time = datetime.fromtimestamp(
                stat.st_birthtime
            )
            dates.append(
                normalize_datetime(birth_time)
            )

        # Modification time
        modify_time = datetime.fromtimestamp(
            stat.st_mtime
        )

        dates.append(
            normalize_datetime(modify_time)
        )

    except OSError as e:
        print(f"WARNING: Could not read filesystem dates: {file_path}")
        print(f"         {e}")

    # --------------------------------------------------------
    # Remove invalid dates
    # --------------------------------------------------------

    dates = [
        normalize_datetime(d)
        for d in dates
        if d is not None
    ]

    if not dates:
        return None, "unknown"

    # At this point EVERY datetime is guaranteed to be naive.
    oldest_date = min(dates)

    # --------------------------------------------------------
    # Determine source tag
    # --------------------------------------------------------

    # If metadata dates exist, identify this as metadata.
    metadata_dates = []

    for field in date_fields:
        value = metadata.get(field)

        if value is None:
            continue

        values = value if isinstance(value, list) else [value]

        for item in values:
            dt = parse_date(item)

            if dt is not None:
                metadata_dates.append(
                    normalize_datetime(dt)
                )

    if metadata_dates:
        return oldest_date, "metadata"

    return oldest_date, "filesystem"


# ============================================================
# DESTINATION
# ============================================================

def get_destination_directory(source_dir, year):
    """
    Return dumpYYYY directory.
    """

    return source_dir / f"dump{year}"


def unique_destination(destination):
    """
    Prevent overwriting an existing file.

    Example:
        IMG_1234.jpg
        IMG_1234_1.jpg
        IMG_1234_2.jpg
    """

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    counter = 1

    while True:

        new_destination = (
            destination.parent /
            f"{stem}_{counter}{suffix}"
        )

        if not new_destination.exists():
            return new_destination

        counter += 1


# ============================================================
# MOVE FILE
# ============================================================

def move_file(file_path, oldest_date):
    """
    Move file into dumpYYYY based on its oldest date.
    """

    if oldest_date is None:
        print(f"    WARNING: No date found: {file_path}")
        return

    year = oldest_date.year

    destination_dir = get_destination_directory(
        SOURCE_DIR,
        year
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = destination_dir / file_path.name

    destination = unique_destination(destination)

    try:

        print()
        print(f"    Date:   {oldest_date}")
        print(f"    Target: {destination}")

        shutil.move(
            str(file_path),
            str(destination)
        )

        print("    MOVED")

    except Exception as e:

        print(
            f"    ERROR moving {file_path}: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHOTO METADATA ORGANIZER")
    print("=" * 70)

    print()
    print(f"Scanning: {SOURCE_DIR}")
    print()

    if not SOURCE_DIR.exists():
        print(
            f"ERROR: Source directory does not exist:\n"
            f"{SOURCE_DIR}"
        )
        return

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = get_media_files(SOURCE_DIR)

    print(f"Found {len(files):,} media files.")

    if not files:
        print("Nothing to process.")
        return

    # --------------------------------------------------------
    # Get metadata
    # --------------------------------------------------------

    print()
    print("Reading metadata with ExifTool...")
    print()

    metadata = get_all_metadata(files)

    print(
        f"Metadata loaded for "
        f"{len(metadata):,} files."
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    processed = 0
    moved = 0
    failed = 0

    for index, file_path in enumerate(files, start=1):

        print(
            f"[{index:,}/{len(files):,}] "
            f"{file_path}"
        )

        try:

            file_metadata = metadata.get(
                str(file_path),
                {}
            )

            oldest_date, source_tag = get_oldest_date(
                file_path,
                file_metadata
            )

            if oldest_date is None:

                print(
                    "    WARNING: Could not determine date."
                )

                failed += 1
                continue

            print(
                f"    Oldest date: "
                f"{oldest_date} "
                f"({source_tag})"
            )

            move_file(
                file_path,
                oldest_date
            )

            moved += 1

        except KeyboardInterrupt:

            print()
            print()
            print("Interrupted by user.")
            break

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

            failed += 1

        processed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Processed: {processed:,}")
    print(f"Moved:     {moved:,}")
    print(f"Failed:    {failed:,}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()