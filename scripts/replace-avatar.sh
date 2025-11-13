#!/bin/bash
# Helper script to replace and optimize the avatar photo
# Usage: ./scripts/replace-avatar.sh [path-to-new-photo.jpg]

set -e

AVATAR_DIR="content/authors/admin"
AVATAR_FILE="$AVATAR_DIR/avatar.jpg"
BACKUP_FILE="$AVATAR_DIR/avatar.backup.jpg"

# Check if new photo path provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/replace-avatar.sh [path-to-new-photo.jpg]"
    echo ""
    echo "This script will:"
    echo "  1. Backup current avatar.jpg"
    echo "  2. Copy and optimize your new photo"
    echo "  3. Replace avatar.jpg with optimized version"
    echo ""
    echo "Requirements:"
    echo "  - Photo should be square (1:1 aspect ratio)"
    echo "  - Minimum size: 400×400px"
    echo "  - Recommended: 800-2000px square"
    exit 1
fi

NEW_PHOTO="$1"

# Check if new photo exists
if [ ! -f "$NEW_PHOTO" ]; then
    echo "Error: File not found: $NEW_PHOTO"
    exit 1
fi

# Backup current avatar
if [ -f "$AVATAR_FILE" ]; then
    echo "Backing up current avatar..."
    cp "$AVATAR_FILE" "$BACKUP_FILE"
    echo "✓ Backup created: $BACKUP_FILE"
fi

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to optimize photo..."
    convert "$NEW_PHOTO" \
        -resize 1000x1000^ \
        -gravity center \
        -extent 1000x1000 \
        -quality 85 \
        -strip \
        "$AVATAR_FILE"
    echo "✓ Photo optimized and saved to $AVATAR_FILE"
elif command -v sips &> /dev/null; then
    echo "Using sips (macOS) to optimize photo..."
    # sips doesn't support square crop easily, so we'll just resize
    sips -Z 1000 "$NEW_PHOTO" --out "$AVATAR_FILE" &> /dev/null
    echo "✓ Photo resized and saved to $AVATAR_FILE"
    echo "⚠ Note: Please ensure photo is square (1:1 aspect ratio)"
else
    echo "No image optimization tools found. Copying file as-is..."
    cp "$NEW_PHOTO" "$AVATAR_FILE"
    echo "✓ Photo copied to $AVATAR_FILE"
    echo "⚠ Warning: Photo may not be optimized. Consider using TinyPNG.com"
fi

# Get file size
FILE_SIZE=$(ls -lh "$AVATAR_FILE" | awk '{print $5}')
echo ""
echo "New avatar file size: $FILE_SIZE"
echo ""
echo "Next steps:"
echo "  1. Test locally: hugo server --disableFastRender"
echo "  2. Verify photo displays correctly on homepage"
echo "  3. Check in both light and dark modes"
echo "  4. Verify photo is square (not stretched)"
echo ""
echo "If photo looks good, commit the changes:"
echo "  git add $AVATAR_FILE"
echo "  git commit -m 'Update professional avatar photo'"

