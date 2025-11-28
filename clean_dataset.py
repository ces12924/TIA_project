import os
from PIL import Image

root_dir = "/Users/clairesheehan/Documents/learning_python/TIA_project/cats-vs-dogs-processed"

bad_files = []

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
            path = os.path.join(subdir, file)
            try:
                with Image.open(path) as img:
                    img.load()
                    if img.mode not in ("RGB", "L", "RGBA"):
                        print(f"❌ Weird mode: {img.mode} → {path}")
                        bad_files.append(path)
                    else:
                        # also check number of channels explicitly
                        if len(img.getbands()) not in (1, 3, 4):
                            print(f"❌ Bad channel count ({len(img.getbands())}) → {path}")
                            bad_files.append(path)
            except Exception as e:
                print(f"❌ Corrupted: {path} ({e})")
                bad_files.append(path)

print("\nFound", len(bad_files), "bad images.")

# remove them
for path in bad_files:
    try:
        os.remove(path)
        print("Deleted:", path)
    except:
        pass
