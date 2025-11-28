import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------
# 1. Paths
# -------------------------------------
original_base = "/Users/clairesheehan/Documents/learning_python/TIA_project/kagglecatsanddogs_3367a/PetImages"   # e.g. "C:/Users/you/datasets/cats_dogs"

# Your dataset might use either: cats/ dogs/
# or: Cat/ Dog/

cat_dir = os.path.join(original_base, "Cat")
dog_dir = os.path.join(original_base, "Dog")

print("Cats directory:", cat_dir)
print("Dogs directory:", dog_dir)

# -------------------------------------
# 2. Create new structured dataset
# -------------------------------------
base_dir = "/Users/clairesheehan/Documents/learning_python/TIA_project/cats-vs-dogs-processed"

subdirs = [
    "train/cats", "train/dogs",
    "val/cats",   "val/dogs",
    "test/cats",  "test/dogs"
]

for sd in subdirs:
    os.makedirs(os.path.join(base_dir, sd), exist_ok=True)

# -------------------------------------
# 3. Function to split and copy data
# -------------------------------------
def split_and_copy(src_dir, train_dir, val_dir, test_dir, train_split=0.8, val_split=0.1):
    files = [f for f in os.listdir(src_dir) if os.path.getsize(os.path.join(src_dir, f)) > 0]
    random.shuffle(files)

    train_end = int(train_split * len(files))
    val_end = int((train_split + val_split) * len(files))

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Copy files
    for fname in train_files:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(train_dir, fname))
    for fname in val_files:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(val_dir, fname))
    for fname in test_files:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(test_dir, fname))

    print(f"{src_dir}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

# -------------------------------------
# 4. Perform the split
# -------------------------------------
split_and_copy(cat_dir,
               f"{base_dir}/train/cats",
               f"{base_dir}/val/cats",
               f"{base_dir}/test/cats")

split_and_copy(dog_dir,
               f"{base_dir}/train/dogs",
               f"{base_dir}/val/dogs",
               f"{base_dir}/test/dogs")

# -------------------------------------
# 5. ImageDataGenerators
# -------------------------------------
train_gen = ImageDataGenerator(rescale=1/255.)
val_gen   = ImageDataGenerator(rescale=1/255.)
test_gen  = ImageDataGenerator(rescale=1/255.)

train_generator = train_gen.flow_from_directory(
    f"{base_dir}/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

val_generator = val_gen.flow_from_directory(
    f"{base_dir}/val",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

test_generator = test_gen.flow_from_directory(
    f"{base_dir}/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)
