#!/usr/bin/env python3
"""
create_data_folder.py

Reads all test cases from every R script and populates the data/ folder.
Each test case gets a numbered subfolder (1, 2, 3, ...) containing:
  - original_image.<ext>   (the "before" image)
  - edited_image.<ext>     (the "after" image)
  - edit_instruction.txt   (the instruction text)
  - test_description.txt   (the description text)
"""

import os
import shutil
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LGIE_DATA = os.path.join(BASE_DIR, "lgie-eval-data")

def resolve_path(p):
    """Resolve a test case path relative to BASE_DIR."""
    if os.path.isabs(p):
        return p
    return os.path.join(BASE_DIR, p)

def copy_image(src_path, dst_dir, dst_name_prefix):
    """Copy an image to dst_dir with the given prefix, preserving extension."""
    src = resolve_path(src_path)
    if not os.path.exists(src):
        # Try alternate extensions
        base, ext = os.path.splitext(src)
        for alt_ext in ['.jpg', '.jpeg', '.png']:
            alt = base + alt_ext
            if os.path.exists(alt):
                src = alt
                break
        else:
            print(f"  WARNING: Image not found: {src_path}")
            return None
    ext = os.path.splitext(src)[1]
    dst = os.path.join(dst_dir, dst_name_prefix + ext)
    shutil.copy2(src, dst)
    return dst

# =====================================================================
# Define ALL test cases from ALL R scripts
# Each entry: (r_script, before, after, instruction, description)
# =====================================================================
all_test_cases = []

# ----- R1: Unmentioned Stationary Objects -----
r1_tests = [
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_2.png",
        "instruction": "The robotic arm removes the spoon, the yellow brush, the banana, and the blue cube from the scene",
        "description": "Spoon, banana and the blue cube moves in the video but the yellow brush remains stationary",
    },
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_2.png",
        "instruction": "The robotic arm removes the spoon, the banana, and the blue cube from the scene",
        "description": "Spoon, banana and the blue cube moves in the video.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_4.png",
        "after": "lgie-eval-data/r1_gemini_output/out_4.png",
        "instruction": "close white1fbox flap",
        "description": "White box flap is closed -> no mentioned objects are stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_8.png",
        "after": "lgie-eval-data/r1_gemini_output/out_8.png",
        "instruction": "turn lever vertical to front",
        "description": "Lever above the sink stays stationary and unturned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_11.png",
        "after": "lgie-eval-data/r1_gemini_output/out_11.png",
        "instruction": "close microwave",
        "description": "The microwave door stayed open/stationary even though instruction clearly mentions it's movement.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_17.png",
        "after": "lgie-eval-data/r1_gemini_output/out_17.png",
        "instruction": "put corn in pot which is in sink",
        "description": "The mentioned corn is stationary but it should have been placed in the pot within the sink.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_20.png",
        "after": "lgie-eval-data/r1_gemini_output/out_20.png",
        "instruction": "open oven",
        "description": "There is an hallucinated opened microwave door but the edit failed to remove the closed microwave door.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_70.png",
        "after": "lgie-eval-data/r1_gemini_output/out_70.png",
        "instruction": "put lid on stove",
        "description": "Another lid occured but edit failed to remove the original lid placed within sink from the image.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_71.png",
        "after": "lgie-eval-data/r1_gemini_output/out_71.png",
        "instruction": "take carrot off plate",
        "description": "Mentioned carrot stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_73.png",
        "after": "lgie-eval-data/r1_gemini_output/out_73.png",
        "instruction": "put pan in sink",
        "description": "Mentioned pan is moved correctly to the sink.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_77.png",
        "after": "lgie-eval-data/r1_gemini_output/out_77.png",
        "instruction": "put detergent in sink",
        "description": "Mentioned white detergent bottle placed beside the sink stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_80.png",
        "after": "lgie-eval-data/r1_gemini_output/out_80.png",
        "instruction": "put clothes in laundry machine",
        "description": "Mentioned clothes have been put in the washing machine but the edit failed to remove clothes from previous place.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_82.png",
        "after": "lgie-eval-data/r1_gemini_output/out_82.png",
        "instruction": "put sweet potato in pot which is in sink",
        "description": "Mentioned red sweet potato stays staionary in the scene but unmentioned objects have moved in the edit.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_84.png",
        "after": "lgie-eval-data/r1_gemini_output/out_84.png",
        "instruction": "put carrot on cutting board",
        "description": "Mentioned carrot has been correctly placed on cutting board.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_87.png",
        "after": "lgie-eval-data/r1_gemini_output/out_87.png",
        "instruction": "put sweet potato in pan which is on stove",
        "description": "Mentioned red sweet potato stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_144.png",
        "after": "lgie-eval-data/r1_gemini_output/out_144.png",
        "instruction": "pick up bowl and put in small4fbox",
        "description": "Mentioned blue bowl near the small cardboard box stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_147.png",
        "after": "lgie-eval-data/r1_gemini_output/out_147.png",
        "instruction": "put broccoli in pot",
        "description": "Mentioned broccoli stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_159.png",
        "after": "lgie-eval-data/r1_gemini_output/out_159.png",
        "instruction": "flip pot upright in sink",
        "description": "Pot in sink has been rotated and placed correctly.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_187.png",
        "after": "lgie-eval-data/r1_gemini_output/out_187.png",
        "instruction": "put sweet potato in pot which is in sink",
        "description": "A sweet potato has been put in the pot but edit failed to delete/remove the sweet potato from the original location.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_194.png",
        "after": "lgie-eval-data/r1_gemini_output/out_194.png",
        "instruction": "put red bottle in sink",
        "description": "Mentioned red bottle stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_197.png",
        "after": "lgie-eval-data/r1_gemini_output/out_197.png",
        "instruction": "turn lever vertical to front",
        "description": "Lever is unturned and remains stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_198.png",
        "after": "lgie-eval-data/r1_gemini_output/out_198.png",
        "instruction": "put pepper in pan",
        "description": "Mentioned pepper (red one within the kitchen sink) is stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_202.png",
        "after": "lgie-eval-data/r1_gemini_output/out_202.png",
        "instruction": "put knife in pot cardboard fence",
        "description": "Mentioned knife stays stationary.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_203.png",
        "after": "lgie-eval-data/r1_gemini_output/out_203.png",
        "instruction": "take broccoli out of pan",
        "description": "Pan has moved but brocolli is still within the pan thus mentioned broccoli is stationary relative to mentioned pan.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_204.png",
        "after": "lgie-eval-data/r1_gemini_output/out_204.png",
        "instruction": "put sweet potato in pan which is on stove",
        "description": "Mentioned sweet potato (red in color) is stationary.",
    },
]
for tc in r1_tests:
    tc["r_script"] = "R1"

# ----- R2: Unmentioned Moved-Within-Scene Objects -----
r2_tests = [
    {
        "before": "lgie-eval-data/input_2.jpg",
        "after": "lgie-eval-data/output_2.png",
        "instruction": "The robotic arm puts the omelette on top of the cabbage",
        "description": "Omelette on cabbage - cabbage also moved",
    },
    {
        "before": "lgie-eval-data/input_1.jpg",
        "after": "lgie-eval-data/output_1_1.png",
        "instruction": "The robotic arm puts the cucumber on top of the blue cube",
        "description": "Cucumber on blue cube - no unmentioned moves",
    },
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_1.png",
        "instruction": "The robotic arm puts the sushi into the pot",
        "description": "Sushi into pot - strawberry also moved",
    },
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_1.png",
        "instruction": "The robotic arm puts the carrot into the pot",
        "description": "Carrot into pot (wrong) - sushi and strawberry moved instead",
    },
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_1.png",
        "instruction": "The robotic arm moves the blue spoon out of the scene",
        "description": "Blue spoon is moved out of the scene - no unmentioned moves",
    },
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_2.png",
        "instruction": "The robotic arm moves the blue spoon out of the scene",
        "description": "Blue spoon, banana (unmentioned) and blue cube (unmentioned) are moved out of the scene - still no unmentioned moves within the scene",
    },
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_3.png",
        "instruction": "The robotic arm moves the blue spoon to the left edge of the table",
        "description": "Blue spoon (mentioned), banana (unmentioned) and the cucumber (unmentioned) are moved within the scene",
    },
    {
        "before": "lgie-eval-data/original_image/oi_23.png",
        "after": "lgie-eval-data/r2_gemini_output/out_23.png",
        "instruction": "pick up sponge and wipe plate",
        "description": "Sponge is picked up and seen wiping plate -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_24.png",
        "after": "lgie-eval-data/r2_gemini_output/out_24.png",
        "instruction": "put bowl on plate cardboard fence",
        "description": "Bowl is correctly place on plate but unmentioned green pepper is placed outside the cardboard fence",
    },
    {
        "before": "lgie-eval-data/original_image/oi_25.png",
        "after": "lgie-eval-data/r2_gemini_output/out_25.png",
        "instruction": "put detergent in sink",
        "description": "Detergent bottle is moved to the sink but unmentioned metal pot and red sushi also mistakenly moved within the sink",
    },
    {
        "before": "lgie-eval-data/original_image/oi_26.png",
        "after": "lgie-eval-data/r2_gemini_output/out_26.png",
        "instruction": "put banana in pot cardboard fence",
        "description": "Banana is correctly put in pot but whole pot is mistakenly moved to the bottom left corner within cardboard fence",
    },
    {
        "before": "lgie-eval-data/original_image/oi_27.png",
        "after": "lgie-eval-data/r2_gemini_output/out_27.png",
        "instruction": "put carrot in pot cardboard fence",
        "description": "Carrot is correctly put in pot but whole pot is mistakenly moved to the bottom left corner within cardboard fence",
    },
    {
        "before": "lgie-eval-data/original_image/oi_28.png",
        "after": "lgie-eval-data/r2_gemini_output/out_28.png",
        "instruction": "open microwave",
        "description": "Unmentioned orange is mistakenly moved within cardboard fence.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_29.png",
        "after": "lgie-eval-data/r2_gemini_output/out_29.png",
        "instruction": "open book",
        "description": "Book is opened but omlette which should be below the book is mistakenly put on top of the opened book.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_30.png",
        "after": "lgie-eval-data/r2_gemini_output/out_30.png",
        "instruction": "flip pot upright in sink",
        "description": "Pot is correctly flipped upright but unmentioned orange pumpkin is mistakenly moved within the pot.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_31.png",
        "after": "lgie-eval-data/r2_gemini_output/out_31.png",
        "instruction": "open brown1fbox flap",
        "description": "Unmentioned corn and beans box have been mistakenly moved to left of the brown box.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_32.png",
        "after": "lgie-eval-data/r2_gemini_output/out_32.png",
        "instruction": "take broccoli out of pan",
        "description": "Broccoli is correctly put out of pan -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_33.png",
        "after": "lgie-eval-data/r2_gemini_output/out_33.png",
        "instruction": "put corn in pot which is in sink",
        "description": "Corn is correctly put in the pot but unmentioned red pepper has been taken out from the sink and moved to kitchen surface.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_34.png",
        "after": "lgie-eval-data/r2_gemini_output/out_34.png",
        "instruction": "put small spoon from basket to tray",
        "description": "Small spoon has been correctly put on basket tray -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_35.png",
        "after": "lgie-eval-data/r2_gemini_output/out_35.png",
        "instruction": "pick up bit holder",
        "description": "Bit holder is lifted -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_36.png",
        "after": "lgie-eval-data/r2_gemini_output/out_36.png",
        "instruction": "turn lever vertical to front",
        "description": "Unmentioned onion, carrot, and knife have been taken out from the sink and mistakenly moved.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_37.png",
        "after": "lgie-eval-data/r2_gemini_output/out_37.png",
        "instruction": "take carrot off plate",
        "description": "Unmentioned plate have been mistakenly taken from sink moved to kitchen surface",
    },
    {
        "before": "lgie-eval-data/original_image/oi_38.png",
        "after": "lgie-eval-data/r2_gemini_output/out_38.png",
        "instruction": "lever vertical to front",
        "description": "No unmentioned moved-within-scene objects",
    },
    {
        "before": "lgie-eval-data/original_image/oi_39.png",
        "after": "lgie-eval-data/r2_gemini_output/out_39.png",
        "instruction": "put banana in colander",
        "description": "Banana has been correctly put in colander but unmentioned spoon, tomato, spatula, and green bell pepper have also been mistakenly put into the colander.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_40.png",
        "after": "lgie-eval-data/r2_gemini_output/out_40.png",
        "instruction": "put pear in bowl",
        "description": "Pear is correctly placed in bowl -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_41.png",
        "after": "lgie-eval-data/r2_gemini_output/out_41.png",
        "instruction": "take carrot off plate",
        "description": "Carrot is removed from the plate -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_42.png",
        "after": "lgie-eval-data/r2_gemini_output/out_42.png",
        "instruction": "take carrot off plate",
        "description": "Carrot is correctly taken off the plate and placed aside -- no unmentioned moved-within-scene objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_43.png",
        "after": "lgie-eval-data/r2_gemini_output/out_43.png",
        "instruction": "close drawer of box",
        "description": "Unmentioned green pear inside the drawer mistakenly moved within the drawer.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_44.png",
        "after": "lgie-eval-data/r2_gemini_output/out_44.png",
        "instruction": "put pepper in pan",
        "description": "Pepper is placed in pan but unmentioned yellow bell pepper is also mistakenly put in pan.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_45.png",
        "after": "lgie-eval-data/r2_gemini_output/out_45.png",
        "instruction": "fold cloth in half",
        "description": "Cloth is folded in half -- no unmentioned moved-within-scene objects.",
    },
]
for tc in r2_tests:
    tc["r_script"] = "R2"

# ----- R3: Unmentioned Removed-from-Scene Objects -----
r3_tests = [
    {
        "before": "lgie-eval-data/input_2.jpg",
        "after": "lgie-eval-data/output_2.png",
        "instruction": "The robotic arm puts the omelette on top of the cabbage",
        "description": "Omelette on cabbage - no objects removed from scene",
    },
    {
        "before": "lgie-eval-data/input_1.jpg",
        "after": "lgie-eval-data/output_1_1.png",
        "instruction": "The robotic arm puts the cucumber on top of the blue cube",
        "description": "Cucumber on blue cube - no objects removed from scene",
    },
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_1.png",
        "instruction": "The robotic arm puts the sushi into the pot",
        "description": "Sushi into pot - no objects removed from scene",
    },
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_1.png",
        "instruction": "The robotic arm puts the carrot into the pot",
        "description": "Carrot into pot (wrong instruction) - no objects removed from scene",
    },
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_1.png",
        "instruction": "The robotic arm moves the blue spoon out of the scene",
        "description": "Blue spoon removed from scene (mentioned) - 0 unmentioned removals",
    },
    {
        "before": "lgie-eval-data/input_4.jpg",
        "after": "lgie-eval-data/output_4_2.png",
        "instruction": "The robotic arm moves the blue spoon out of the scene",
        "description": "Blue spoon, banana, blue cube removed - 2 unmentioned (banana, blue cube)",
    },
    {
        "before": "lgie-eval-data/original_image/oi_0.png",
        "after": "lgie-eval-data/r3_gemini_output/out_0.png",
        "instruction": "put carrot on plate",
        "description": "The instruction mentions manipulation of carrot and plate, but the model removed blue bowl which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_1.png",
        "after": "lgie-eval-data/r3_gemini_output/out_1.png",
        "instruction": "put broccoli in pot or pan",
        "description": "The instruction mentions manipulation of broccoli, and pot (pan) but the model removed carrot, blue bowl, green container, and red bell pepper which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_2.png",
        "after": "lgie-eval-data/r3_gemini_output/out_2.png",
        "instruction": "pick up pan from stove",
        "description": "The instruction mentions manipulation of pan and stove, but the model removed yellow bell pepper, orange pumpkin, and green plate which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_3.png",
        "after": "lgie-eval-data/r3_gemini_output/out_3.png",
        "instruction": "take sushi out of pan",
        "description": "The instruction mentions manipulation of sushi and pan, but the model removed corn which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_4.png",
        "after": "lgie-eval-data/r3_gemini_output/out_4.png",
        "instruction": "close white1fbox flap",
        "description": "The instruction mentions manipulation of white1fbox flap, but the model removed bean container which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_5.png",
        "after": "lgie-eval-data/r3_gemini_output/out_5.png",
        "instruction": "put fork from basket to tray",
        "description": "The instruction mentions manipulation of fork, basket, and tray, but the model removed robotic arm and gray basket which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_6.png",
        "after": "lgie-eval-data/r3_gemini_output/out_6.png",
        "instruction": "put red bottle in sink",
        "description": "The instruction mentions manipulation of red bottle and sink, but the model removed carrot and blue bowl which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_7.png",
        "after": "lgie-eval-data/r3_gemini_output/out_7.png",
        "instruction": "pour almonds in pot",
        "description": "The instruction mentions manipulation of almonds and pot, but the model removed red bell pepper which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_8.png",
        "after": "lgie-eval-data/r3_gemini_output/out_8.png",
        "instruction": "turn lever vertical to front",
        "description": "The instruction mentions manipulation of lever, but the model removed metal pot, red bell pepper, and green vegetable which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_9.png",
        "after": "lgie-eval-data/r3_gemini_output/out_9.png",
        "instruction": "put spoon on plate",
        "description": "The instruction mentions manipulation of spoon and plate, but the model removed fork and knife which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_10.png",
        "after": "lgie-eval-data/r3_gemini_output/out_10.png",
        "instruction": "pick up bit holder",
        "description": "The instruction mentions manipulation of bit holder, but the model removed big screwdriver on right and bit holder which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_11.png",
        "after": "lgie-eval-data/r3_gemini_output/out_11.png",
        "instruction": "close microwave",
        "description": "The instruction mentions manipulation of microwave, but the model removed red sushi which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_12.png",
        "after": "lgie-eval-data/r3_gemini_output/out_12.png",
        "instruction": "put banana in colander",
        "description": "The instruction mentions manipulation of banana and colander, but the model removed green cutting board which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_13.png",
        "after": "lgie-eval-data/r3_gemini_output/out_13.png",
        "instruction": "lever vertical to front",
        "description": "The instruction mentions manipulation of lever, and no unmentioned objects were removed.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_14.png",
        "after": "lgie-eval-data/r3_gemini_output/out_14.png",
        "instruction": "flip pot upright in sink",
        "description": "The instruction mentions manipulation of pot and sink, but the model removed red bell pepper and orange pumpkin which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_15.png",
        "after": "lgie-eval-data/r3_gemini_output/out_15.png",
        "instruction": "put fork from basket to tray",
        "description": "The instruction mentions manipulation of fork, basket, and tray, and no unmentioned objects were removed.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_16.png",
        "after": "lgie-eval-data/r3_gemini_output/out_16.png",
        "instruction": "flip pot upright in sink",
        "description": "The instruction mentions manipulation of pot and sink, but the model removed corn and yellow bell pepper which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_17.png",
        "after": "lgie-eval-data/r3_gemini_output/out_17.png",
        "instruction": "put corn in pot which is in sink",
        "description": "The instruction mentions manipulation of corn, pot, and sink, but the model removed green vegetable which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_18.png",
        "after": "lgie-eval-data/r3_gemini_output/out_18.png",
        "instruction": "put broccoli in bowl",
        "description": "The instruction mentions manipulation of broccoli and bowl, but the model removed red sauce bottle which is unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_19.png",
        "after": "lgie-eval-data/r3_gemini_output/out_19.png",
        "instruction": "pick up closest rainbow Allen key set",
        "description": "The instruction mentions manipulation of rainbow Allen key set, and no unmentioned objects were removed.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_20.png",
        "after": "lgie-eval-data/r3_gemini_output/out_20.png",
        "instruction": "open oven",
        "description": "The instruction mentions manipulation of oven, and no unmentioned objects were removed.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_21.png",
        "after": "lgie-eval-data/r3_gemini_output/out_21.png",
        "instruction": "put pot in sink",
        "description": "The instruction mentions manipulation of pot and sink, but the model removed green cutting board, green plate, yellow plastic knife, yellow bell pepper, corn, red bell pepper, lettuce, red chilly, and orange pumpkin which are unmentioned.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_22.png",
        "after": "lgie-eval-data/r3_gemini_output/out_22.png",
        "instruction": "take clothes out of laundry machine",
        "description": "The instruction mentions manipulation of clothes and laundry machine, but the model removed red cloth and blue cloth inside spin tub which are unmentioned.",
    },
]
for tc in r3_tests:
    tc["r_script"] = "R3"

# ----- R4: Unmentioned Hallucinated Objects -----
r4_tests = [
    {
        "before": "lgie-eval-data/input_3.jpg",
        "after": "lgie-eval-data/output_3_1.png",
        "instruction": "The robotic arm puts the sushi on top of the omelette",
        "description": "A hallucinated sushi is put on top of a hallucinated omelette. The originals remain at their original places respectively",
    },
    {
        "before": "lgie-eval-data/input_3.jpg",
        "after": "lgie-eval-data/output_3_2.png",
        "instruction": "The robotic arm puts the sushi on top of the omelette",
        "description": "The original sushi is put on top of the original omelette -- no hallucinated objects",
    },
    {
        "before": "lgie-eval-data/input_1.jpg",
        "after": "lgie-eval-data/output_1_hal.png",
        "instruction": "The robotic arm puts the blue spoon on top of the cube",
        "description": "A new banana is hallucinated at the far left (background object duplication -- opposite to where the existing banana is). A new red ball is hallucinated (new object -- appears out of nowhere). The blue spoon is structurally changed significantly (thus we can say that the new spoon structure is a hallucinated one), but is placed correctly on top of the cube as mentioned in the given manipulation instruction.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_69.png",
        "after": "lgie-eval-data/r4_gemini_output/out_69.png",
        "instruction": "pick up violet Allen key",
        "description": "A hallucinated violet allen key appeared being held by robotic arm.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_70.png",
        "after": "lgie-eval-data/r4_gemini_output/out_70.png",
        "instruction": "put lid on stove",
        "description": "Lid is being held by a hallucinated robotic arm on the right of the scene whereas the original robotic arm stays above the stove on the top left.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_71.png",
        "after": "lgie-eval-data/r4_gemini_output/out_71.png",
        "instruction": "take carrot off plate",
        "description": "A hallucinated green plate has occured near the pot in the scene.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_72.png",
        "after": "lgie-eval-data/r4_gemini_output/out_72.png",
        "instruction": "put knife on cutting board cardboard fence",
        "description": "Another knife is hallucinated by the model and is placed on cutting board.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_73.png",
        "after": "lgie-eval-data/r4_gemini_output/out_73.png",
        "instruction": "put pan in sink",
        "description": "Pan is put in sink -- no hallucinated objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_74.png",
        "after": "lgie-eval-data/r4_gemini_output/out_74.png",
        "instruction": "put cucumber in cup",
        "description": "Cucumber is put in cup -- no hallucinated objects",
    },
    {
        "before": "lgie-eval-data/original_image/oi_75.png",
        "after": "lgie-eval-data/r4_gemini_output/out_75.png",
        "instruction": "put bowl on plate cardboard fence",
        "description": "Another blue bowl has been hallucinated by the model and is placed on its side on blue plate.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_76.png",
        "after": "lgie-eval-data/r4_gemini_output/out_76.png",
        "instruction": "close oven",
        "description": "A white, flat object that appears to be a toy plate which is resting on the bottom rack of the oven has been hallucinated by the model",
    },
    {
        "before": "lgie-eval-data/original_image/oi_77.png",
        "after": "lgie-eval-data/r4_gemini_output/out_77.png",
        "instruction": "put detergent in sink",
        "description": "Model has hallucinated detergent fluid spilled in sink.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_78.png",
        "after": "lgie-eval-data/r4_gemini_output/out_78.png",
        "instruction": "put knife on cutting board cardboard fence",
        "description": "An hallucinated knife is placed on perimeter of cardboard fence.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_79.png",
        "after": "lgie-eval-data/r4_gemini_output/out_79.png",
        "instruction": "put broccoli in pot",
        "description": "Broccolis is put in pan -- no hallucinated objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_80.png",
        "after": "lgie-eval-data/r4_gemini_output/out_80.png",
        "instruction": "put clothes in laundry machine",
        "description": "Hallucinated red, green, blue clothes have appeared in the spinning tub of laundry machine.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_81.png",
        "after": "lgie-eval-data/r4_gemini_output/out_81.png",
        "instruction": "put corn in pan which is on stove",
        "description": "A hallucinated hand is placing an hallucinated corn in the scene.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_82.png",
        "after": "lgie-eval-data/r4_gemini_output/out_82.png",
        "instruction": "put sweet potato in pot which is in sink",
        "description": "The model misterpreted yellow bell pepper as sweet potato and hallucinated another yellow pepper in the pot.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_83.png",
        "after": "lgie-eval-data/r4_gemini_output/out_83.png",
        "instruction": "flip pot upright in sink",
        "description": "The model hallucinated a corn and a green vegetable inside the pot.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_84.png",
        "after": "lgie-eval-data/r4_gemini_output/out_84.png",
        "instruction": "put carrot on cutting board",
        "description": "Carrot has been put on cutting board -- no hallucinated objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_85.png",
        "after": "lgie-eval-data/r4_gemini_output/out_85.png",
        "instruction": "take sushi off plate",
        "description": "Sushi has been taken off the plate -- no hallucinated objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_86.png",
        "after": "lgie-eval-data/r4_gemini_output/out_86.png",
        "instruction": "put broccoli in pot or pan",
        "description": "A hallucinated hand is placing the broccoli in the pot.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_87.png",
        "after": "lgie-eval-data/r4_gemini_output/out_87.png",
        "instruction": "put sweet potato in pan which is on stove",
        "description": "Instead of red sweet potato model hallucinated a corn and put it in pan.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_88.png",
        "after": "lgie-eval-data/r4_gemini_output/out_88.png",
        "instruction": "pick up blue pen and put into drawer",
        "description": "Model has hallucinated an extra robotic arm and an extra blue pen.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_89.png",
        "after": "lgie-eval-data/r4_gemini_output/out_89.png",
        "instruction": "flip pot upright in sink",
        "description": "Pot is upright in sink -- no hallucinated objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_90.png",
        "after": "lgie-eval-data/r4_gemini_output/out_90.png",
        "instruction": "pick up pot from sink",
        "description": "Pot is raised a little -- no hallucinated objects.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_91.png",
        "after": "lgie-eval-data/r4_gemini_output/out_91.png",
        "instruction": "put corn in pan which is on stove",
        "description": "A hallucinated corn has appeared in the pan.",
    },
]
for tc in r4_tests:
    tc["r_script"] = "R4"

# ----- R5-6-7: Object Change Detection (color, texture, shape) -----
r567_tests = [
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_2.png",
        "instruction": "The robotic arm puts the corn into the pot",
        "description": "The corn is initially half but becomes a full corn in the final frame -- shape change. The robotic arm also changes shape across frames.",
    },
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_3.png",
        "instruction": "The robotic arm puts the corn into the pot",
        "description": "No color, texture or shape changes occur in any of the objects in the scene. Some objects might have become partially/fully occluded in the final/edited scene but that's not to be counted as color/texture/shape change.",
    },
    {
        "before": "lgie-eval-data/input_5.jpg",
        "after": "lgie-eval-data/output_5_4.png",
        "instruction": "The robotic arm puts the steel pot on top of the stove",
        "description": "No color, texture or shape changes occur in any of the objects in the scene. Some objects (like the stove) might have become partially/fully occluded in the final/edited scene but that's not to be counted as color/texture/shape change. Some objects (like the orange slices) might have appeared newly in the final/edited frame -- but this is not to be detected as color/texture/shape change since we do not know how these objects looked like in the first frame",
    },
    {
        "before": "lgie-eval-data/input_1.jpg",
        "after": "lgie-eval-data/output_1_1.png",
        "instruction": "The robotic arm puts the cucumber on top of the blue cube",
        "description": "No background changes -- as per the instruction",
    },
    {
        "before": "lgie-eval-data/input_1.jpg",
        "after": "lgie-eval-data/output_1_2.png",
        "instruction": "The robotic arm puts the cucumber on top of the blue cube",
        "description": "The blue spoon no longer has the grey scoop -- it's completely blue in the final frame. Almost every object changed shape. Surfaces changed texture.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_92.png",
        "after": "lgie-eval-data/r567_gemini_output/out_92.png",
        "instruction": "open microwave",
        "description": "No background changes -- as per the instruction",
    },
    {
        "before": "lgie-eval-data/original_image/oi_93.png",
        "after": "lgie-eval-data/r567_gemini_output/out_93.png",
        "instruction": "put eggplant in pot or pan",
        "description": "The eggplant is placed in pot but it's texture is altered.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_94.png",
        "after": "lgie-eval-data/r567_gemini_output/out_94.png",
        "instruction": "take carrot off plate",
        "description": "Instruction is followed but kitchen top's surface texture is altered.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_95.png",
        "after": "lgie-eval-data/r567_gemini_output/out_95.png",
        "instruction": "open small4fbox flaps",
        "description": "Box flaps are open and box shape is dramatically wider than before.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_96.png",
        "after": "lgie-eval-data/r567_gemini_output/out_96.png",
        "instruction": "put clothes in laundry machine",
        "description": "Laundry machine texture is unexpectedly altered to be more shiny and reflective, along with it's color which has become greenish.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_97.png",
        "after": "lgie-eval-data/r567_gemini_output/out_97.png",
        "instruction": "put spatula on cutting board",
        "description": "Spatula has been placed on cutting board but table's shape is altered and is round in shape.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_98.png",
        "after": "lgie-eval-data/r567_gemini_output/out_98.png",
        "instruction": "take carrot off plate",
        "description": "Instruction isn't followed, and scene has become brighter, but no background or foreground objects' change -- as per the instruction",
    },
    {
        "before": "lgie-eval-data/original_image/oi_99.png",
        "after": "lgie-eval-data/r567_gemini_output/out_99.png",
        "instruction": "open microwave",
        "description": "Viewing angle has changed, but no background or foreground objects' change -- as per the instruction.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_100.png",
        "after": "lgie-eval-data/r567_gemini_output/out_100.png",
        "instruction": "take carrot off plate",
        "description": "Kitchen top's surface color is clearly changed.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_101.png",
        "after": "lgie-eval-data/r567_gemini_output/out_101.png",
        "instruction": "close brown1fbox flap",
        "description": "The background surface is unexpectedly altered and looks like cardboard texture.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_102.png",
        "after": "lgie-eval-data/r567_gemini_output/out_102.png",
        "instruction": "put spoon on plate",
        "description": "Stationery plate has changed color and has become white from gray.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_103.png",
        "after": "lgie-eval-data/r567_gemini_output/out_103.png",
        "instruction": "flip pot upright in sink",
        "description": "Instruction is adhered and image became brighter but no background or foreground objects' change -- as per the instruction.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_104.png",
        "after": "lgie-eval-data/r567_gemini_output/out_104.png",
        "instruction": "flip pot upright in sink",
        "description": "Kitchen surface's texture is unexpectedly altered.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_105.png",
        "after": "lgie-eval-data/r567_gemini_output/out_105.png",
        "instruction": "open white1fbox flap",
        "description": "The table's texture has noticeably become blurry.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_106.png",
        "after": "lgie-eval-data/r567_gemini_output/out_106.png",
        "instruction": "right pepper shaker",
        "description": "Everything including the surface has become noisy, surface has also become blurry.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_107.png",
        "after": "lgie-eval-data/r567_gemini_output/out_107.png",
        "instruction": "put banana in pot cardboard fence",
        "description": "Table color has changed and texture is gone",
    },
    {
        "before": "lgie-eval-data/original_image/oi_107.png",
        "after": "lgie-eval-data/r567_gemini_output/out_107_1.png",
        "instruction": "put banana in pot cardboard fence",
        "description": "Table edges have been warped",
    },
    {
        "before": "lgie-eval-data/original_image/oi_107.png",
        "after": "lgie-eval-data/r567_gemini_output/out_107_2.png",
        "instruction": "put banana in pot cardboard fence",
        "description": "Red strawberry and orange carrot shapes have changed.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_110.png",
        "after": "lgie-eval-data/r567_gemini_output/out_110.png",
        "instruction": "put carrot on plate",
        "description": "Lighting hue turned purple-ish.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_111.png",
        "after": "lgie-eval-data/r567_gemini_output/out_111.png",
        "instruction": "put bowl on plate cardboard fence",
        "description": "Unexpected lighting hue shift detected.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_112.png",
        "after": "lgie-eval-data/r567_gemini_output/out_112.png",
        "instruction": "put knife on cutting board cardboard fence",
        "description": "Background wall has changed color to green from white.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_113.png",
        "after": "lgie-eval-data/r567_gemini_output/out_113.png",
        "instruction": "turn faucet front to left",
        "description": "Hallucinated objects have appeared but no background or foreground objects' change -- as per the instruction",
    },
    {
        "before": "lgie-eval-data/original_image/oi_114.png",
        "after": "lgie-eval-data/r567_gemini_output/out_114.png",
        "instruction": "turn lever vertical to front",
        "description": "The objects: orange pumpkin and yellow bell pepper have been distorted.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_114.png",
        "after": "lgie-eval-data/r567_gemini_output/out_114_1.png",
        "instruction": "turn lever vertical to front",
        "description": "Hallucinated pumpkin has appeared but yellow bell pepper has been distorted, it's shape seems enlarged.",
    },
]
for tc in r567_tests:
    tc["r_script"] = "R5-6-7"

# ----- R8-9: Lighting Changes -----
r89_tests = [
    {
        "before": "lgie-eval-data/input_3.jpg",
        "after": "lgie-eval-data/output_3_1.png",
        "instruction": "The robotic arm puts the sushi on top of the omelette",
        "description": "Hallucinations are there, but no lighting changes",
    },
    {
        "before": "lgie-eval-data/input_3.jpg",
        "after": "lgie-eval-data/output_3_2.png",
        "instruction": "The robotic arm puts the omelette on top of the cabbage",
        "description": "Instruction mismatch, but no lighting changes",
    },
    {
        "before": "lgie-eval-data/original_image/oi_115.png",
        "after": "lgie-eval-data/r89_gemini_output/out_115.png",
        "instruction": "take broccoli out of pan",
        "description": "Instruction followed but unexpectedly scene became noticeably brighter",
    },
    {
        "before": "lgie-eval-data/original_image/oi_116.png",
        "after": "lgie-eval-data/r89_gemini_output/out_116.png",
        "instruction": "take lid off pot or pan",
        "description": "Instruction followed but unexpectedly scene became significantly darker",
    },
    {
        "before": "lgie-eval-data/original_image/oi_117.png",
        "after": "lgie-eval-data/r89_gemini_output/out_117.png",
        "instruction": "turn lever vertical to front",
        "description": "Instruction mismatch, and scene became dimmer.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_118.png",
        "after": "lgie-eval-data/r89_gemini_output/out_118.png",
        "instruction": "put carrot on plate",
        "description": "Instruction mismatch, but no lighting changes",
    },
    {
        "before": "lgie-eval-data/original_image/oi_119.png",
        "after": "lgie-eval-data/r89_gemini_output/out_119.png",
        "instruction": "put banana in pot cardboard fence",
        "description": "Perfect edit -- Instruction followed, and no lighting changes",
    },
    {
        "before": "lgie-eval-data/original_image/oi_120.png",
        "after": "lgie-eval-data/r89_gemini_output/out_120.png",
        "instruction": "put corn in pot which is in sink",
        "description": "Instruction followed, but no environment lighting becamer dimmer",
    },
    {
        "before": "lgie-eval-data/original_image/oi_121.png",
        "after": "lgie-eval-data/r89_gemini_output/out_121.png",
        "instruction": "pour almonds in pot",
        "description": "Instruction followes, but scene became darker.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_122.png",
        "after": "lgie-eval-data/r89_gemini_output/out_122.png",
        "instruction": "put pepper in pot or pan",
        "description": "Lighting became noticeably brighter and more illuminated.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_123.png",
        "after": "lgie-eval-data/r89_gemini_output/out_123.png",
        "instruction": "put broccoli in pot",
        "description": "Instruction followed, but Lighting became dramatically brighter and more illuminated.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_124.png",
        "after": "lgie-eval-data/r89_gemini_output/out_124.png",
        "instruction": "put broccoli in pot",
        "description": "Instruction mismatch, and lighting brighter.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_125.png",
        "after": "lgie-eval-data/r89_gemini_output/out_125.png",
        "instruction": "open small4fbox flaps",
        "description": "Instruction mismatch, but no lighting changes",
    },
    {
        "before": "lgie-eval-data/original_image/oi_126.png",
        "after": "lgie-eval-data/r89_gemini_output/out_126.png",
        "instruction": "close large4fbox flaps",
        "description": "Instruction mismatch, but no lighting changes",
    },
    {
        "before": "lgie-eval-data/original_image/oi_127.png",
        "after": "lgie-eval-data/r89_gemini_output/out_127.png",
        "instruction": "fold cloth in half",
        "description": "Instruction mismatch, and lighting became dramatically bright.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_128.png",
        "after": "lgie-eval-data/r89_gemini_output/out_128.png",
        "instruction": "close brown1fbox flap",
        "description": "Instruction mismatch, and lighting became dramatically bright.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_129.png",
        "after": "lgie-eval-data/r89_gemini_output/out_129.png",
        "instruction": "flip pot upright in sink",
        "description": "Instruction mismatch, and lighting became dramatically bright.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_130.png",
        "after": "lgie-eval-data/r89_gemini_output/out_130.png",
        "instruction": "put bowl on plate",
        "description": "Instruction mismatch, and lighting became noticeably brighter.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_131.png",
        "after": "lgie-eval-data/r89_gemini_output/out_131.png",
        "instruction": "pick up scissors and put into drawer",
        "description": "Instruction mismatch, and lighting became dimmer.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_132.png",
        "after": "lgie-eval-data/r89_gemini_output/out_132.png",
        "instruction": "turn faucet front to left",
        "description": "Instruction mismatch, and lighting became noticeably darker.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_133.png",
        "after": "lgie-eval-data/r89_gemini_output/out_133.png",
        "instruction": "put lid on pot or pan",
        "description": "Instruction mismatch, and lighting became extremely dim.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_134.png",
        "after": "lgie-eval-data/r89_gemini_output/out_134.png",
        "instruction": "put lid on pot",
        "description": "Instruction mismatch, and lighting became extremely dim.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_135.png",
        "after": "lgie-eval-data/r89_gemini_output/out_135.png",
        "instruction": "open oven",
        "description": "Instruction mismatch, and lighting became dimmer..",
    },
    {
        "before": "lgie-eval-data/original_image/oi_136.png",
        "after": "lgie-eval-data/r89_gemini_output/out_136.png",
        "instruction": "pick up pan from stove",
        "description": "Instruction mismatch, and lighting became dramatically bright.",
    },
    {
        "before": "lgie-eval-data/original_image/oi_137.png",
        "after": "lgie-eval-data/r89_gemini_output/out_137.png",
        "instruction": "turn lever vertical to front",
        "description": "Instruction mismatch, and lighting became noticeably bright.",
    },
]
for tc in r89_tests:
    tc["r_script"] = "R8-9"

# ----- R10-11-12: Camera Changes -----
r101112_tests = [
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_1.png", "instruction": "The robotic arm puts the sushi on top of the omelette", "description": "Hallucinations are there, but no unrelated camera changes"},
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_2.png", "instruction": "The robotic arm puts the omelette on top of the cabbage", "description": "Instruction mismatch, but no unrelated camera changes"},
    {"before": "lgie-eval-data/input_5.png", "after": "lgie-eval-data/ft_5.png", "instruction": "The robotic arm picks up glue and puts it into the drawer", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_6.png", "after": "lgie-eval-data/ft_6.png", "instruction": "The robotic arm puts the broccoli in pot or pan", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_7.png", "after": "lgie-eval-data/ft_7.png", "instruction": "The robotic arm puts the banana in colander", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_8.png", "after": "lgie-eval-data/ft_8.png", "instruction": "The robotic arm takes the banana out of the colander", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_9.png", "after": "lgie-eval-data/ft_9.png", "instruction": "The robotic arm takes the broccoli out of the pan", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_10.png", "after": "lgie-eval-data/ft_10.png", "instruction": "The robotic arm puts the cucumber in the cup", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_11.png", "after": "lgie-eval-data/ft_11.png", "instruction": "The robotic arm puts the small spoon from basked in the tray", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_12.png", "after": "lgie-eval-data/ft_12.png", "instruction": "The robotic arm closes the brown box's flap", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_13.png", "after": "lgie-eval-data/ft_13.png", "instruction": "The robotic arm puts the corn in the pan which is on the stove", "description": "Zoomed out"},
    {"before": "lgie-eval-data/input_14.png", "after": "lgie-eval-data/ft_14.png", "instruction": "The robotic arm picks up pan from the stove", "description": "Zoomed out"},
    {"before": "lgie-eval-data/original_image/oi_138.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_138.png", "instruction": "put banana in pot cardboard fence", "description": "Instruction followed, but unexpectedly camera perspective is altered to be a low-angle view."},
    {"before": "lgie-eval-data/original_image/oi_139.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_139.png", "instruction": "take sushi off plate", "description": "Instruction followed, but unexpectedly camera perspective is altered to be a high-angle top-down view."},
    {"before": "lgie-eval-data/original_image/oi_140.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_140.png", "instruction": "flip pot upright in sink", "description": "Instruction followed, but unexpectedly camera perspective is altered to be a oblique side-angle view."},
    {"before": "lgie-eval-data/original_image/oi_141.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_141.png", "instruction": "open white1fbox flap", "description": "Instruction match, but camera seems to have zoomed-in to white-box as red plate near the right edge is out of frame."},
    {"before": "lgie-eval-data/original_image/oi_142.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_142.png", "instruction": "take carrot off plate", "description": "Instruction followed, but camera perspective seems to be changed from right high-angle to left high-angle, and camera has zoomed-out as more objects are in view and field-of-view is wider."},
    {"before": "lgie-eval-data/original_image/oi_143.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_143.png", "instruction": "pour almonds in pot", "description": "Instruction followed, but camera seems to have zoomed-in as field-of-view has narrowed."},
    {"before": "lgie-eval-data/original_image/oi_144.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_144_1.png", "instruction": "pick up bowl and put in small4fbox", "description": "Instruction mismatch, and camera seems to zoomed-in from right as part of banana is out-of-frame. Also the blue bowl and sushi is blacked-out (cropped)."},
    {"before": "lgie-eval-data/original_image/oi_145.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_145_2.png", "instruction": "put green squash in pot or pan", "description": "Camera zoomed-in on the pan, and the top of the top-most pot has been blacked-out (cropped)."},
    {"before": "lgie-eval-data/original_image/oi_146.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_146_1.png", "instruction": "pick up pan from stove", "description": "Instruction followed, but camera zoomed-in on the pan cutting frame from left, bottom, and right edge; camera also seems to have altered perspective to be lower in position than before; also the whole pan above the stove has been blacked-out (cropped) along with the top of pot which is placed in kitchen sink."},
    {"before": "lgie-eval-data/original_image/oi_147.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_147.png", "instruction": "put broccoli in pot", "description": "Instruction mismatch, but no unrelated camera changes"},
    {"before": "lgie-eval-data/original_image/oi_148.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_148.png", "instruction": "flip pot upright in sink", "description": "Instruction mismatch, but no unrelated camera changes"},
    {"before": "lgie-eval-data/original_image/oi_149.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_149.png", "instruction": "pick up scissors and put into drawer", "description": "Instruction mismatch, but no unrelated camera changes"},
    {"before": "lgie-eval-data/original_image/oi_150.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_150.png", "instruction": "close fridge", "description": "Camera perspective altered and camera seems to have zoomed-out enlarging the field-of-view."},
    {"before": "lgie-eval-data/original_image/oi_151.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_151.png", "instruction": "turn faucet front to left", "description": "Camera zoomed-in narrowing the field-of-view."},
    {"before": "lgie-eval-data/original_image/oi_152.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_152.png", "instruction": "put sweet potato in pot which is in sink", "description": "Camera has zoomed-out significantly showing the whole table."},
    {"before": "lgie-eval-data/original_image/oi_153.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_153.png", "instruction": "take broccoli out of pan", "description": "Camera has noticeably zoomed-out as faucet in the right of scene is now in frame."},
    {"before": "lgie-eval-data/original_image/oi_154.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_154.png", "instruction": "pour almonds in pot", "description": "Instruction followed, but camera has zoomed-in very close to the pot."},
    {"before": "lgie-eval-data/original_image/oi_155.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_155.png", "instruction": "put lid on pot or pan", "description": "Instruction followed, but camera's perspective seem to be altered to be a low, side-on perspective."},
    {"before": "lgie-eval-data/original_image/oi_156.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_156.png", "instruction": "pick up bowl and put in small4fbox", "description": "Camera's perspective have been altered and looks like high-angle, slightly tilted view"},
    {"before": "lgie-eval-data/original_image/oi_157.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_157.png", "instruction": "put lid on pot", "description": "Along with some hallucinations camera's perspective have been altered to be steep low-angle view."},
    {"before": "lgie-eval-data/original_image/oi_158.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_158.png", "instruction": "put knife on cutting board cardboard fence", "description": "Camera perspective have been altered to an angled side perspective, as if the camera is positioned slightly to the left of the setup and tilted downward at a shallow angle."},
    {"before": "lgie-eval-data/original_image/oi_159.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_159.png", "instruction": "flip pot upright in sink", "description": "Hallucinations occured, but no unrelated camera changes."},
    {"before": "lgie-eval-data/original_image/oi_160.png", "after": "lgie-eval-data/r10-11-12_gemini_output/out_160.png", "instruction": "put small spoon from basket to tray", "description": "Perfect edit -- Instruction followed and no unrelated camera changes."},
]
for tc in r101112_tests:
    tc["r_script"] = "R10-11-12"

# ----- R13: Photorealism -----
r13_tests = [
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_1.png", "instruction": "", "description": "photorealistic edit"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_2.png", "instruction": "", "description": "Edited with heavy smoothing -- looks artificial -> not photorealistic"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_3.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_4.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_6.png", "instruction": "", "description": "Edited with added light, updated brush, image is twinkling -> not photorealistic"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_7.png", "instruction": "", "description": "Image is cartoonish"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_9.png", "instruction": "", "description": "Image is plasticky"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_8.png", "instruction": "", "description": "Image is plasticky"},
    {"before": "lgie-eval-data/input_1.jpg", "after": "lgie-eval-data/output_1_10.png", "instruction": "", "description": "Image is plasticky"},
    {"before": "lgie-eval-data/input_2.jpg", "after": "lgie-eval-data/output_2.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_2.jpg", "after": "lgie-eval-data/output_3_1.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_2.jpg", "after": "lgie-eval-data/output_3_2.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_2.jpg", "after": "lgie-eval-data/output_3_3.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_2.jpg", "after": "lgie-eval-data/output_3_4.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_4.jpg", "after": "lgie-eval-data/output_4.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_4.jpg", "after": "lgie-eval-data/output_4_1.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/input_4.jpg", "after": "lgie-eval-data/output_4_2.png", "instruction": "", "description": "Realistic"},
    {"before": "lgie-eval-data/original_image/oi_161.png", "after": "lgie-eval-data/r13_gemini_output/out_161.png", "instruction": "put lid on pot", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_162.png", "after": "lgie-eval-data/r13_gemini_output/out_162.png", "instruction": "put lid on stove", "description": "The edit looks like an illustration with simplified forms and textureless objects -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_163.png", "after": "lgie-eval-data/r13_gemini_output/out_163.png", "instruction": "put small spoon from basket to tray", "description": "Edit has become a sketch -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_164.png", "after": "lgie-eval-data/r13_gemini_output/out_164.png", "instruction": "put lid on pot", "description": "Edited with simplified geometry, non-physical lighting, and artistic color choices -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_165.png", "after": "lgie-eval-data/r13_gemini_output/out_165.png", "instruction": "put pear in bowl", "description": "Edited with heavy smoothing -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_166.png", "after": "lgie-eval-data/r13_gemini_output/out_166.png", "instruction": "take lid off pot or pan", "description": "Not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_167.png", "after": "lgie-eval-data/r13_gemini_output/out_167.png", "instruction": "lever vertical to front", "description": "Edited flat or semi-flat colors, bold outlines, simplified shapes, and minimal shading -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_168.png", "after": "lgie-eval-data/r13_gemini_output/out_168.png", "instruction": "put lid on stove", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_169.png", "after": "lgie-eval-data/r13_gemini_output/out_169.png", "instruction": "put pot in sink", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_170.png", "after": "lgie-eval-data/r13_gemini_output/out_170.png", "instruction": "flip pot upright in sink", "description": "Edited with thick oil paint texture, visible layered brush strokes -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_171.png", "after": "lgie-eval-data/r13_gemini_output/out_171.png", "instruction": "fold cloth in half", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_172.png", "after": "lgie-eval-data/r13_gemini_output/out_172.png", "instruction": "close microwave", "description": "Edited in style of acrylic brush strokes, opaque paint layers, it has visible directional strokes, and rough edges -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_174.png", "after": "lgie-eval-data/r13_gemini_output/out_174.png", "instruction": "put banana on plate", "description": "Edited with pastel chalk style, with grainy pigment texture -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_175.png", "after": "lgie-eval-data/r13_gemini_output/out_175.png", "instruction": "put carrot on plate", "description": "Edited in gouache painting style -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_176.png", "after": "lgie-eval-data/r13_gemini_output/out_176.png", "instruction": "put bowl on plate cardboard fence", "description": "Edited as a charcoal sketch, with rough strokes, tonal shading, smudges, and high contrast linework -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_177.png", "after": "lgie-eval-data/r13_gemini_output/out_177.png", "instruction": "put broccoli in pot", "description": "Edited using flat saturated colors, visible marker strokes -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_178.png", "after": "lgie-eval-data/r13_gemini_output/out_178.png", "instruction": "close microwave", "description": "Edited with with a crayon texture, waxy strokes, uneven color fill -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_181.png", "after": "lgie-eval-data/r13_gemini_output/out_181.png", "instruction": "put banana in colander", "description": "Edited with visible linework -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_182.png", "after": "lgie-eval-data/r13_gemini_output/out_182.png", "instruction": "close large4fbox flaps", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_183.png", "after": "lgie-eval-data/r13_gemini_output/out_183.png", "instruction": "close drawer of box", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_184.png", "after": "lgie-eval-data/r13_gemini_output/out_184.png", "instruction": "take clothes out of laundry machine", "description": "Edited scene with flattened shapes, removing depth cues and realistic volume -> not photorealistic."},
    {"before": "lgie-eval-data/original_image/oi_185.png", "after": "lgie-eval-data/r13_gemini_output/out_185.png", "instruction": "take sushi out of pan", "description": "Photorealistic edit"},
    {"before": "lgie-eval-data/original_image/oi_186.png", "after": "lgie-eval-data/r13_gemini_output/out_186.png", "instruction": "put spatula on cutting board", "description": "Edited image with posterized shapes, reducing detail into discrete color regions with sharp transitions -> not photorealistic."},
]
for tc in r13_tests:
    tc["r_script"] = "R13"

# ----- R14-15: Instruction Adherence -----
r1415_tests = [
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_2.png", "instruction": "The robotic arm puts the sushi on top of the omelette", "description": "Sushi is the correct source, and omelette is the correct destination"},
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_2.png", "instruction": "The robotic arm puts the omelette on top of the cabbage", "description": "Source is sushi instead of omelette and destination is omelette instead of cabbage"},
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_2.png", "instruction": "The robotic arm puts the sushi on top of the cabbage", "description": "Sushi is the correct source, but destination is omelette instead of cabbage"},
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_2.png", "instruction": "The robotic arm puts the cabbage on top of the omelette", "description": "Omelette is the correct destination, but source is sushi instead of cabbage"},
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/input_3.jpg", "instruction": "The robotic arm puts the sushi on top of the omelette", "description": "Input and output are the same image, no manipulation happened"},
    {"before": "lgie-eval-data/input_3.jpg", "after": "lgie-eval-data/output_3_3.png", "instruction": "The robotic arm puts the sushi on top of the omelette", "description": "Extra omlette added, omlette is put on top of cabbage here."},
    {"before": "lgie-eval-data/input_5.png", "after": "lgie-eval-data/gt_5.png", "instruction": "The robotic arm picks up glue and puts it into the drawer", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_6.png", "after": "lgie-eval-data/gt_6.png", "instruction": "The robotic arm puts the broccoli in pot or pan", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_7.png", "after": "lgie-eval-data/gt_7.png", "instruction": "The robotic arm puts the banana in colander", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_8.png", "after": "lgie-eval-data/gt_8.png", "instruction": "The robotic arm takes the banana out of the colander", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_9.png", "after": "lgie-eval-data/gt_9.png", "instruction": "The robotic arm takes the broccoli out of the pan", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_10.png", "after": "lgie-eval-data/gt_10.png", "instruction": "The robotic arm puts the cucumber in the cup", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_11.png", "after": "lgie-eval-data/gt_11.png", "instruction": "The robotic arm puts the small spoon from basked in the tray", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_12.png", "after": "lgie-eval-data/gt_12.png", "instruction": "The robotic arm closes the brown box's flap", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_13.png", "after": "lgie-eval-data/gt_13.png", "instruction": "The robotic arm puts the corn in the pan which is on the stove", "description": "Ground truth."},
    {"before": "lgie-eval-data/input_14.png", "after": "lgie-eval-data/gt_14.png", "instruction": "The robotic arm picks up pan from the stove", "description": "Ground truth."},
    {"before": "lgie-eval-data/original_image/oi_184.png", "after": "lgie-eval-data/r14-15_gemini_output/out_184.png", "instruction": "take clothes out of laundry machine", "description": "Clothes are the correct source, and they have vanished from the image, but they are to be placed on surface or basket."},
    {"before": "lgie-eval-data/original_image/oi_185.png", "after": "lgie-eval-data/r14-15_gemini_output/out_185.png", "instruction": "take sushi out of pan", "description": "Sushi is the correct source, and kitchen surface is the desired destination but sushi has vanished from the scene"},
    {"before": "lgie-eval-data/original_image/oi_186.png", "after": "lgie-eval-data/r14-15_gemini_output/out_186.png", "instruction": "put spatula on cutting board", "description": "Spatuala is the correct source, but it isn't put on cutting board, only moved on the surface."},
    {"before": "lgie-eval-data/original_image/oi_187.png", "after": "lgie-eval-data/r14-15_gemini_output/out_187.png", "instruction": "put sweet potato in pot which is in sink", "description": "Red sweet potato is the desired source, and pot in kitchen sink is the correct destination but a new sweet potato has been hallucinated and placed in the pot."},
    {"before": "lgie-eval-data/original_image/oi_188.png", "after": "lgie-eval-data/r14-15_gemini_output/out_188.png", "instruction": "take carrot off plate", "description": "Carrot is the correct source, and surface is the correct destination when no explicit destination is mentioned."},
    {"before": "lgie-eval-data/original_image/oi_189.png", "after": "lgie-eval-data/r14-15_gemini_output/out_189.png", "instruction": "put broccoli in pot", "description": "Brocolli is the correct source, and Metal pot is the correct destination"},
    {"before": "lgie-eval-data/original_image/oi_190.png", "after": "lgie-eval-data/r14-15_gemini_output/out_190.png", "instruction": "take broccoli out of pan", "description": "Broccoli is the correct source, but it has vanished from the scene, it should be placed somewhere in the scene if no explicit destination is mentioned."},
    {"before": "lgie-eval-data/original_image/oi_191.png", "after": "lgie-eval-data/r14-15_gemini_output/out_191.png", "instruction": "open oven", "description": "Oven is the correct source, and is in open state -> model correctly followed the instruction."},
    {"before": "lgie-eval-data/original_image/oi_192.png", "after": "lgie-eval-data/r14-15_gemini_output/out_192.png", "instruction": "take carrot off plate", "description": "Carrot is the correct source, but it has moved out of scene."},
    {"before": "lgie-eval-data/original_image/oi_193.png", "after": "lgie-eval-data/r14-15_gemini_output/out_193.png", "instruction": "open book", "description": "Book is opened but omelette which should be below the book is on top of it."},
    {"before": "lgie-eval-data/original_image/oi_194.png", "after": "lgie-eval-data/r14-15_gemini_output/out_194.png", "instruction": "put red bottle in sink", "description": "Red chilly has been moved and removed from the scene but it is not the correct source neither the destination is kitchen sink."},
    {"before": "lgie-eval-data/original_image/oi_195.png", "after": "lgie-eval-data/r14-15_gemini_output/out_195.png", "instruction": "put knife in pot cardboard fence", "description": "Knife is the correct source, but pot inside the cardboard fence is the correct destination not the cardboard fence itself."},
    {"before": "lgie-eval-data/original_image/oi_196.png", "after": "lgie-eval-data/r14-15_gemini_output/out_196.png", "instruction": "put corn in pot which is in sink", "description": "Corn is the correct source, and pot inside the sink is the correct destination"},
    {"before": "lgie-eval-data/original_image/oi_197.png", "after": "lgie-eval-data/r14-15_gemini_output/out_197.png", "instruction": "turn lever vertical to front", "description": "Lever is unturned."},
    {"before": "lgie-eval-data/original_image/oi_198.png", "after": "lgie-eval-data/r14-15_gemini_output/out_198.png", "instruction": "put pepper in pan", "description": "Yellow bell pepper is not the correct source instead red chilly pepper is, though pan is the correct destination."},
    {"before": "lgie-eval-data/original_image/oi_199.png", "after": "lgie-eval-data/r14-15_gemini_output/out_199.png", "instruction": "put corn in pot which is in sink", "description": "Corn is the correct source, but pan is not the correct destination"},
    {"before": "lgie-eval-data/original_image/oi_200.png", "after": "lgie-eval-data/r14-15_gemini_output/out_200.png", "instruction": "take sushi out of pan", "description": "Sushi is the correct source, and is placed outside the pan which is correct destination."},
    {"before": "lgie-eval-data/original_image/oi_201.png", "after": "lgie-eval-data/r14-15_gemini_output/out_201.png", "instruction": "put spatula on cutting board", "description": "Spatula is the correct source, but is vanished from the scene."},
    {"before": "lgie-eval-data/original_image/oi_202.png", "after": "lgie-eval-data/r14-15_gemini_output/out_202.png", "instruction": "put knife in pot cardboard fence", "description": "Pot is placed on top of knife whereas knife should have been placed inside of pot."},
    {"before": "lgie-eval-data/original_image/oi_203.png", "after": "lgie-eval-data/r14-15_gemini_output/out_203.png", "instruction": "take broccoli out of pan", "description": "Pan has moved but brocolli is still within the pan."},
    {"before": "lgie-eval-data/original_image/oi_204.png", "after": "lgie-eval-data/r14-15_gemini_output/out_204.png", "instruction": "put sweet potato in pan which is on stove", "description": "Pan on the stove is the correct destination but carrot is not the correct source."},
    {"before": "lgie-eval-data/original_image/oi_205.png", "after": "lgie-eval-data/r14-15_gemini_output/out_205.png", "instruction": "put broccoli in pot or pan", "description": "Brocolli is the correct source, and it has moved but not placed in the pot or pan."},
    {"before": "lgie-eval-data/original_image/oi_206.png", "after": "lgie-eval-data/r14-15_gemini_output/out_206.png", "instruction": "put knife on cutting board", "description": "Knife is the correct source, and cutting board is the correct destination"},
]
for tc in r1415_tests:
    tc["r_script"] = "R14-15"

# =====================================================================
# Combine all test cases in order: R1, R2, R3, R4, R5-6-7, R8-9, R10-11-12, R13, R14-15
# =====================================================================
all_test_cases = (r1_tests + r2_tests + r3_tests + r4_tests + 
                  r567_tests + r89_tests + r101112_tests + r13_tests + r1415_tests)

# =====================================================================
# Create data folder
# =====================================================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    manifest = []  # Track what was created
    warnings = []
    
    for idx, tc in enumerate(all_test_cases, start=1):
        subfolder = os.path.join(DATA_DIR, str(idx))
        os.makedirs(subfolder, exist_ok=True)
        
        # Copy before image
        before_copied = copy_image(tc["before"], subfolder, "original_image")
        if before_copied is None:
            warnings.append(f"Test {idx}: before image missing: {tc['before']}")
        
        # Copy after image
        after_copied = copy_image(tc["after"], subfolder, "edited_image")
        if after_copied is None:
            warnings.append(f"Test {idx}: after image missing: {tc['after']}")
        
        # Save instruction
        with open(os.path.join(subfolder, "edit_instruction.txt"), "w") as f:
            f.write(tc["instruction"])
        
        # Save description
        with open(os.path.join(subfolder, "test_description.txt"), "w") as f:
            f.write(tc["description"])
        
        manifest.append({
            "test_number": idx,
            "r_script": tc["r_script"],
            "before_src": tc["before"],
            "after_src": tc["after"],
            "instruction": tc["instruction"],
            "description": tc["description"],
        })
        
        print(f"  [{idx:3d}] {tc['r_script']:>10s} | {tc['description'][:60]}...")
    
    # Save manifest
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Created {len(all_test_cases)} test case subfolders in {DATA_DIR}")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("All images copied successfully!")
    print(f"Manifest saved to {os.path.join(DATA_DIR, 'manifest.json')}")

if __name__ == "__main__":
    main()
